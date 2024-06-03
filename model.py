import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import re
from utils.dataloader import data_generator
from torch.utils.data import DataLoader
import itertools
from supar.models.srl import VISemanticRoleLabelingModel
import json
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.transform import CoNLL
from supar.utils import Dataset
from utils.utils import test_to_BES_graph_return_list
from get_metric import process_label_with_aligner


# copy from supar
def prepare_viterbi_BES(vocab):
        '''
        for BES schema
        '''
        # -2 is 'I' and -1 is "O"
        strans = [-float('inf')] * (len(vocab)+2)
        trans = [[-float('inf')] * (len(vocab)+2) for _ in range((len(vocab)+2))]
        B_idxs = []
        E_idxs = []
        S_idxs = []
        permit_trans = set()
        pair_dict = {}
        for i, label in enumerate(vocab.itos):
            if(label.startswith('E-')):
                # strans[i] = -float('inf')  # cannot start with E-
                E_idxs.append(i)
            elif label.startswith('B-'):
                strans[i] = 0
                B_idxs.append(i)
                role = label[2:]
                corres_e_label = 'E-'+role
                corres_e_idx = vocab.stoi[corres_e_label]
                permit_trans.add((i, corres_e_idx))
            elif label.startswith('S-'):
                strans[i] = 0
                S_idxs.append(i)
            elif label == '[prd]':
                pass
        # can strat with 'O'
        strans[-1] = 0
        # label
        for x, y in permit_trans:
            trans[x][y] = 0
        # start from B-
        for i in B_idxs:
            trans[i][-2] = 0

            # for j in E_idxs:
            #     trans[i][j] = 0

        # start from E-
        for i in E_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0

        # start from S-
        for i in S_idxs:
            for j in B_idxs:
                trans[i][j] = 0
            for j in S_idxs:
                trans[i][j] = 0
            trans[i][-1] = 0

        # start from I
        for j in E_idxs:
            trans[-2][j] = 0
        trans[-2][-2] = 0

        # start from O
        for j in B_idxs:
            trans[-1][j] = 0
        for j in S_idxs:
            trans[-1][j] = 0
        trans[-1][-1] = 0

        for x, y in permit_trans:
            pair_dict[x] = y
            pair_dict[y] = x

        return torch.tensor(strans), torch.tensor(trans), B_idxs, E_idxs, S_idxs, pair_dict, vocab.stoi['[prd]']

def get_suda_features(batch_texts, batch_srl_labels, transform, n_buckets=1, update_steps=1):
    datas = []
    batch_size = 500
    for text, label in zip(batch_texts, batch_srl_labels):
        datas.append({'text': text, 'srl': label})
    format_datas = test_to_BES_graph_return_list(datas)
    transform.train()
    train = Dataset(transform, format_datas)
    train.build(batch_size//update_steps, n_buckets=n_buckets)
    pad_index = transform.FORM[0].pad_index
    return train.loader, pad_index
  

class E2EModel(nn.Module):

    def __init__(self, whiper_model, srl_config, processor, tokenizer, transform, aligner, is_e2e=False, is_use_gumbel_softmax_loss=False, is_remove_augment=False):
        super(E2EModel, self).__init__()
        
        self.whisper_model = whiper_model
        self.is_remove_augment = is_remove_augment
        self.tokenizer = tokenizer
        bert_vocab = self.tokenizer.get_vocab()
        self.srl_config = srl_config
        self.aligner = aligner
        self.transform = transform
        self.srl_model = VISemanticRoleLabelingModel(n_words=len(bert_vocab), n_labels=srl_config.srl_num_labels, bert=srl_config.srl_encoder_path, encoder='bert')
        
        self.processor = processor
        vocab = self.processor.tokenizer.get_vocab()
        self.is_e2e = is_e2e
        self.is_use_gumbel_softmax_loss = is_use_gumbel_softmax_loss
        self.alpha = 0.99
        if self.is_use_gumbel_softmax_loss:
            self.gumbel_loss = nn.CrossEntropyLoss()
        if self.is_e2e:
            datas = []
            with open('./mapping_data.json', 'r', encoding='utf-8') as f:
                for line in f:
                    datas.append(json.loads(line))
            self.mapping = torch.zeros((len(vocab), len(bert_vocab))).float()
            self.mapping.requires_grad = False
            
            for data in datas:
                bert_token_ids = data['bert_token_ids']
                whisper_token_ids = data['whisper_token_ids']
                for token in whisper_token_ids:
                    self.mapping[token, bert_token_ids] = 1

    def predict(self, input_features, decoder_input_ids, forced_decoder_ids, fix_len=20, gold_text=None):
        device = input_features.device
        decoded_preds = []
        if gold_text is None:
            trans_output = self.whisper_model.generate(
                            input_features=input_features,
                            decoder_input_ids=decoder_input_ids,
                            forced_decoder_ids=forced_decoder_ids,
                            max_new_tokens=512).cpu().numpy()
            
            decoded_preds = self.processor.tokenizer.batch_decode(trans_output, skip_special_tokens=True)
        else:
            decoded_preds = gold_text
        sentences = []
        for text in decoded_preds:
            words = list(text)
            for i, (word) in enumerate(words, 1):
                sentences.append('\t'.join([str(i), word, '_', '_', '_', '_', '_', '_', '_', '_']))
            sentences.append('')
        batch_size = 500
        self.transform.train()
        data = Dataset(self.transform, sentences)
        data.build(batch_size, 1)
        pad_index = self.transform.FORM[0].pad_index
        loader = data.loader
        preds = {'labels': []}

        strans, trans, B_idxs, E_idxs, S_idxs, pair_dict, prd_idx = prepare_viterbi_BES(self.transform.PHEAD.vocab)
        if(torch.cuda.is_available()):
            strans = strans.cuda()
            trans = trans.cuda()
        for words, *feats, labels in loader:
            words, labels = words.to(device), labels.to(device)
            word_mask = words.ne(pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            n_mask = mask[:, :, 0]
            lens = mask[:, 1].sum(-1).tolist()
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0]
            s_edge, s_sib, s_cop, s_grd, x = self.srl_model(words, feats, if_prd)
            s_edge, s_label = self.srl_model.loss(s_edge, s_sib, s_cop, s_grd,
                                           x, labels, mask, True)
            label_preds, p_num, con_p_num, p_label = self.srl_model.viterbi_decode_BES(s_edge, s_label, strans, trans, n_mask, mask, B_idxs, E_idxs, S_idxs, pair_dict, prd_idx)
            label_preds.masked_fill_(~mask, -1)
            label_preds = self.srl_model.fix_label_cft_BES(label_preds, B_idxs, E_idxs, S_idxs, pair_dict,prd_idx, p_label)
            preds['labels'].extend(chart[1:i, :i].tolist()
                                   for i, chart in zip(lens, label_preds))
            
        preds['labels'] = [
            CoNLL.build_relations(
                [[self.transform.PHEAD.vocab[i] if i >= 0 else None for i in row]
                 for row in chart]) for chart in preds['labels']
        ]

        for name, value in preds.items():
            setattr(data, name, value)
        return data.sentences, decoded_preds


    def get_srl_loss(self, text_list, srl_labels, device, bert_embeddings=None):
        loader, pad_index = get_suda_features(text_list, srl_labels, self.transform)
        loss = None

        for batch in loader:
            words, *feats, labels = batch
            words, labels = words.to(device), labels.to(device)
            word_mask = words.ne(pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            edges = labels.ge(0) & mask
            if_prd = edges[..., 0].contiguous()
            s_edge, s_sib, s_cop, s_grd, s_label = self.srl_model(words, feats, if_prd=if_prd, bert_embeddings=bert_embeddings)
            if loss is None:
                loss, _, _ = self.srl_model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
            else:
                tmp_loss, _, _ = self.srl_model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
                loss =loss + tmp_loss
        loss = loss / len(loader)
        return loss

    def forward(self, input_features, labels=None, text_list=None, srl_labels=None, mode='train'):
        torch.autograd.set_detect_anomaly(True) 
        device = input_features.device
        if self.is_e2e:
            _, lm_logits, _= self.whisper_model(input_features=input_features, labels=labels, return_dict=False)
        
        srl_loss = self.get_srl_loss(text_list, srl_labels, device)

        if not self.is_e2e and self.is_use_gumbel_softmax_loss:
            if self.is_use_asr_loss:
                _, new_lm_logits, _= self.whisper_model(input_features=input_features, labels=labels, return_dict=False)
                sample_lm_logits = F.gumbel_softmax(new_lm_logits, tau=self.srl_config.gumbel_tp, hard=True)
            else:
                
                
                sample_lm_logits = F.gumbel_softmax(lm_logits, tau=self.srl_config.gumbel_tp, hard=True)
            gumbel_loss = self.gumbel_loss(sample_lm_logits.view(-1, sample_lm_logits.shape[-1]), labels.view(-1))
            return {'loss': srl_loss + gumbel_loss}

        if self.is_e2e:

            batch_srl_data = []
            self.mapping = self.mapping.to(device)
            sample_lm_logits = F.gumbel_softmax(lm_logits, tau=self.srl_config.gumbel_tp, hard=True)
            gumbel_loss = None
            if self.is_use_gumbel_softmax_loss:
                gumbel_loss = self.gumbel_loss(sample_lm_logits.view(-1, sample_lm_logits.shape[-1]), labels.view(-1))
            logits = torch.argmax(sample_lm_logits, dim=-1)
            reduce_lm_logits = []
            special_tokens = self.processor.tokenizer.additional_special_tokens 
            special_tokens.extend(self.processor.tokenizer.unique_no_split_tokens)
            for batch_idx, logit in enumerate(logits):
                
                tran = self.processor.tokenizer.decode(logit, skip_special_tokens=False)
                pattern = r'({})|.'.format('|'.join(re.escape(token) for token in special_tokens))
                result = [match for match in re.finditer(pattern, tran)]
                result = [match.group(0) for match in result]
                no_special_tran = ''
                for t in result:
                    if t not in special_tokens:
                        no_special_tran += t
                bert_tokens = []
                length = []
                bert_tokens.append(101)
                length.append(-1)

                if len(no_special_tran) > self.srl_config.maxlen:
                    no_special_tran = no_special_tran[:self.srl_config.maxlen]
                if len(no_special_tran) == 0:
                    filter_notChinese = ''
                else:
                    bert_encode = self.tokenizer(list(no_special_tran), max_length=self.srl_config.maxlen, truncation=True, add_special_tokens=False)['input_ids']
                    filter_notChinese = ''
                    for i, token_id in enumerate(bert_encode):
                        if len(token_id) == 1:
                            bert_tokens.append(token_id[0])
                            filter_notChinese += no_special_tran[i]
                            length.append(len(self.processor.tokenizer.encode(no_special_tran[i], add_special_tokens=False)))
                        elif len(token_id) == 0:
                            bert_tokens.append(100)
                            filter_notChinese += no_special_tran[i]
                            length.append(len(self.processor.tokenizer.encode(no_special_tran[i], add_special_tokens=False)))

                if len(filter_notChinese) == 0:
                    filter_notChinese = ' '
                    bert_tokens.append(100)
                    length.append(-1)

                if self.is_remove_augment and filter_notChinese != text_list[batch_idx]:

                    return {'loss':  srl_loss}
                else: 
                    index = 0
                    new_t = []
                    length_index = 0
                    i = 0
                    result = ['[CLS]'] + result
                    while i < len(result):
                        r = result[i]
                        if r in special_tokens or r == '':
                            index += 1
                            i += 1
                        else:
                            if length_index >= len(length):
                                break
                            a = length[length_index]
                            bert_id = bert_tokens[length_index]
                                
                            if a == -1:
                                vector = np.zeros((1, self.mapping.size()[1]))
                                vector[0, bert_id] = 1
                                new_t.append(torch.from_numpy(vector).float().to(device))
                                i += 1
                            else:
                                temp = torch.sum(sample_lm_logits[batch_idx, index: index + a], dim=0)
                                temp = torch.matmul(temp, self.mapping).unsqueeze(0)
                                new_temp = torch.cat((temp[:, 0:bert_id], (temp[:, bert_id] - 1).unsqueeze(1), temp[:, bert_id+1:]), dim=1)
                                new_t.append(temp - new_temp)
                                index += a
                                del temp, new_temp
                                torch.cuda.empty_cache()
                                i += 1
                                
                            length_index += 1
                    while length_index < len(length):
                        a = length[length_index]
                        bert_id = bert_tokens[length_index]
                            
                        vector = np.zeros((1, self.mapping.size()[1]))

                        vector[0, bert_id] = 1
                        new_t.append(torch.from_numpy(vector).float().to(device))
                        length_index += 1
                    batch_srl_data.append(process_label_with_aligner(filter_notChinese, text_list[batch_idx], srl_labels[batch_idx], self.aligner))
                    reduce_lm_logits.append(torch.cat(new_t, dim=0).unsqueeze(0))
                    
            if len(reduce_lm_logits) > 1:

                max_length = max(tensor.shape[1] for tensor in reduce_lm_logits)

                padded_tensors = [torch.cat((tensor, torch.zeros(tensor.shape[0], max_length - tensor.shape[1], tensor.shape[2]).to(device)), dim=1)
                                if tensor.shape[1] < max_length
                                else tensor
                                for tensor in reduce_lm_logits]
                reduce_lm_logits = torch.cat(padded_tensors, dim=0)                
            else:
                reduce_lm_logits = torch.cat(reduce_lm_logits, dim=0)

            if reduce_lm_logits.shape[1] > self.srl_config.maxlen:
                reduce_lm_logits = reduce_lm_logits[:, :self.srl_config.maxlen,:]
            bert_embeddings = torch.matmul(reduce_lm_logits, self.srl_model.encoder.bert.embeddings.word_embeddings.weight)
            new_text_list, new_srl_labels = [], []
            for d in batch_srl_data:
            
                new_text_list.append(d['text'])
                new_srl_labels.append(d['srl'])

            e2e_loss = self.get_srl_loss(new_text_list, new_srl_labels, device, bert_embeddings)
            loss = e2e_loss
            loss = loss + srl_loss
            if gumbel_loss is not None:
                if filter_notChinese == ' ':
                    loss = 0*loss + 1*gumbel_loss
                else:
                    loss =(1-self.alpha)*loss + self.alpha*gumbel_loss
               
            return {"loss": loss}

        else:
            loss = self.get_srl_loss(text_list, srl_labels, device)
            return {'loss': loss}
