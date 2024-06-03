import hashlib
import os
import tarfile
import urllib.request
import torch
from tqdm import tqdm
from supar.utils.common import bos, pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.transform import CoNLL
from supar.utils import Dataset

def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str_none(val):
    if val == 'None':
        return None
    else:
        return val


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print(f"Downloading {url} to {filepath} ...")
        with urllib.request.urlopen(url) as source, open(filepath, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                      unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        print(f"\nMD5 Chesksum {filepath} ...")
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print(f"File exists, skip downloading. ({filepath})")
    return filepath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar:
        os.remove(filepath)


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
    

def prepare_viterbi_BES(labels):
        '''
        for BES schema
        '''
        # -2 is 'I' and -1 is "O"
        label2id = {label:i for i,label in enumerate(labels)}
        strans = [-float('inf')] * (len(labels)+2)
        trans = [[-float('inf')] * (len(labels)+2) for _ in range((len(labels)+2))]
        B_idxs = []
        E_idxs = []
        S_idxs = []
        p_idxs = None # 在想是否还需要这个？ 只有srl围绕谓词吧
        permit_trans = set()
        pair_dict = {}
        for i, label in enumerate(labels):
            if(label.startswith('E-')):
                E_idxs.append(i)
            elif label.startswith('B-'):
                strans[i] = 0
                B_idxs.append(i)
                role = label[2:]
                corres_e_label = 'E-'+role
                corres_e_idx = label2id[corres_e_label]
                permit_trans.add((i, corres_e_idx))
            elif label.startswith('S-'):
                strans[i] = 0
                S_idxs.append(i)
            # else:
            elif label == '[prd]' or label == 'root':
                p_idxs = i
        # can strat with 'O'
        strans[-1] = 0

        # construct transition matrix

        # B-E pairs
        for x, y in permit_trans:
            trans[x][y] = 0

        # start from B-
        for i in B_idxs:
            trans[i][-2] = 0

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

        return torch.tensor(strans), torch.tensor(trans), B_idxs, E_idxs, S_idxs, pair_dict, p_idxs


def test_to_BES_graph_return_list(file_name):
    if isinstance(file_name, str):
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
    elif isinstance(file_name, list):
        datas = file_name

    new_sentence_lsts = []
    
    for data in datas:
        new_sentence_lst = []
        text = data['text']
        srl = data['srl']
        sorted_srl = sorted(srl, key=lambda x: x['position'][0])
        words = list(text)
        for num, word in enumerate(words, 1):
            new_line_lst = [str(num), word, '_', '_', '_', '_', '_', '_']
            new_sentence_lst.append(new_line_lst)
        
        arc_lsts = [[] for i in range(len(new_sentence_lst))]
        for pred in sorted_srl:
            arcs = [[] for i in range(len(new_sentence_lst))]

            pred_begin, pred_end = pred['position']
            arcs[pred_begin - 1].append((0, '[prd]'))

            for i in range(pred_begin + 1, pred_end + 1):
                if pred_begin + 1 == pred_end:
                    arcs[i - 1].append((pred_begin, 'S-V'))
                else:
                    if i == pred_begin + 1:
                        arcs[i - 1].append((pred_begin, 'B-V'))
                    elif i == pred_end:
                        arcs[i - 1].append((pred_begin, 'E-V'))
            
            for arg in pred['arguments']:
                arg_begin, arg_end = arg['position']
                for i in range(arg_begin, arg_end + 1):
                    if arg_begin == arg_end:
                        arcs[i - 1].append((pred_begin, 'S-'+arg['role']))
                    else:
                        if i == arg_begin:
                            arcs[i - 1].append((pred_begin, 'B-'+arg['role']))
                        elif i == arg_end:
                            arcs[i - 1].append((pred_begin, 'E-'+arg['role']))

            for i in range(len(arcs)):
                arc_lsts[i] += arcs[i]
        
        for i in range(len(arc_lsts)):
            arc_values = []
            for arc in arc_lsts[i]:
                head_idx = arc[0]
                label = arc[1]
                arc_values.append(str(head_idx)+':'+label)
            if(len(arc_values) > 0):
                new_sentence_lst[i] += ['|'.join(arc_values), '_']
            else:
                new_sentence_lst[i] += ['_', '_']
        new_sentence_lsts.append(new_sentence_lst)
    format_list = []
    for sentence_lst in new_sentence_lsts:
        for line_lst in sentence_lst:
            format_list.append('\t'.join(line_lst))
        format_list.append('')
    return format_list


def prepare(sentences):

    idx = 0
    all_sentence_spans = []

    all_spans = []
    all_preds = []
    for sentence in sentences:
        idx += 1
        this_sents_spans = []
        sentence_lst = []
        text = []
        ids, words, labels = sentence.values[0],sentence.values[1], sentence.values[8]
        for id, word, label in zip(ids, words, labels):
            sentence_lst.append([id, word, '_', '_', '_', '_', '_', '_', label, '_'])
        text = words
        num_words = len(sentence_lst)
        prd_map = {}  
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                    
                        arc_value[prd_map[head_idx] - 1].append(rel)
                arc_values.append(arc_value)

        re_prd_map = {}  
        for key, value in prd_map.items():
            re_prd_map[value] = key
        
        spans = []
        for p_num, p_id in re_prd_map.items():

            for i, word_arcs in enumerate(arc_values, 1):
                if len(word_arcs) >= p_num:
                    arc = word_arcs[p_num - 1]
                    if len(arc) != 0:
                        role = arc[0]
                        if role.startswith('B'):
                            start = i
                            pre_label = '-'.join(role.split('-')[1:])
                        elif role.startswith('S'):
                            start = i
                            end = i
                            label = '-'.join(role.split('-')[1:])
                            spans.append((p_id, start, end, label))
                            start = -1
                            end = -1
                        elif role.startswith('E'):
                            end = i
                            label = '-'.join(role.split('-')[1:])
                            if pre_label == label:
                                spans.append((p_id, start, end, label))
                            
                            start = -1 
                            end = -1
                            pre_label = None
        predToId = {}
        preds =[]
        
        for s in spans:
            p, begin, end, role = s
            if role == 'V':
                if p not in predToId:
                    predToId[p] = len(predToId)
                    preds.append({'pred': ''.join(text[p-1: end]), 'position': [p, end], 'arguments': []})
        for key, _ in prd_map.items():
            if key not in predToId:
                predToId[key] = len(predToId)
                preds.append({'pred': text[key - 1], 'position': [key, key], 'arguments': []})

        for s in spans:
            p, begin, end, role = s
            if role == 'V':
                continue
            if p not in predToId:
                continue
            else:
                index = predToId[p] 
            value = ''.join(text[begin-1: end])
            arg = {'value': value, 'position': [begin, end], 'role': role}
            if arg not in preds[index]['arguments']:
                preds[index]['arguments'].append(arg)
        all_preds.append(preds)
    return all_preds

def build_transform(tokenizer, train_data, fix_len=20):
    WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
    TAG, CHAR, LEMMA, BERT = None, None, None, None
    WORD = SubwordField(
        'words',
        pad=tokenizer.pad_token,
        unk=tokenizer.unk_token,
        bos=tokenizer.bos_token or tokenizer.cls_token,
        fix_len=fix_len,
        tokenize=tokenizer.tokenize)
    WORD.vocab = tokenizer.get_vocab()
    LABEL = ChartField('labels', fn=CoNLL.get_labels)
    transform = CoNLL(FORM=(WORD, CHAR, BERT),
                    LEMMA=LEMMA,
                    POS=TAG,
                    PHEAD=LABEL)
    data = Dataset(transform, train_data)
    LABEL.build(data)
    transform.PHEAD = LABEL
    return transform