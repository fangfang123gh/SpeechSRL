import argparse
import functools
import gc
import os
import numpy as np
import torch
import json
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor,BertTokenizerFast

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments
from model import E2EModel
from peft import PeftModel
from fast_align_zms.build_new.force_align import Aligner
from get_metric import process_label_with_aligner


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("save_path",   type=str, default="",            help="保存结果的路径")
add_arg("test_data",   type=str, default="",            help="数据集的路径")
add_arg("bert_path",  type=str, default="", help="bert预训练模型的路径")
add_arg("base_path",  type=str, default="", help="asr base的模型路径")
add_arg("aligner_fwd_params_path",   type=str, default="",            help="aligner的fwd_param path")
add_arg("aligner_fwd_err_path",   type=str, default="",            help="aligner的fwd_err path")
add_arg("aligner_rev_params_path",   type=str, default="",            help="aligner的rev_param path")
add_arg("aligner_rev_err_path",   type=str, default="",            help="aligner的rev_err path")
add_arg("batch_size",  type=int, default=2,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=False,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=False,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",     type=str, default="wer",        choices=['cer', 'wer'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
aligner = Aligner(args.aligner_fwd_params_path, args.aligner_fwd_err_path, args.aligner_rev_params_path, args.aligner_rev_err_path)

# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
processor = WhisperProcessor.from_pretrained(args.base_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids()
# 获取模型
whisper_model = WhisperForConditionalGeneration.from_pretrained(args.base_path,
                                                        local_files_only=args.local_files_only)


# 获取测试数据
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"测试数据：{len(test_dataset)}")

# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

whisper_model.eval()
whisper_model.to(device)
predictions = []
tp = 1e-5
special_tokens = processor.tokenizer.additional_special_tokens 
special_tokens.extend(processor.tokenizer.unique_no_split_tokens)
with open(args.save_path, 'w', encoding='utf-8') as fout:
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                _, lm_logits, _, _= whisper_model(input_features=batch['input_features'].to(device), labels=batch['labels'].to(device), return_dict=False)
                labels = batch["labels"].cpu().numpy()
                text_list = batch['text_list']
                srl_labels = batch['srl_labels']
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                sample_lm_logits = torch.nn.functional.gumbel_softmax(lm_logits, tau=tp, hard=True)
                decoded_labels = processor.tokenizer.batch_decode(torch.argmax(sample_lm_logits, dim=-1), skip_special_tokens=True)
                
                for text, decoded_label, srl_label in zip(text_list, decoded_labels, srl_labels):
                    if len(decoded_label) == 0:
                        decoded_label = ' '
                        continue
                    bert_encode = tokenizer(list(decoded_label), max_length=512, truncation=True, add_special_tokens=False)['input_ids']
                    filter_notChinese = ''
                    for i, token_id in enumerate(bert_encode):
                        if len(token_id) == 1:

                            filter_notChinese += decoded_label[i]
                            
                    if len(filter_notChinese) == 0:
                        continue

                    map_srl = process_label_with_aligner(filter_notChinese, text, srl_label, aligner)['srl']
                    fout.write(json.dumps({'text': filter_notChinese, 'srl': map_srl, 'gold_text': text}, ensure_ascii=False)+'\n')
