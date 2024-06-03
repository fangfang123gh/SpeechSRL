import argparse
import functools
import gc
import os
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor,BertTokenizerFast

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments
from model_with_suda import E2EModel
from peft import PeftModel
from get_metric import get_metric
from fast_align_zms.build_new.force_align import Aligner
from utils.utils import prepare, build_transform
from ASR_Evaluation_CN.algorithm.evaluations import calculate_WER

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("is_use_loramodel",  type=bool, default=True,     help="是否加载lora模块")
add_arg("is_use_goldtext",  type=bool, default=False,     help="是否使用金标文本")
add_arg("save_path",   type=str, default="",            help="保存预测结果的路径")
add_arg("test_data",   type=str, default="",            help="测试集的路径")
add_arg("train_data",   type=str, default="",            help="训练集的路径")
add_arg("asr_model_path",  type=str, default="", help="asr模型的路径")
add_arg("srl_model_path",  type=str, default="", help="srl任务的模型的路径")
add_arg("bert_path",  type=str, default="/data/hfmodel/bert_base_chinese", help="bert预训练模型的路径")
add_arg("base_path",  type=str, default="/data/lxx/Whisper-Finetune/models/cpb_real_decoderonly/whisper-finetune", help="asr base的模型路径")
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
processor = WhisperProcessor.from_pretrained(args.base_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids()
whisper_model = WhisperForConditionalGeneration.from_pretrained(args.base_path,
                                                        local_files_only=args.local_files_only)
if args.is_use_loramodel:
    whisper_model = PeftModel.from_pretrained(whisper_model, args.asr_model_path)


tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
transform = build_transform(tokenizer, args.train_data)
class SRLConfig:
    def __init__(self, srl_encoder_path, srl_num_labels, schema, maxlen):
        self.srl_encoder_path = srl_encoder_path
        self.srl_num_labels = srl_num_labels
        self.schema = schema
        self.maxlen = maxlen

srl_config = SRLConfig(srl_encoder_path=args.bert_path, srl_num_labels=len(transform.PHEAD.vocab), maxlen=512)
model = E2EModel(whisper_model, srl_config, processor, tokenizer, transform, None).to(device)
srl_checkpoint = torch.load(args.srl_model_path)
model.srl_model.load_state_dict(srl_checkpoint)
model.eval()

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

predictions = []
with open(args.save_path, 'w', encoding='utf-8') as fout:

    # 开始评估
    reference = []
    response = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                input_features = batch["input_features"]
                input_features = input_features.to(device)
                decoder_input_ids=batch["labels"][:, :4]

                decoder_input_ids = decoder_input_ids.to(device)
                gold_texts = batch['text_list']
                if args.is_use_goldtext:
                    conll_list, trans = model.predict(input_features, decoder_input_ids, forced_decoder_ids, gold_text=gold_texts)
                else:
                    conll_list, trans = model.predict(input_features, decoder_input_ids, forced_decoder_ids)
                
                pred = prepare(conll_list)

                gold = batch['srl_labels']

                reference.extend(gold_texts)
                response.extend(trans)
               
                for tran, pred_srl, gold_srl, gold_text in zip(trans, pred, gold, gold_texts):
                    fout.write(json.dumps({'text': tran, 'pred_srl': pred_srl, 'gold_srl': gold_srl, 'gold_text':gold_text}, ensure_ascii=False)+'\n')
       
get_metric(aligner, args.save_path, args.test_path, True)
calculate_WER(reference, response)