import argparse
import functools
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, BertTokenizerFast, WhisperTokenizer
import json
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments
from model_with_suda import E2EModel
import torch
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from fast_align_zms.build_new.force_align import Aligner
from utils.utils import build_transform

class MyTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        
        
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="",      help="Whisper的基础模型")
add_arg("bert_model",    type=str, default="",      help="Whisper的基础模型")
add_arg("output_dir",    type=str, default="",                  help="训练保存模型的路径")
add_arg("aligner_fwd_params_path",   type=str, default="",            help="aligner的fwd_param path")
add_arg("aligner_fwd_err_path",   type=str, default="",            help="aligner的fwd_err path")
add_arg("aligner_rev_params_path",   type=str, default="",            help="aligner的rev_param path")
add_arg("aligner_rev_err_path",   type=str, default="",            help="aligner的rev_err path")
add_arg("warmup_steps",  type=int, default=50,      help="训练预热步数")
add_arg("logging_steps", type=int, default=10,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=500,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=500,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=8,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=3e-5,  help="学习率大小")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒")
add_arg("use_adalora",   type=bool,  default=False,  help="是否使用AdaLora而不是Lora")
add_arg("fp16",          type=bool,  default=False,  help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,  default=False, help="是否将模型量化为8位")
add_arg("timestamps",    type=bool,  default=False, help="训练时是否使用时间戳数据")
add_arg("use_compile",   type=bool, default=False, help="是否使用Pytorch2.0的编译器")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=20,      help="训练的轮数")
add_arg("language",      type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("augment_config_path",         type=str, default=None, help="数据增强配置文件路径")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=1,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=1,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("is_e2e", type=bool, default=True,    help="梯度累积步数")
add_arg("tune_whisper", type=str, default='none',    help="梯度累积步数")
add_arg("is_use_gumbel_softmax_loss", type=bool, default=False,    help="梯度累积步数")
add_arg("is_remove_augment", type=bool, default=False,    help="梯度累积步数")
add_arg("seed", type=int, default=888,    help="梯度累积步数")
add_arg("gumbel_tp", type=float, default=1e-5,  help="学习率大小")
args = parser.parse_args()
print_arguments(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids()

tokenizer = BertTokenizerFast.from_pretrained(args.bert_model, do_lower_case=True)
aligner = Aligner(args.aligner_fwd_params_path, args.aligner_fwd_err_path, args.aligner_rev_params_path, args.aligner_rev_err_path)

# 读取数据
train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              mode='train',
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path)
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             mode='eval',
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
print(train_dataset)

# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 获取Whisper模型
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# 获取模型
whisper_model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                        load_in_8bit=args.use_8bit,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only)
whisper_model.config.forced_decoder_ids = None
whisper_model.config.suppress_tokens = []
# 量化模型
whisper_model = prepare_model_for_kbit_training(whisper_model)
# 注册forward，否则多卡训练会失败
whisper_model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


if False:
    # 恢复训练时加载Lora参数
    print("Loading adapters from checkpoint.")
    whisper_model = PeftModel.from_pretrained(whisper_model, args.resume_from_checkpoint, is_trainable=True)
else:
    if args.tune_whisper == 'none':
        for name, param in whisper_model.named_parameters():
            param.requires_grad = False
    else:
        if args.tune_whisper == 'decoder_only':
            print('加载LoRA模块...')
            target_modules = []
            for i in range(32):
                for name in ['encoder_attn', 'self_attn']:
                    target_modules.extend([f'model.decoder.layers.{i}.{name}.k_proj', f'model.decoder.layers.{i}.{name}.q_proj', f'model.decoder.layers.{i}.{name}.v_proj'])
        elif args.tune_whisper == 'all':
            target_modules = ["k_proj", "q_proj", "v_proj"]
        elif args.tune_whisper == 'encoder_only':
            target_modules = []
            for i in range(32):
                for name in ['encoder_attn', 'self_attn']:
                    target_modules.extend([f'model.encoder.layers.{i}.{name}.k_proj', f'model.encoder.layers.{i}.{name}.q_proj', f'model.encoder.layers.{i}.{name}.v_proj'])
        else:
            pass
        if args.use_adalora:
            config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules)
        else:
            config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
        whisper_model = get_peft_model(whisper_model, config)



transform = build_transform(tokenizer)

class SRLConfig:
    def __init__(self, srl_encoder_path, srl_num_labels, schema, maxlen):
        self.srl_encoder_path = srl_encoder_path
        self.srl_num_labels = srl_num_labels
        self.maxlen = maxlen
        self.gumbel_tp = args.gumbel_tp

srl_config = SRLConfig(srl_encoder_path=args.bert_model, srl_num_labels=len(transform.PHEAD.vocab), maxlen=512)
model = E2EModel(whisper_model, srl_config, processor, tokenizer, transform, aligner, args.is_e2e, args.is_use_gumbel_softmax_loss, args.is_remove_augment)

if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]
output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))
# 定义训练参数
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # 保存检查点和意志的目录
                             per_device_train_batch_size=args.per_device_train_batch_size,  # 训练batch_size大小
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # 评估batch_size大小
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
                             learning_rate=args.learning_rate,  # 学习率大小
                             warmup_steps=args.warmup_steps,  # 预热步数点
                             evaluation_strategy="epoch",  # 指定按照步数评
                             num_train_epochs=args.num_train_epochs,  # 微调训练轮数
                             weight_decay=0.01,
                             warmup_ratio=0.1,
                             save_strategy="epoch",  # 指定按照步数保存检查估模型
                             load_best_model_at_end=True,  # 指定是否在结束时加载最优模型
                             fp16=args.fp16,  # 是否使用半精度训练
                             report_to=["tensorboard"],  # 指定使用tensorboard保存log
                             save_steps=args.save_steps,  # 指定保存检查点的步数
                             eval_steps=args.eval_steps,  # 指定评估模型的步数
                             seed=args.seed,
                             data_seed = args.seed,
                             torch_compile=args.use_compile, # 使用Pytorch2.0的编译器
                             save_total_limit=1,  # 只保存最新检查点的数量
                             optim='adamw_torch',  # 指定优化方法
                             ddp_find_unused_parameters=True,  # 分布式训练设置
                             dataloader_num_workers=args.num_workers,  # 设置读取数据的线程数量
                             logging_steps=args.logging_steps,  # 指定打印log的步数
                             remove_unused_columns=False,
                             label_names=["labels"])  # 删除模型不需要的数据列
# 定义训练器
trainer = MyTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback, EarlyStoppingCallback(early_stopping_patience=3)]) 

model.whisper_model.config.use_cache = False

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最后的模型
trainer.save_model()
trainer.save_state()
torch.save(args, os.path.join(args.output_dir, 'training_config.bin'))
if training_args.local_rank == 0 or training_args.local_rank == -1:

    model.whisper_model.save_pretrained(os.path.join(output_dir, "checkpoint-final-asr"))
    torch.save(model.srl_model.state_dict(), os.path.join(output_dir, "final.pth"))
aligner.close()
