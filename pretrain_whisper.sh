export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20006 finetune.py \
--base_model= \
--output_dir= \
--num_train_epochs 50 \
--train_data  \
--test_data  \
--tune_whisper decoder_only \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--eval_steps 2000 \
--save_steps 2000 \