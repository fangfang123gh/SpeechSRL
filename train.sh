export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port 20005 finetune.py \
--base_model  \
--output_dir  \
--num_train_epochs 20 \
--train_data  \
--test_data  \
--tune_whisper decoder_only \
--eval_steps 1000 \
--save_steps 1000 \
--logging_steps 50 \
--is_e2e True \
--is_use_gumbel_softmax_loss False \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 3e-5 \
--gumbel_tp 1e-5 \
--is_remove_augment True \