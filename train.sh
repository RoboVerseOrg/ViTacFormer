python3 imitate_episodes.py \
--task_name flip_book \
--ckpt_dir ckpt_dir/flip_book \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-4 \
--seed 0 \
--use_tactile \
--tactile_teacher_forcing_epochs 75 \
# --resume_path ckpt_dir/flip_book/<run>/policy_last.ckpt
