RUN_DIR=ckpt_dir/flip_book/20260808_120000_tactile

python3 inference.py \
--ckpt_path "$RUN_DIR/policy_best.ckpt" \
--stats_path "$RUN_DIR/normalize.pkl" \
--use_tactile
