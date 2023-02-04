python -m torch.distributed.launch --nproc_per_node=1 --use_env sample_seq2seq.py \
--model_path $model_path$\
--step 2000 \
--batch_size 100 \
--seed2 123 \
--split test \
--out_dir generation_outputs \
--top_p -1