export CUDA_VISIBLE_DEVICES=0
python /data1/gyl/RS_Code/GLNet/train_deep_globe.py \
--n_class 7 \
--data_path "/ssd1/chenwy/deep_globe/data/" \
--model_path "/home/chenwy/deep_globe/saved_models/" \
--log_path "/home/chenwy/deep_globe/runs/" \
--task_name "fpn_deepglobe_global" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \