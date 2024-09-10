export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--n_class 25 \
--data_path "/data1/gyl/RS_DATASET/FBP" \
--model_path "../../saved_models/fbp" \
--log_path "../../runs/fbp" \
--task_name "fpn_fbp_global2local" \
--mode 2 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_fbp_global120.pth" \