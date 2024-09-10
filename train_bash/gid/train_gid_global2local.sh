export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--n_class 6 \
--data_path "/data1/gyl/RS_DATASET/FBP" \
--model_path "../../saved_models/gid" \
--log_path "../../runs/gid" \
--task_name "fpn_gid_global2local" \
--mode 2 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_gid_global120.pth" \