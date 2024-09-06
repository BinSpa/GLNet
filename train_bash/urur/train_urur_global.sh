# export CUDA_VISIBLE_DEVICES=0
python /data1/gyl/RS_Code/GLNet/train_deep_globe.py \
--n_class 8 \
--dataset "urur" \
--data_path "/data1/gyl/RS_DATASET/URUR" \
--model_path "../../saved_models/urur" \
--log_path "../../runs/urur" \
--task_name "fpn_urur_global" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508