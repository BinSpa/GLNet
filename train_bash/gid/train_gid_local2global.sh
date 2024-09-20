python train_deep_globe.py \
--n_class 6 \
--dataset "gid" \
--data_path "/data1/gyl/RS_DATASET/FBP" \
--model_path "../../saved_models/gid" \
--log_path "../../runs/gid" \
--task_name "fpn_gid_local2global" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 32 \
--size_g 508 \
--size_p 508 \
--path_g "gidfpn_gid_global120.pth" \
--path_g2l "gidfpn_gid_global2local50.pth" \