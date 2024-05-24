# fuser -v /dev/nvidia*
# pids=$(fuser -v /dev/nvidia* | sed 's/^ *[0-9]* //')
# for pid in $pids; do kill -9 $pid; done

# variables
binDriOriga_mean="0.88702821 0.5666646 0.29283205"
binDriOriga_std="0.15141807 0.17632471 0.1514714"
riga_mean="0.83512329 0.49202078 0.22558286"
riga_std="0.14766443 0.16120118 0.12717136"
combined_mean="0.768 0.476 0.289"
combined_std="0.221 0.198 0.165"

# # train
# python3 fl_glaucoma_seg.py \
#     --train_csv /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_train.csv \
#     --val_csv /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_val.csv \
#     --csv_img_path_col image_path \
#     --csv_label_path_col label_path \
#     --output_directory /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs_messidor_alone \
#     --dataset_mean $combined_mean \
#     --dataset_std $combined_mean \
#     --lr 0.00003 \
#     --batch_size 8 \
#     --jitters 0.5 0.5 0.25 0.1 0.75 \
#     --num_epochs 100 \
#     --patience 2 \
#     --num_val_outputs_to_save 5 \
#     --num_workers 16 \
#     --cuda_num 1 \
#     --fl_finetuned /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train__outputs_binDriOriga/model/best_global_model_epoch31.pth
    # --fl_finetuned /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_RIGA/model/best_global_model_epoch62.pth

# --train_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_train.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_train.csv
# --val_csv /home/thakuriu/fl_glaucoma_seg/csvs/binrushed_val.csv /home/thakuriu/fl_glaucoma_seg/csvs/drishti_val.csv   

# # inference
python3 run_inference_multiprocess.py \
    --model_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs_messidor_alone/model/best_global_model_epoch11.pth \
    --input_csv /home/thakuriu/fl_glaucoma_seg/csvs/combined_test_riga.csv \
    --csv_path_col_name image_path \
    --output_root_dir /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs_messidor_alone \
    --num_processes 1 \
    --cuda_num 1

# To eval disc:
# python3 evaluate_jaccard_dice_institution.py \
#     --prediction_folder /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs_messidor_fl_finetuned_riga/outputs \
#     --label_folder /sddata/data/retina_datasets_preprocessed/federated_learning_public/ \
#     --csv_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs_messidor_fl_finetuned_riga/results.csv \
#     --eval_disc \
#     --cuda_num 1

# # To eval cup:
# python3 evaluate_jaccard_dice_institution.py \
#     --prediction_folder /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs_messidor_fl_finetuned_riga/outputs \
#     --label_folder /sddata/data/retina_datasets_preprocessed/federated_learning_public/ \
#     --csv_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs_messidor_fl_finetuned_riga/results.csv \
#     --cuda_num 0

# train
# python3 fl_glaucoma_seg.py \
#     --train_csv /sddata/data/retina_datasets_preprocessed/output_csvs/binrushed_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/chaksu_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/drishti_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/magrabi_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/origa_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/refuge_train.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/rimone_train.csv \
#     --val_csv /sddata/data/retina_datasets_preprocessed/output_csvs/binrushed_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/chaksu_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/drishti_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/g1020_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/magrabi_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/messidor_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/origa_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/refuge_val.csv \
#                 /sddata/data/retina_datasets_preprocessed/output_csvs/rimone_val.csv \
#     --csv_img_path_col image_path \
#     --csv_label_path_col label_path \
#     --output_directory /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_RIGA \
#     --dataset_mean 0.768 0.476 0.289 \
#     --dataset_std 0.221 0.198 0.165 \
#     --lr 0.00003 \
#     --batch_size 8 \
#     --jitters 0.5 0.5 0.25 0.1 0.75 \
#     --num_epochs 100 \
#     --patience 8 \
#     --num_val_outputs_to_save 5 \
#     --num_workers 16