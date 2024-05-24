# # inference
python3 ../../run_inference_multiprocess.py \
    --model_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs/train_outputs_magrabi_v2/model/best_global_model_epoch59.pth \
    --input_csv /home/thakuriu/fl_glaucoma_seg/csvs/combined_test_riga.csv \
    --csv_path_col_name image_path \
    --output_root_dir /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_magrabi_v2 \
    --num_processes 1 \
    --cuda_num 0

# To eval disc:
# python3 ../../evaluate_jaccard_dice_institution.py \
#     --prediction_folder /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_magrabi_v2/outputs \
#     --label_folder /sddata/data/retina_datasets_preprocessed/federated_learning_public/ \
#     --csv_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_magrabi_v2/results.csv \
#     --eval_disc \
#     --cuda_num 1

# # To eval cup:
# python3 ../../evaluate_jaccard_dice_institution.py \
#     --prediction_folder /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_magrabi_v2/outputs \
#     --label_folder /sddata/data/retina_datasets_preprocessed/federated_learning_public/ \
#     --csv_path /home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/test_outputs/test_outputs_magrabi_v2/results.csv \
#     --cuda_num 1