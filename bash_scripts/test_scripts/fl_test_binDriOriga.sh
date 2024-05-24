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