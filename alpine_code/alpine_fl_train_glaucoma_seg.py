import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
import evaluate
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from custom_datasets import ImageSegmentationDataset
from torchvision.transforms import functional as F
from utils import color_palette
import copy
import time
import transformers
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--train_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all train datasets")
    parser.add_argument("--val_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all validation datasets")
    parser.add_argument("--csv_img_path_col", type=str, default='image_path', help="Column name in the csv for the path to the image")
    parser.add_argument("--csv_label_path_col", type=str, default='label_path', help="Column name in the csv for the path to the segmentation label")
    parser.add_argument("--output_directory", type=str, default='/home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs', help="Desired path for output files (model, val inferences, etc)")
    parser.add_argument('--dataset_mean', nargs='+', type=float, default=[0.768, 0.476, 0.289], help='Array of float values for mean')
    parser.add_argument('--dataset_std', nargs='+', type=float, default=[0.221, 0.198, 0.165], help='Array of float values for std')
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and testing")
    parser.add_argument('--jitters', nargs='+', type=float, default=[0.2, 0.2, 0.05, 0.05, 0.75], help='Array of float jitter values: brightness, contrast, saturation, hue, probability')
    parser.add_argument("--num_epochs", type=int, default=100, help="Max number of epochs to train")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping")
    parser.add_argument("--num_val_outputs_to_save", type=int, default=5, help="Number of examples from val to save.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    parser.add_argument("--fl_finetuned", type=str, default="", help="Path to the .pth file of the federated learning fine-tuned model")
    return parser.parse_args()

# Function to load a dataset from a CSV file and make it a ImageSegmentationDataset Obj(len==dataset_size)
def load_dataset(csv_file, transform, csv_img_path_col, csv_label_path_col):
    data_df = pd.read_csv(csv_file)
    image_paths = data_df[csv_img_path_col].tolist()
    label_paths = data_df[csv_label_path_col].tolist()
    return ImageSegmentationDataset(image_paths, label_paths, transform=transform)

# Federated averaging of weights of multiple models(for all keys in the model_weights eg. fc1.weight, fc1.bias etc.)
def FedAvg(models_weights):
    num_models = len(models_weights)
    averaged_weights = {}
    # Iterating through each key in the state dictionary of the first model
    for key in models_weights[0].keys():
        sum_weights = models_weights[0][key].clone()
        for i in range(1, num_models):
            sum_weights += models_weights[i][key]
        # Avg the weights and store them in dict
        averaged_weights[key] = sum_weights / num_models
    return averaged_weights

# Function to train one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
        
        # Ensure class_labels are being extracted correctly
        if "class_labels" in batch:
            class_labels = [label.long().to(device) for label in batch["class_labels"]]
        else:
            raise ValueError("class_labels not found in batch")
        print("Training ongoing")
        optimizer.zero_grad()
        try:
            outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
        except RuntimeError as e:
            print(f"RuntimeError during forward pass: {e}")
            continue  

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_model(model, val_loader, preprocessor, criterion, device, metric, palette, inference_directory, epoch, num_val_outputs_to_save, id2label):
    model.eval()
    total_val_loss = 0.0
    total_samples = 0
    mean_ious = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            print("Val going")
            pixel_values = batch["pixel_values"].to(device)
            # Forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]])
            # Loss calculation
            loss = outputs.loss
            total_val_loss += loss.item()
            # Post-processing for metric calculation
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            # Metric update
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]
            metric.add_batch(predictions=predicted_segmentation_maps, references=ground_truth_segmentation_maps)
            # Visualization and saving
            if idx < num_val_outputs_to_save:
                for i, (original_img, predicted_map, ground_truth_map) in enumerate(zip(original_images, predicted_segmentation_maps, batch["original_segmentation_maps"])):
                    # For predicted segmentation map
                    pred_color_seg_map = np.zeros_like(original_img)
                    for label_id, color in enumerate(palette):
                        pred_color_seg_map[predicted_map.cpu() == label_id, :] = color
                    blended_pred_image = (original_img * 0.7 + pred_color_seg_map * 0.3).astype(np.uint8)

                    # For ground truth segmentation map
                    gt_color_seg_map = np.zeros_like(original_img)
                    for label_id, color in enumerate(palette):
                        gt_color_seg_map[ground_truth_map == label_id, :] = color
                    blended_gt_image = (original_img * 0.7 + gt_color_seg_map * 0.3).astype(np.uint8)

                    # Save both images
                    plt.figure(figsize=(10, 10))
                    plt.imshow(blended_pred_image)
                    plt.axis('off')
                    plt.savefig(f"{inference_directory}/epoch_{epoch}_batch_{idx}_image_{i}_val_segmentation.png", bbox_inches='tight')
                    plt.close()

                    plt.figure(figsize=(10, 10))
                    plt.imshow(blended_gt_image)
                    plt.axis('off')
                    plt.savefig(f"{inference_directory}/epoch_{epoch}_batch_{idx}_image_{i}_gt_segmentation.png", bbox_inches='tight')
                    plt.close()
            total_samples += len(pixel_values)

        mean_iou = metric.compute(num_labels=len(id2label), ignore_index=0)["mean_iou"]
        mean_ious.append(mean_iou)
        print("Mean IoU:", mean_iou)

    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0
    avg_mean_iou = sum(mean_ious) / len(mean_ious) if mean_ious else 0
    return avg_val_loss, avg_mean_iou

def find_latest_checkpoint(model_directory):
    """
    Find the latest checkpoint in the model directory based on the epoch number.

    Args:
    model_directory (str): The directory where model checkpoints are stored.

    Returns:
    str: The path to the latest checkpoint file.
    """
    checkpoint_pattern = re.compile(r'best_global_model_epoch(\d+)\.pth')
    latest_epoch = -1
    latest_checkpoint = None
    for filename in os.listdir(model_directory):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > latest_epoch:
                latest_epoch = epoch_number
                latest_checkpoint = os.path.join(model_directory, filename)
    return latest_checkpoint

def main():
    start_time = time.time()
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    csv_img_path_col = args.csv_img_path_col
    csv_label_path_col = args.csv_label_path_col
    output_directory = args.output_directory
    dataset_mean = args.dataset_mean
    dataset_std = args.dataset_std
    lr = args.lr
    batch_size = args.batch_size
    jitters = args.jitters
    num_epochs = args.num_epochs
    patience = args.patience
    num_val_outputs_to_save = args.num_val_outputs_to_save
    num_workers = args.num_workers
    cuda_num = args.cuda_num
    
    assert len(jitters) == 5, 'jitters must have 5 values'
    assert (jitters[0] < 1 and jitters[0] > 0 
            and jitters[1] <= 1 and jitters[1] >= 0 
            and jitters[2] <= 1 and jitters[2] >= 0 
            and jitters[3] <= 1 and jitters[3] >= 0
            and jitters[4] <= 1 and jitters[4] >= 0), 'jitters must be [0,1]'
    assert len(dataset_mean) == 3, 'dataset mean must have 3 float values'
    assert len(dataset_std) == 3, 'dataset std must have 3 float values'
    
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)
    
    # Find and load the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(model_directory)
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        # Modify according to your model loading logic
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        # If you have a single model architecture setup
        model.load_state_dict(checkpoint)
        # For multiple models stored in a list named 'models'
        # for model in models:
        #     model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, starting training from scratch.")
            
    # Initialize output folders
    os.makedirs(args.output_directory, exist_ok=True)
    new_base_directory = "/scratch/alpine/uthakuria@xsede.org/fl_glaucoma_checkpoints"
    project_name = os.path.basename(output_directory)
    model_directory = os.path.join(new_base_directory, project_name, "models")
    os.makedirs(model_directory, exist_ok=True)
    inference_directory = os.path.join(new_base_directory, project_name, 'inference')
    os.makedirs(inference_directory, exist_ok=True)
    epoch_info_path = os.path.join(args.output_directory, 'epoch_info.txt')
    
    dataset_names = [os.path.basename(csv_file).replace('_train.csv', '') for csv_file in args.train_csv]
    epoch_log_files = {name: open(os.path.join(args.output_directory, f'epoch_details_{name}.txt'), 'w') for name in dataset_names}
    
    # Save args as JSON
    args_dict = vars(args)  # Convert args namespace to dict
    args_dict['train_py_file'] = os.path.basename(__file__) 
    with open(os.path.join(args.output_directory, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)
    
    # for classes
    id2label = {
        0: "unlabeled",
        1: "bg",
        2: "disc",
        3: "cup"
    }
    
    # for vis
    palette = color_palette()

    # transforms
    ADE_MEAN = np.array(dataset_mean)
    ADE_STD = np.array(dataset_std)

    # Transforms
    train_transform = A.Compose([
        A.ColorJitter(brightness=args.jitters[0], contrast=args.jitters[1], saturation=args.jitters[2], hue=args.jitters[3], p=args.jitters[4]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=args.dataset_mean, std=args.dataset_std),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])
    
    # Load custom datasets for each CSV file, list of ImageSegmentationDataset Objs()
    train_datasets = [load_dataset(csv_file, train_transform, args.csv_img_path_col, args.csv_label_path_col) for csv_file in args.train_csv]
    val_datasets = [load_dataset(csv_file, val_transform, args.csv_img_path_col, args.csv_label_path_col) for csv_file in args.val_csv]
    
    # Create an empty preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    def collate_fn(batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]

        # Apply the preprocessor
        processed_batch = preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )

        # Check if processed_batch is a BatchFeature object and convert to dict if necessary
        if isinstance(processed_batch, transformers.image_processing_utils.BatchFeature):
            processed_batch = dict(processed_batch)

        # Add original images and segmentation maps to the batch
        processed_batch["original_images"] = inputs[2]
        processed_batch["original_segmentation_maps"] = inputs[3]
        
        return processed_batch
    

    # Data loaders for each dataset
    train_dataloaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn) for ds in train_datasets]
    val_dataloaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn) for ds in val_datasets]

    # Model, optimizer, scheduler and criterion initialization for each dataset
    if args.fl_finetuned:
        checkpoint = torch.load(args.fl_finetuned, map_location=device)
        models = [Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                        id2label=id2label, 
                                                                        ignore_mismatched_sizes=True).to(device) for _ in args.train_csv]
        for model in models:
            model.load_state_dict(checkpoint)
        print(f"\nLoaded the FL-finetuned weights from {args.fl_finetuned}\n")
    else:
        models = [Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                        id2label=id2label, 
                                                                        ignore_mismatched_sizes=True).to(device) for _ in args.train_csv]
        print("\nLoading the pre-trained mask2former-swin-large-cityscapes-semantic weights\n")
        
    optimizers = [optim.AdamW(model.parameters(), lr=lr) for model in models]
    schedulers = [lr_scheduler.ExponentialLR(optimizer, gamma=0.9) for optimizer in optimizers]
    criterion = torch.nn.CrossEntropyLoss()
    
    # Experiment ID for metrics
    experiment_id = output_directory.split('/')[-1] + str(time.time())
    metric = evaluate.load("mean_iou", experiment_id=experiment_id)

    # Initialize lists for tracking losses and accuracies
    train_losses, val_losses = [], []
    training_times, inference_times = [], []
    
    # Initialize tracking for individual datasets
    individual_train_losses = {name: [] for name in dataset_names}
    individual_val_losses = {name: [] for name in dataset_names}
    
    print(f'begin train, len train: {len(train_dataloaders[0])}, len val: {len(val_dataloaders[0])}')

    # Main training loop
    best_val_loss = float('inf')  # Initialize best validation loss
    best_mean_iou = 0.0  # Initialize best mean IoU

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Initialize variables to track epoch training details
        epoch_train_losses = []
        epoch_train_start_time = time.time()  # Start time for training

        # Train each model separately and collect their state_dicts
        weights = []
        for i, (model, train_loader, optimizer) in enumerate(zip(models, train_dataloaders, optimizers)):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Dataset {i+1} - Training Loss: {train_loss}")
            weights.append(model.state_dict())
            epoch_train_losses.append(train_loss)
            dataset_name = dataset_names[i]
            individual_train_losses[dataset_name].append(train_loss)

            # Step the scheduler
            schedulers[i].step()
            
        # average and append training loss for this epoch
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        training_time = time.time() - epoch_train_start_time
        training_times.append(training_time)  # Track training time

        # Federated averaging
        global_weights = FedAvg(weights)
        for model in models:
            model.load_state_dict(global_weights)
            
        # Initialize variables to track validation details
        epoch_val_start_time = time.time()

        # Validate the global model on each validation dataset and compute metrics
        total_val_loss = 0
        total_mean_iou = 0
        for i, val_loader in enumerate(val_dataloaders):
            val_loss, mean_iou = validate_model(model, val_loader, preprocessor, criterion, device, metric, palette, inference_directory, epoch, args.num_val_outputs_to_save, id2label)
            print(f"Validation Loss: {val_loss}, Mean IoU: {mean_iou}")
            total_val_loss += val_loss
            total_mean_iou += mean_iou
            dataset_name = dataset_names[i]
            individual_val_losses[dataset_name].append(val_loss)
            epoch_log_files[dataset_name].write(f"Epoch {epoch+1}: Training Loss: {individual_train_losses[dataset_name][-1]}, Validation Loss: {val_loss}, Mean IoU: {mean_iou}\n")
            epoch_log_files[dataset_name].flush()
            
        avg_val_loss = total_val_loss / len(val_dataloaders)
        avg_mean_iou = total_mean_iou / len(val_dataloaders)
        print(f"Average Validation Loss: {avg_val_loss}, Average Mean IoU: {avg_mean_iou}")
        val_losses.append(avg_val_loss)
        
        # Record validation (inference) time for this epoch
        inference_time = time.time() - epoch_val_start_time
        
        # Record the epoch metrics in epoch_info.txt
        with open(epoch_info_path, 'a') as epoch_info_file:
            epoch_info_file.write(
                f"Epoch {epoch+1}: "
                f"Training Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"Mean IoU: {avg_mean_iou:.4f}, "
                f"Training Time: {training_time:.2f}s, "
                f"Inference Time: {inference_time:.2f}s\n"
            )
        # Plot individual loss curves
        for dataset_name in dataset_names:
            plt.figure(figsize=(10, 7))
            plt.plot(individual_train_losses[dataset_name], label=f'Train Loss - {dataset_name}')
            plt.plot(individual_val_losses[dataset_name], label=f'Validation Loss - {dataset_name}')
            plt.title(f'Training and Validation Loss Over Epochs - {dataset_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            # Save the plot with a static filename for each dataset
            plt.savefig(os.path.join(args.output_directory, f'dynamic_loss_curves_{dataset_name}.png'))
            plt.close()

        # Update the training-validation loss plot after each epoch
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_directory, 'dynamic_loss_curves.png'))
        plt.close()

        # Log the training and inference times
        with open(os.path.join(args.output_directory, 'epoch_times.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: Training Time: {training_time:.2f}s, Inference Time: {inference_time:.2f}s\n")

        # Early stopping check based on validation loss, and tracking best mean IoU
        if avg_val_loss < best_val_loss or avg_mean_iou > best_mean_iou:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            if avg_mean_iou > best_mean_iou:
                best_mean_iou = avg_mean_iou
            counter = 0
            # Save the best model
            torch.save(global_weights, os.path.join(model_directory, f'best_global_model_epoch{epoch}.pth'))
            print(f"Best model (Loss/IoU) saved at epoch {epoch+1}")
        else:
            counter += 1
            if counter >= args.patience:
                print("Early stopping triggered")
                break
            
        # Total training and inference time
        total_time = time.time() - epoch_start_time
        print(f"Total Time for Epoch {epoch+1}: {total_time:.2f}s")

    # Plot training and validation losses
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'loss_curves.png'))
    plt.close()
    
    # Save training and inference times to a text file
    with open(os.path.join(args.output_directory, 'time_log.txt'), 'w') as f:
        f.write("Training Times (seconds): {}\n".format(training_times))
        f.write("Inference Times (seconds): {}\n".format(inference_times))
        total_time = time.time() - start_time
        f.write("Total Time Taken (seconds): {:.2f}\n".format(total_time))
    
    for log_file in epoch_log_files.values():
        log_file.close()

if __name__ == "__main__":
    main()