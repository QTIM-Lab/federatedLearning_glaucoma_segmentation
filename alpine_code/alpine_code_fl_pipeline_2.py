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

def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--train_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all train datasets")
    parser.add_argument("--val_csv", nargs='+', type=str, required=True, help="Paths to .csv files with rows for all validation datasets")
    parser.add_argument("--local_sites_training_epochs", type=int, default=1, help="Number of epochs to train on each site before federated averaging")
    parser.add_argument("--fl_rounds", type=int, default=100, help="Number of federated learning rounds")
    parser.add_argument("--csv_img_path_col", type=str, default='image_path', help="Column name in the csv for the path to the image")
    parser.add_argument("--csv_label_path_col", type=str, default='label_path', help="Column name in the csv for the path to the segmentation label")
    parser.add_argument("--output_directory", type=str, default='/home/thakuriu/fl_glaucoma_seg/detection_segmentation_v2/segmentation_train_and_inference/train_outputs', help="Desired path for output files (model, val inferences, etc)")
    parser.add_argument('--dataset_mean', nargs='+', type=float, default=[0.768, 0.476, 0.289], help='Array of float values for mean')
    parser.add_argument('--dataset_std', nargs='+', type=float, default=[0.221, 0.198, 0.165], help='Array of float values for std')
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and testing")
    parser.add_argument('--jitters', nargs='+', type=float, default=[0.2, 0.2, 0.05, 0.05, 0.75], help='Array of float jitter values: brightness, contrast, saturation, hue, probability')
    parser.add_argument("--patience", type=int, default=7, help="Early stopping")
    parser.add_argument("--num_val_outputs_to_save", type=int, default=5, help="Number of examples from val to save.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    parser.add_argument("--fl_finetuned", type=str, default="", help="Path to the .pth file of the federated learning fine-tuned model")
    parser.add_argument("--fl_patience", type=int, default=5, help="Early stopping for federated learning rounds")
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
def train_one_epoch(model, dataloader, optimizer, criterion, device, batch_size):
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
    return total_loss / (len(dataloader) * batch_size)

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
    patience = args.patience
    num_val_outputs_to_save = args.num_val_outputs_to_save
    num_workers = args.num_workers
    cuda_num = args.cuda_num
    local_sites_training_epochs = args.local_sites_training_epochs
    fl_patience = args.fl_patience
    fl_rounds = args.fl_rounds
    
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
            
    # Initialize output folders
    os.makedirs(args.output_directory, exist_ok=True)
    new_base_directory = "/scratch/alpine/uthakuria@xsede.org/fl_glaucoma_checkpoints"
    project_name = os.path.basename(output_directory)
    model_directory = os.path.join(new_base_directory, project_name, "models")
    os.makedirs(model_directory, exist_ok=True)
    inference_directory = os.path.join(new_base_directory, project_name, 'inference')
    os.makedirs(inference_directory, exist_ok=True)
    dataset_names = [os.path.basename(csv_file).replace('_train.csv', '') for csv_file in args.train_csv]
    

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
    
    # Experiment ID for metrics
    experiment_id = output_directory.split('/')[-1] + str(time.time())
    metric = evaluate.load("mean_iou", experiment_id=experiment_id)
    # print(f'begin train, len train: {len(train_dataloaders[0])}, len val: {len(val_dataloaders[0])}')

    
    def log_metrics(site_name, fl_round, epoch, train_loss, val_loss, mean_iou):
        """Logs metrics to a dynamically updated text file."""
        metrics_file_path = os.path.join(args.output_directory, f'{site_name}_metrics.txt')
        with open(metrics_file_path, 'a') as f:
            f.write(f"FL Round: {fl_round+1}, Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Mean IoU: {mean_iou}\n")

    def plot_loss_curves(site_name, train_losses, val_losses):
        """Updates and saves loss curves after each epoch."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Training and Validation Loss - {site_name}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_directory, f'{site_name}_loss_curves.png'))
        plt.close()
        
    def save_model(model, model_directory, fl_round, metric):
        """Saves the global model checkpoint."""
        model_path = os.path.join(model_directory, f'best_global_model_round_{fl_round+1}_metric_{metric:.4f}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def early_stopping_check(val_loss, site_best_val_loss, site_best_weights, model, counter, patience):
        if val_loss < site_best_val_loss:
            site_best_val_loss = val_loss
            site_best_weights = copy.deepcopy(model.state_dict())
            counter = 0
            print(f"New best model for site saved with val loss: {val_loss}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered for site after {patience} rounds without improvement.")
                return True, site_best_val_loss, site_best_weights, counter
        return False, site_best_val_loss, site_best_weights, counter

    def global_early_stopping_check(current_metric, best_metric, counter, patience):
        updated = False  # Initialize the updated flag
        if current_metric < best_metric:
            best_metric = current_metric
            counter = 0
            updated = True  # Update flag to True when a new best metric is found
            print(f"New best global model saved with metric: {current_metric}")
        else:
            counter += 1
            if counter >= patience:
                print("Global early stopping triggered.")
                return True, best_metric, counter, updated  # Ensure to return 'updated' here
        return False, best_metric, counter, updated  # And also here

    global_weights = None
    global_best_val_metric = float('inf')
    early_stop_counter = 0
    # fl_round loop
    for fl_round in range(fl_rounds):
        print(f"Federated Learning Round {fl_round+1}/{fl_rounds}")
        model_states = []

        round_best_weights = {}
        round_best_val_loss = {site_name: float('inf') for site_name in dataset_names}
        # collect best validation metrics from each site
        round_val_metrics = []

        for site_index in range(len(train_dataloaders)):
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                        id2label=id2label, 
                                                                        ignore_mismatched_sizes=True).to(device)
            # If a fine-tuning checkpoint is specified, load it
            if args.fl_finetuned:
                checkpoint = torch.load(args.fl_finetuned, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"Loaded the FL-finetuned weights from {args.fl_finetuned}")
            elif global_weights:
                # Or if a global model state exists from previous rounds, load it
                model.load_state_dict(global_weights)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            criterion = torch.nn.CrossEntropyLoss()
            
            train_loader = train_dataloaders[site_index]
            val_loader = val_dataloaders[site_index]
            
            site_name = dataset_names[site_index]
            site_best_val_loss = float('inf')
            site_best_weights = None
            counter = 0 # Reset counter for early stopping
            train_losses, val_losses = [], []  # Reset losses for dynamic plotting

            for epoch in range(local_sites_training_epochs):
                print(f"Site {site_name}, Epoch {epoch+1}/{local_sites_training_epochs}")
                
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, batch_size)
                val_loss, mean_iou = validate_model(model, val_loader, preprocessor, criterion, device, metric, palette, inference_directory, fl_round * local_sites_training_epochs + epoch, num_val_outputs_to_save, id2label)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if val_loss < round_best_val_loss[site_name]:
                    round_best_val_loss[site_name] = val_loss
                    round_best_weights[site_name] = copy.deepcopy(model.state_dict())

                # Early stopping check and update best model if improved
                stop, site_best_val_loss, site_best_weights, counter = early_stopping_check(val_loss, site_best_val_loss, site_best_weights, model, counter, patience)
                if stop:
                    break
                scheduler.step(val_loss)

                # Logging and plotting
                log_metrics(site_name, fl_round, epoch, train_loss, val_loss, mean_iou)
                print(f"Before plotting for {site_name}, train_losses: {train_losses}, val_losses: {val_losses}")
                plot_loss_curves(site_name, train_losses, val_losses)
            
            model_states.append(copy.deepcopy(model.state_dict()))
            round_best_weights[site_name] = site_best_weights
            round_val_metrics.append(round_best_val_loss[site_name])
            # And before moving to the next site, cleanup to free up GPU memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            
        # Compute the mean validation metric across sites for the current round
        mean_val_metric = sum(round_val_metrics) / len(round_val_metrics)
        # Check for global early stopping
        stop, global_best_val_metric, early_stop_counter, updated = global_early_stopping_check(mean_val_metric, global_best_val_metric, early_stop_counter, args.fl_patience)
        if stop:
            print(f"Global early stopping triggered after {fl_round+1} rounds.")
            break  # Exit the FL rounds loop if global early stopping condition is met
        
        # Federated averaging with best models and update global model
        best_weights = [round_best_weights[site_name] for site_name in dataset_names]
        global_weights = FedAvg(best_weights)
        
        if updated:
        # Save the updated global model with the new global weights
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", 
                                                                        id2label=id2label, 
                                                                        ignore_mismatched_sizes=True)
            model.to(device).load_state_dict(global_weights)
            save_model(model, model_directory, fl_round, global_best_val_metric)  # Adjust save_model to handle a model instance
            del model  # Cleanup
            torch.cuda.empty_cache()

        print(f"Federated Learning Round {fl_round+1} completed.")

if __name__ == "__main__":
    main()