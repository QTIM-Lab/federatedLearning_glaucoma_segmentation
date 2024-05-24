import argparse
import os
import time
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from transformers import MaskFormerImageProcessor
# from transformers import MaskFormerForInstanceSegmentation
from transformers import Mask2FormerForUniversalSegmentation
import evaluate
import torch
# from tqdm.auto import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from custom_datasets import ImageSegmentationDataset
from utils import color_palette


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--train_csv", type=str, default='./outputs', help="Path to .csv file with rows for all train")
    parser.add_argument("--val_csv", type=str, default='./outputs', help="Path to .csv file with rows for all val")
    parser.add_argument("--csv_img_path_col", type=str, default='image', help="Column name in the csv for the path to the image")
    parser.add_argument("--csv_label_path_col", type=str, default='label', help="Column name in the csv for the path to the segmentation label")
    parser.add_argument("--output_directory", type=str, default='./outputs', help="Desired path for output files (model, val inferences, etc)")
    parser.add_argument('--dataset_mean', nargs='+', type=float, help='Array of float values for mean i.e. 0.709 0.439 0.287')
    parser.add_argument('--dataset_std', nargs='+', type=float, help='Array of float values for std i.e. 0.210 0.220 0.199')
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and testing")
    parser.add_argument('--jitters', nargs='+', type=float, help='Array of float jitter values: brightness, contrast, saturation, hue, probability')
    parser.add_argument("--num_epochs", type=int, default=50, help="Max number of epochs to train")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping")
    parser.add_argument("--num_val_outputs_to_save", type=int, default=3, help="Number of examples from val to save, so you can see your model improve on it during training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders")
    return parser.parse_args()

def main():
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

    assert len(jitters) == 5, 'jitters must have 5 values'
    assert (jitters[0] < 1 and jitters[0] > 0 
            and jitters[1] <= 1 and jitters[1] >= 0 
            and jitters[2] <= 1 and jitters[2] >= 0 
            and jitters[3] <= 1 and jitters[3] >= 0
            and jitters[4] <= 1 and jitters[4] >= 0), 'jitters must be [0,1]'
    assert len(dataset_mean) == 3, 'dataset mean must have 3 float values'
    assert len(dataset_std) == 3, 'dataset std must have 3 float values'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)

    ### Initialize output folders for the model, val inferences
    os.makedirs(output_directory, exist_ok=True)
    new_base_directory = "/scratch/alpine/uthakuria@xsede.org/fl_glaucoma_checkpoints"
    project_name = os.path.basename(output_directory)
    model_directory = os.path.join(new_base_directory, project_name, "models")
    os.makedirs(model_directory, exist_ok=True)
    inference_directory = os.path.join(new_base_directory, project_name, 'inference')
    os.makedirs(inference_directory, exist_ok=True)

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

    train_transform = A.Compose([
        A.ColorJitter(brightness=jitters[0], contrast=jitters[1], saturation=jitters[2], hue=jitters[3], p=jitters[4]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])

    ### Get dataset
    train_data_df = pd.read_csv(train_csv)
    val_data_df = pd.read_csv(val_csv)
    train_image_paths = train_data_df[csv_img_path_col].tolist()
    train_mask_paths = train_data_df[csv_label_path_col].tolist()
    val_image_paths = val_data_df[csv_img_path_col].tolist()
    val_mask_paths = val_data_df[csv_label_path_col].tolist()

    # Use custom dataset
    train_dataset = ImageSegmentationDataset(train_image_paths, train_mask_paths, transform=train_transform)
    val_dataset = ImageSegmentationDataset(val_image_paths, val_mask_paths, transform=val_transform)

    # Create an empty preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    def collate_fn(batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]
        # this function pads the inputs to the same size,
        # and creates a pixel mask
        # actually padding isn't required here since we are cropping
        batch = preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )

        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]
        
        return batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Replaces the head of the pre-trained model
    # model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
    #                                                         id2label=id2label,
    #                                                         ignore_mismatched_sizes=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic",
                                                                id2label=id2label,
                                                                ignore_mismatched_sizes=True)
    model.to(device)

    # experiment id should be a unique string value for the experiment. can use the output folders name
    metric = evaluate.load("mean_iou", experiment_id=output_directory.split('/')[-1] + str(time.time()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_losses = []
    val_losses = []

    print(f'begin train, len train: {len(train_dataloader)}, len val: {len(val_dataloader)}')

    best_mean_iou = 0.0  # Track the best mean IoU
    counter = 0  # Counter to monitor epochs since improvement
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_samples = 0

        model.train()
        for idx, batch in enumerate(train_dataloader):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 1 == 0:
                print("Loss: ", loss)

            # Optimization
            optimizer.step()

        train_loss = running_loss/num_samples
        train_losses.append(train_loss)
        
        model.eval()
        for idx, batch in enumerate(val_dataloader):
            pixel_values = batch["pixel_values"]

            # Forward pass
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values.to(device))

            # get original images
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            # predict segmentation maps
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)
            
            if idx < num_val_outputs_to_save:
                # Val
                segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

                color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
                for label, color in enumerate(palette):
                    color_segmentation_map[segmentation_map - 1 == label, :] = color
                # Convert to BGR
                pred_color_seg = color_segmentation_map[..., ::1]

                img = original_images[0] * 0.7 + pred_color_seg * 0.3
                img = img.astype(np.uint8)

                epoch_directory = os.path.join(inference_directory, f'{epoch}')
                os.makedirs(epoch_directory, exist_ok=True)
                channel_directory = os.path.join(epoch_directory, f'{idx}')
                os.makedirs(channel_directory, exist_ok=True)
                plt.figure(figsize=(15, 10))
                plt.imshow(img)
                plt.show()
                plt.axis('off')
                loss_str = str(outputs.loss)
                loss_str = loss_str.replace('.', '_')
                channel_filename = os.path.join(channel_directory, f'val_segmentation_loss{loss_str}.png')
                plt.savefig(channel_filename, bbox_inches='tight')
                plt.close()

                # Ground truth
                segmentation_map = batch["original_segmentation_maps"][0]

                color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
                for label, color in enumerate(palette):
                    color_segmentation_map[segmentation_map - 1 == label, :] = color
                # Convert to BGR
                ground_truth_color_seg = color_segmentation_map[..., ::1]

                img = original_images[0] * 0.7 + ground_truth_color_seg * 0.3
                img = img.astype(np.uint8)

                plt.figure(figsize=(15, 10))
                plt.imshow(img)
                plt.show()
                plt.axis('off')
                channel_filename = os.path.join(channel_directory, f'gt_segmentation.png')
                plt.savefig(channel_filename, bbox_inches='tight')
                plt.close()

            # get ground truth segmentation maps
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)

        # Step the scheduler to update the learning rate
        scheduler.step()
        
        # NOTE this metric outputs a dict that also includes the mIoU per category as keys
        # so if you're interested, feel free to print them as well
        mean_iou = metric.compute(num_labels=len(id2label), ignore_index=0)['mean_iou']
        val_losses.append(mean_iou)
        print("Mean IoU:", mean_iou)
        
        # Save the model if mean IoU improves
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            counter = 0  # Reset the counter
            model_filename = os.path.join(model_directory, f'model_epoch{epoch}_ckpt.pt')
            torch.save(model.state_dict(), model_filename)  # Save the model
        else:
            counter += 1

        # Check for early stopping based on patience
        if counter >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    plt.figure(figsize=(15, 10))
    plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Mean IOU Curves')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'loss_curves.png'))
    plt.close()


if __name__ == "__main__":
    main()
