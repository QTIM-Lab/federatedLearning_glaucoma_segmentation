from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, images, masks, transform):
        """
        Args:
            dataset
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path_to_img = self.images[idx]
        path_to_mask = self.masks[idx]

        original_image = np.array(Image.open(path_to_img))
        original_segmentation_map = np.array(Image.open(path_to_mask).convert('L'))
        original_segmentation_map[original_segmentation_map == 0] = 1
        original_segmentation_map[original_segmentation_map == 127] = 2
        original_segmentation_map[original_segmentation_map == 255] = 3
        
        
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2,0,1)

        return image, segmentation_map, original_image, original_segmentation_map
