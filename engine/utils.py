from PIL import Image
import numpy as np

# The reason we convert to a single channel is because MaskFormer wants the labels to be shaped (h,w,1)
# originally the images had 3 channels for cup, disc, background so was (h,w,3), but we just create color_map to fix
def convert_to_single_channel(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert to RGB mode if not already in that mode
    image = image.convert('RGB')

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Compute the max value along the channel axis
    max_channel = np.argmax(img_array, axis=2) + 1  # Adding 1 to make red 1, green 2, blue 3

    # Define the color mappings after max channel computation
    color_map = {
        1: 1,  # Red
        2: 2,  # Green
        3: 3,  # Blue
    }

    # Create masks based on the max channel values
    masks = np.zeros((img_array.shape[0], img_array.shape[1], len(color_map)), dtype=np.uint8)
    for index, label in color_map.items():
        masks[:, :, index - 1] = max_channel == label

    # Create the single-channel image using the masks
    single_channel_image = np.argmax(masks, axis=2) + 1  # Adding 1 to adjust labels

    return single_channel_image

def convert_grayscale_disc_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert to grayscale mode if not already in that mode
    image = image.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(image, dtype=np.uint8)

    # Replace black pixels with 1 (background) and white pixels with 2 (disc)
    img_array[img_array >= 1] = 2  # Assuming white pixels have intensity 255
    img_array[img_array == 0] = 1  # Assuming black pixels have intensity 0

    return img_array


def color_palette():
    """Color palette that maps each class to RGB values.
    
    This one is actually taken from ADE20k.
    """
    return [[255,0,0], [0,255,0], [0,0,255]]
