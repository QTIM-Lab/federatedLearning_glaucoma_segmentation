import matplotlib.pyplot as plt
# import imageio as image
from PIL import Image
import argparse

print("Hey")
parser = argparse.ArgumentParser(description="num_workers")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders")

# Path to the image file
image_path = "/scratch/alpine/uthakuria@xsede.org/glaucoma_seg_dataset/binrushed/images/cropped_br1_image10prime.png"

# Specify the path where you want to save the image
save_path = "/scratch/alpine/uthakuria@xsede.org/results/image.png"

# Read the image
# try:
img = Image.open(image_path)

# Display the image
# img.show()

# Save the image to a new location
img.save(save_path)
print(f"Image saved to {save_path}")

# except Exception as e:
#     print(f"An error occurred: {e}")



