# Nicholas Moreland
# 1001886051
# 02/06/2024

from img_transforms import resize_img
from PIL import Image
import numpy as np
import sys
import os

# height is the number of levels in the pyramid
def create_image_pyramid(img, height):
    # Open the image
    image = np.asarray(Image.open(img))

    # Get the image name and extension
    image_name, ext = os.path.splitext(img)

    # Loop through the number of levels in the pyramid
    for i in range(1, height):
        # Scale the image
        scale = 2 ** i
        resized_image = resize_img(image, 1 / scale)

        # Save the image
        resized_image.save(f"{image_name}_{scale}x{ext}")
        print(f"Image saved as {image_name}_{scale}x{ext}")

def main(argv, argc):
    # Check the number of arguments
    if argc != 3:
        print("Usage: python3 create_img_pyramid.py <filename> <height>")
        exit(0)

    # Get the filename 
    filename = argv[1]

    # Pyramid height
    height = int(argv[2])

    # Create the image pyramid
    create_image_pyramid(filename, height)

# $ python3 create_img_pyramid.py <filename> <height>
if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
