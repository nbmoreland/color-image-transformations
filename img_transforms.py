# Nicholas Moreland
# 1001886051
# 02/06/2024

from color_space_test import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import numpy as np
import sys
from matplotlib import pyplot as plt

# Function to generate a random square crop of an image
def random_crop(img, size):
    # Get the dimensions of the image
    height, width, c = img.shape

    # Check if the size is within the range of the image
    if size not in range(0, (min(height, width) + 1)):
        print(f"Size should be between 0 and {min(height, width)}")
        return None

    # Generate random coordinates for the crop
    x = np.random.randint(size / 2, (width - size / 2) + 1)
    y = np.random.randint(size / 2, (height - size / 2) + 1)

    # Calculate the coordinates for the crop
    left = int(x - size / 2)
    right = int(x + size / 2)
    top = int(y - size / 2)
    bottom = int(y + size / 2)

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    return cropped_img

# Function to extract patches from an image
def extract_patch(img, num_patches):
    # Get the dimensions of the image
    height, width, c = img.shape

    # Check if the number of patches is within the range of the image
    size = int(height / (num_patches / 2))
    shape = [height // num_patches, width // num_patches] + [size, size, 3]

    # Generate patches from the image
    strides = [size * s for s in img.strides[:2]] + list(img.strides)
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    return patches

# Helper function to generate a set of indices for a given axis
def per_axis(in_sz, out_sz):
    # Generate a set of indices for a given axis
    if out_sz == 0:
        return np.array([])

    ratio = 0.5 * in_sz / out_sz
    return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

# Function to resize an image
def resize_img(img, factor):
    shape = [img.shape[0], img.shape[1]]

    rows = per_axis(img.shape[0], int(shape[0] * factor))
    cols = per_axis(img.shape[1], int(shape[1] * factor))

    # Ensure indices are within bounds
    rows = np.clip(rows, 0, img.shape[0] - 1)
    cols = np.clip(cols, 0, img.shape[1] - 1)

    resized_array = img[rows[:, None], cols]

    resized_image = Image.fromarray(resized_array.astype(np.uint8))

    return resized_image

# Function to apply color jitter to an image
### This function is not implemented - ran out of time
def color_jitter(img, hue, saturation, value):
    # Convert the image to HSV
    hsv_img = rgb_to_hsv(np.array(img))

    # Randomly perturb the HSV values
    hsv_img[:, :, 0] += np.random.uniform(-hue, hue)
    hsv_img[:, :, 1] += np.random.uniform(-saturation, saturation)
    hsv_img[:, :, 2] += np.random.uniform(-value, value)

    # Clip values to ensure they stay within valid HSV range
    hsv_img = np.clip(hsv_img, 0, 1)

    # Convert the modified HSV image back to RGB
    jittered_img = Image.fromarray((hsv_to_rgb(hsv_img) * 255).astype(np.uint8))

    return jittered_img

# Main function
def main(argv, argc):
    # Check the number of arguments
    if argc < 3:
        print("Usage: python3 img_transforms.py <filename> <option> <args>")
        exit(0)

    # Get the filename and option
    filename = argv[1]
    option = argv[2]

    # Open the image
    image = np.asarray(Image.open(filename))

    # Perform the specified operation
    if option == "random_crop":
        crop = random_crop(image, int(argv[3]))
        if crop is not None:
            plt.imshow(crop)
            plt.show()
    elif option == "extract_patch":
        size = int(argv[3])
        patches = extract_patch(image, size)

        # Display the patches
        rows = int(size / 2 )
        cols = int(size / 2 )

        fig = plt.figure()
        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                x = fig.add_subplot(rows, cols, i + 1)
                x.imshow(patches[row, col, :, :])
                x.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig.savefig('patches.png')
        plt.show()
    elif option == "resize_img":
        factor = int(sys.argv[3])
        resized = resize_img(image, factor)
        plt.imshow(resized)
        plt.show()
    elif option == "color_jitter":
        jittered = color_jitter(image, float(argv[3]), float(argv[4]), float(argv[5]))
        plt.imshow(jittered)
        plt.show()

# $ python3 img_transforms.py <filename> <option> <args>
if __name__ == "__main__":
    main(sys.argv, len(sys.argv))