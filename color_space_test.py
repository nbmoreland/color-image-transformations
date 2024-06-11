# Nicholas Moreland
# 1001886051
# 02/06/2024

import sys
import numpy as np
from PIL import Image

# Function to convert an RGB image to HSV
def rgb_to_hsv(image_array):
    # Normalize the RGB values
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    r, g, b = r / 255, g / 255, b / 255

    # Calculate the value, chroma, and saturation
    v = np.max(image_array, axis=2)
    chroma = v - np.min(image_array, axis=2)
    s = np.where(v != 0, chroma / v, 0)

    # Calculate the hue based on the peicewise conditions
    hue_comp = np.where((chroma != 0) & (v == r), ((g - b) / chroma) % 6, 0)
    hue_comp = np.where((chroma != 0) & (v == g), ((b - r) / chroma) + 2, hue_comp)
    hue_comp = np.where((chroma != 0) & (v == b), ((r - g) / chroma) + 4, hue_comp)
    
    # Calculate the hue
    h = 60 * hue_comp
    return np.stack((h, s, v), axis=2)

# Function to convert an HSV image to RGB
def hsv_to_rgb(image_array):
    # Calculate the chroma, hue component, and x
    h, s, v = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    chroma = v * s
    hue_comp = h / 60
    x = chroma * (1 - np.abs((hue_comp % 2) - 1))

    # Calculate the RGB components based on the piecewise conditions
    cond1 = (hue_comp >= 0) & (hue_comp < 1)
    cond2 = (hue_comp >= 1) & (hue_comp < 2)
    cond3 = (hue_comp >= 2) & (hue_comp < 3)
    cond4 = (hue_comp >= 3) & (hue_comp < 4)
    cond5 = (hue_comp >= 4) & (hue_comp < 5)
    cond6 = (hue_comp >= 5) & (hue_comp < 6)

    # Calculate the RGB components
    R_comp = np.select([cond1, cond2, cond3, cond4, cond5, cond6],
                      [chroma, x, 0, 0, x, chroma])
    G_comp = np.select([cond1, cond2, cond3, cond4, cond5, cond6],
                      [x, chroma, chroma, x, 0, 0])
    B_comp = np.select([cond1, cond2, cond3, cond4, cond5, cond6],
                      [0, 0, x, chroma, chroma, x])

    # Calculate the RGB values
    m = v - chroma
    r = R_comp + m
    g = G_comp + m
    b = B_comp + m
    return np.stack((r, g, b), axis=2)

# Main function
def main(argv, argc):
    # Check the number of arguments
    if argc != 5:
        print("Usage: python3 color_space_test.py <filename> <hue> <saturation> <value>")
        exit(0)

    # Get the filename and the HSV values
    filename = argv[1]
    hue = float(argv[2])
    saturation = float(argv[3])
    value = float(argv[4])

    # Check the HSV values
    if hue < 0 or hue > 360:
        print("The Hue input is out of the range [0, 360]")
        exit(0)
    elif saturation < 0 or saturation > 1:
        print("The Saturation input is out of the range [0, 1]")
        exit(0)
    elif value < 0 or value > 1:
        print("The Value input is out of the range [0, 1]")
        exit(0)

    # Open the image
    im = Image.open(filename)
    image_array = np.array(im)

    # Convert to HSV
    image_array = rgb_to_hsv(image_array)

    # Modify HSV values
    image_array[:,:,0] += hue
    image_array[:,:,1] += saturation
    image_array[:,:,2] += value

    # Convert back to RGB
    image_array = hsv_to_rgb(image_array)

    # Ensure the values are in the valid range [0, 255] and convert to uint8
    image_array = (image_array * 255).astype(np.uint8)

    # Save the modified image
    im = Image.fromarray(image_array)
    im.save("newimage.png")

if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
