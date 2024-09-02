import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np

from network import Generator




char_to_label = {}

# Add digits (0-9)
for i in range(10):
    char_to_label[str(i)] = i

# Add uppercase letters (A-Z)
for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ', start=10):
    char_to_label[char] = i

# Add lowercase letters (a-z)
for i, char in enumerate('abcdefghijklmnopqrstuvwxyz', start=36):
    char_to_label[char] = i

char_to_label[' '] = 62


def calibrate_label(input_label):
    labels = [char_to_label[char] for char in input_label]
    return labels





def generate_character(generator, label):
    z = torch.randn(1, 100).cuda() # Random noise
    label_tensor = torch.LongTensor([label]).cuda()
    with torch.no_grad():
        img = generator(z, label_tensor).squeeze().cpu().numpy()
    img = 0.5 * img + 0.5  # Rescale image to [0, 1]
    img = cv2.flip(img, 0)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate to get the correct orientation
    return img


def generate_plot_a4(characters, model_path):

    #characters = calibrate_label(characters)
    if isinstance(model_path, str):
        generator = Generator()
        generator.load_state_dict(torch.load(model_path))
        generator.cuda()
    else:
        generator = model_path






    # Create a blank A4-sized canvas (210mm x 297mm converted to pixels, assuming 100 DPI)
    dpi = 100
    a4_width = int(210 * dpi / 25.4)
    a4_height = int(297 * dpi / 25.4)
    canvas = np.ones((a4_height, a4_width)) * 255  # White background


    # Set starting position for the first character
    x, y = 50, 50
    font_size = 20  # Size of each character image

    
    # Generate and place each character on the canvas
    for char in characters:
        if char == ' ':
            x += font_size  # Add space between words
            continue
        label = char_to_label[char]
        img = generate_character(generator, label)
        img = 1 - img
        img_resized = cv2.resize(img, (font_size, font_size))  # Resize to match font size

    # Place the character image on the canvas
        canvas[y:y+font_size, x:x+font_size] = img_resized * 255
    #x += font_size + 0  # Move to the next position
        x += 15  # Move to the next position

    # If the end of the line is reached, move to the next line
        if x + font_size > a4_width:
            x = 50
            y += font_size + 10

    # Plot the final simulated handwriting on A4 paper
    plt.figure(figsize=(11.69, 8.27))  # A4 in landscape
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')  # Turn off axis for clean output
    plt.show()