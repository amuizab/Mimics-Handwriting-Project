import os
import numpy as np
from torch.utils.data import DataLoader

from train import CustomDataset, fine_tune_model
from char_n_plot import calibrate_label, generate_plot_a4




def generate_default():
    model_path = 'models/generator.pt'
    print('generating')
    output_desired = input('Type Desired Output Character: ')
    print(output_desired)

    generate_plot_a4(output_desired, model_path)
#input characters
# generate and plot




def mimics_handwrite_main():
    print('mimics')
    # output_desired = input('Type Desired Output Character: ')
    # print(output_desired)
    # image_paths = input('Image Folder Path: ')
    # print(image_paths)
    # label_map = input('Label Map: ')
    # print(label_map)

    output_desired = 'aku lapar sekaly 12345'


    image_paths = 'image_train'

    list_image = []

    for filename in os.listdir(image_paths):
        if os.path.isfile(os.path.join(image_paths, filename)):
            temp = image_paths + '/' + filename
            list_image.append(temp)

    print('-------------------------------------')
    print(list_image)



    label_map = '1,2,3,4,5'


    label_map = label_map.split(',')
    label_map = np.array(label_map)

    labels = calibrate_label(label_map)

    #augmented_dataset = CustomDataset(image_paths, labels, transform=augmentation_transforms, num_augmentations=num_augmentations)
    augmented_dataset = CustomDataset(list_image, labels)

    # Create DataLoader for the augmented dataset
    batch_size = 128
    dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

    x,y = next(iter(dataloader))
    print(y)
    
    model_path = fine_tune_model(dataloader)

    generate_plot_a4(output_desired, model_path)

    
################################################# FINETUNE
#input characterssssssssssssssssssss (desired output handwriting)
#input image path [img1,img2, img3, ...]
#input label map -> def char to label

#call custom dataset
#call training -> freeze parameters
# call generate a4 with new generator


def main():
    print(" 1. Generate Handwrite\n",
          "2. Mimics Your Handwrite")

    decision = input('Pick A Number:')
    print(f"You Choose, {decision}!")

    if int(decision) == 1:
        generate_default()
    elif int(decision) == 2:
        mimics_handwrite_main()
    else:
        print("Invalid Input")

if __name__ == '__main__':
    main()


