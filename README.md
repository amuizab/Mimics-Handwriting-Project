# Mimics Handwriting Generation
The handwriting generation pipeline begins with loading and preprocessing the EMNIST dataset, followed by training a cGAN with a Generator and Discriminator. User handwriting samples are collected, preprocessed, and used to fine-tune the cGAN, adapting it to the user’s style. Input text is tokenized, normalized, and combined with the user’s handwriting style to generate personalized handwritten text images. The final output is rendered on virtual A4 paper. Continuous improvement is achieved by periodically retraining the model with new user data, ensuring the generated handwriting remains accurate and personalized.

## Pipeline
1. Data Preparation:
    - Load and preprocess EMNIST dataset
    - Augment data if necessary (e.g., rotations, slight distortions)
2. Initial Model Training:
    - Train Generator network
    - Train Discriminator network
    - Implement adversarial training loop for cGAN
3. User Input Development:
    - Create input mechanism for desired text
4. User Handwriting Collection:
    - Guide user to provide handwriting samples / Image for now
    - Preprocess and normalize user samples
5. Model Fine-tuning:
    - Create a small dataset from user's handwriting samples
    - Fine-tune pre-trained cGAN model on user's dataset
    - Freeze parameters from the pretrained Model
6. Text Preprocessing:
    - Tokenize and normalize input text
    - Handle characters and numerals
7. Handwriting Generation:
    - Feed preprocessed text and style parameters into fine-tuned cGAN
    - Generate mimics handwriting image
8. Output Rendering:
    - Create final handwritten text image on A4 paper
9. Continuous Improvement:
    - Periodically retrain model with new user data

## Demo
Generated with trained Generator
![number generated](https://github.com/user-attachments/assets/d293abe0-2c13-49c9-8f20-d97afd3d8434)
![generated](https://github.com/user-attachments/assets/c8e2c87a-0254-40aa-b6a1-61a6b8c30c77)


## Next Development
    - Gather more hand writing user image for each label so that it could generate better
    - Develop tool for capturing user's handwriting samples
    - Instead of feeding images to the training custom dataset, we could try the user input (handwriting) and develop a model to detect each character belongs to which label and use it as a dataset for finetune mimics handwriting generation
    - Apply image enhancements (e.g., smooth connections between letters)
