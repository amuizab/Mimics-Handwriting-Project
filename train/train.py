import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from network import Generator, Discriminator


augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=augmentation_transforms, num_augmentations=1000):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx):
        # Determine the original image index
        original_idx = idx // self.num_augmentations
        image_path = self.image_paths[original_idx]
        label = self.labels[original_idx]

        # Load image
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply transformation (augmentation)
        if self.transform:
            image = self.transform(image)

        return image, label
    


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 62, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())

    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 62, batch_size))).cuda()
    fake_images = generator(z, fake_labels)


    fake_validity = discriminator(fake_images, fake_labels)

    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def fine_tune_model(dataloader):
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    generator.load_state_dict(torch.load('models/generator.pt'))
    discriminator.load_state_dict(torch.load('models/discriminator.pt'))

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    # Freeze most layers of the generator
    for param in generator.model[:-2].parameters():
        param.requires_grad = False

    # Freeze most layers of the discriminator
    for param in discriminator.model[:-2].parameters():
        param.requires_grad = False

    # Keep the embedding layers trainable
    generator.label_emb.requires_grad = True
    discriminator.label_emb.requires_grad = True

    writer = SummaryWriter()

    batch_size = 128
    num_epochs = 50
    n_critic = 3
    display_step = 50
    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        for i, (images, labels) in enumerate(dataloader):

            step = epoch * len(dataloader) + i + 1
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            generator.train()

            d_loss = 0
            
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                                generator, d_optimizer, criterion,
                                                real_images, labels)

            for _ in range(n_critic):
                g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': (d_loss / n_critic)}, step)

            if step % display_step == 0:
                generator.eval()
                z = Variable(torch.randn(9, 100)).cuda()
                labels = Variable(torch.LongTensor(np.arange(9))).cuda()
                sample_images = generator(z, labels).unsqueeze(1)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)
        print('Done!', g_loss, d_loss)
    
    return generator