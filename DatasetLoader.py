import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class DatasetLoader(Dataset):

    def __init__(self, csv_path, dims):
        self.to_tensor = transforms.ToTensor()
        # resize image function
        self.resize = transforms.Resize(dims)
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_name = self.image_arr[index]
        # Open image
        image = Image.open(image_name)

        image = self.resize(image)

        # Transform image to tensor
        torch_image = self.to_tensor(image)

        # Get label(class) of the image based on the cropped pandas column
        label = self.label_arr[index]

        data = {'image': torch_image, 'label': label}

        return data

    def __len__(self):
        return self.data_len

