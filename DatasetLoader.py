import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class DatasetLoader(Dataset):

	def __init__(self, csv_path):
		self.to_tensor = transforms.ToTensor()
		# load the csv file
		self.data_info = pd.read_csv(csv_path, header=None)
		# load the image pages into a numpy array
		self.image_arr = np.asarray(self.data_info.iloc[:, 0])
		# load the labels into a numpy array
		self.label_arr = np.asarray(self.data_info.iloc[:, 1])
		# length of the data
		self.data_len = len(self.data_info.index)

	def __getitem__(self, index):
		# get image
		single_image_name = self.image_arr[index]
		# opent the image
		image = Image.open(single_image_name)

		image = transforms.Resize((224,224))

		# transform image to a tensor
		image_as_tensor = self.to_tensor(image)

		# get label
		single_image_label = self.label_arr[index]

		return (image_as_tensor, single_image_label)

	def __len__(self):
		return self.data_len