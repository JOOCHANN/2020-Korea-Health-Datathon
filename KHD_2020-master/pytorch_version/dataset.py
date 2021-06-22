import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from RandAugment import RandAugment

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
size = 600

#train_data
def train_transform(Rsize):
    transforms_train = transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        transforms.RandomCrop(Rsize, padding=30),  # 이미지를 Crop한 뒤에 빈 곳을 padding함
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)), # translate a = width(0.1), b = height(0)
        transforms.ColorJitter(brightness=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    #transforms_train.transforms.insert(0, RandAugment(2, 14))
    return transforms_train

#test_data
def test_transform(Rsize):
    return transforms.Compose([
        transforms.Resize((Rsize, Rsize), interpolation=Image.BICUBIC), #이미지 크기를 Rsize, Rsize로 Resizing함
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

class PathDataset(Dataset): 
    def __init__(self,image_path, labels=None, test_mode=True): 
        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode
        if test_mode == False:
            print("number of training data :", self.len)
            self.transform = train_transform(Rsize=size)
        else:
            print("number of test data :", self.len)
            self.transform = test_transform(Rsize=size)

    def __getitem__(self, index): 
        im = Image.open(self.image_path[index])
        im = self.transform(im)

        if self.mode:
            return im
        else:
            return im, torch.tensor(self.labels[index] ,dtype=torch.long)

    def __len__(self): 
        return self.len