#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from torchvision.io.video import re
from tqdm import tqdm
import csv
import timm
import wandb

from PIL import Image
import torchvision.transforms.v2 as transforms

# UTILITIES

def extract_first_frame(video_path):
    reader = io.VideoReader(video_path, "video")
    frame = next(reader)['data']  # Read the first frame
    return frame.unsqueeze(0)  # Add a batch dimension

def smart_resize(data, size): # kudos louis
    # Prends un tensor de shape [...,C,H,W] et le resize en [...C,size,size]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]

    if full_height > full_width:
        alt_height = size
        alt_width = int(full_width / (full_height / size))
    elif full_height < full_width:
        alt_height = int(full_height / (full_width / size))
        alt_width = size
    else:
        alt_height = size
        alt_width = size
    tr = transforms.Compose([
        transforms.Resize((alt_height, alt_width)),
        transforms.CenterCrop(size)
    ])
    return tr(data)

def resize_data(data, new_height, new_width, x=0, y=0, height=None, width=None):
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]
    height = full_height - y if height is None else height
    width = full_width -x if width is None else width

    ratio = new_height/new_width
    if height/width > ratio:
        expand_height = height
        expand_width = int(height / ratio)
    elif height/width < ratio:
        expand_height = int(width * ratio)
        expand_width = width
    else:
        expand_height = height
        expand_width = width
    tr = transforms.Compose([
        transforms.CenterCrop((expand_height, expand_width)),
        transforms.Resize((new_height, new_width))
    ])
    x = data[...,y:min(y+height, full_height), x:min(x+width, full_width)].clone()
    return tr(x)


# SETUP DATASET

dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
nb_frames = 10

## MAKE RESIZED DATASET
resized_dir = os.path.join(dataset_dir, "resized_dataset")
"""
create_small_dataset = False
errors = []
if not os.path.exists(resized_dir) or create_small_dataset:
    os.mkdir(resized_dir)
    os.mkdir(os.path.join(resized_dir, "train_dataset"))
    os.mkdir(os.path.join(resized_dir, "test_dataset"))
    os.mkdir(os.path.join(resized_dir, "experimental_dataset"))
    train_files = [f for f in os.listdir(os.path.join(dataset_dir, "train_dataset")) if f.endswith('.mp4')]
    test_files = [f for f in os.listdir(os.path.join(dataset_dir, "test_dataset")) if f.endswith('.mp4')]
    experimental_files = [f for f in os.listdir(os.path.join(dataset_dir, "experimental_dataset")) if f.endswith('.mp4')]
    def resize(in_video_path, out_video_path, nb_frames=10):
        video = extract_frames(in_video_path, nb_frames=nb_frames)
        t1 = time.time()
        #video, audio, info = io.read_video(in_video_path, pts_unit='sec', start_pts=0, end_pts=10, output_format='TCHW')
        video = smart_resize(video, 256)
        t2 = time.time()
        torch.save(video, out_video_path)
        t3 = time.time()
        print(f"resize: {t2-t1}\nsave: {t3-t2}")
        #video = video.permute(0,2,3,1)
        #io.write_video(video_path, video, 15, video_codec='h264')

    
    for f in tqdm(train_files):
        in_video_path = os.path.join(dataset_dir, "train_dataset", f)
        out_video_path = os.path.join(resized_dir, "train_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from train")
    
    for f in tqdm(test_files):
        in_video_path = os.path.join(dataset_dir, "test_dataset", f)
        out_video_path = os.path.join(resized_dir, "test_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from test")
    for f in tqdm(experimental_files):
        in_video_path = os.path.join(dataset_dir, "experimental_dataset", f)
        out_video_path = os.path.join(resized_dir, "experimental_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from experimental")
    os.system(f"cp {os.path.join(dataset_dir, 'train_dataset', 'metadata.json')} {os.path.join(resized_dir, 'train_dataset', 'metadata.json')}")
    os.system(f"cp {os.path.join(dataset_dir, 'dataset.csv')} {os.path.join(resized_dir, 'dataset.csv')}")
    os.system(f"cp {os.path.join(dataset_dir, 'experimental_dataset', 'metadata.json')} {os.path.join(resized_dir, 'experimental_dataset', 'metadata.json')}")
    if errors:
        print(errors)
"""
use_small_dataset = True
if use_small_dataset:
    dataset_dir = resized_dir

nb_frames = 10


# Modified VideoDataset Class
class VideoDataset(Dataset):
    def __init__(self, root_dir, dataset_choice="train"):
        self.root_dir = os.path.join(root_dir, f"{dataset_choice}_dataset")
        with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
            self.data = json.load(file)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize each frame to 256x256
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        frame = extract_first_frame(video_path)
        frame = self.transform(frame)  # Apply transformations
        label = torch.tensor(float(self.data[self.video_files[idx]] == 'fake'))
        return frame.squeeze(0), label  # Remove batch dimension and return frame and label



#train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
train_dataset = VideoDataset(dataset_dir, dataset_choice="train")

#test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test")

#experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)
experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental")


# MODELE

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN4(nn.Module):
    def __init__(self):
        in_channels = 3
        out_channels = 32
        k_size = 3
        stride_ = 1
        padding_ = 1
        pool_k_size = 2
        pool_stride = 2
        pool_padding = 0
        dropout_rate = 0.5

        

        super(EnhancedCNN4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= k_size, stride=stride_, padding=padding_)
        self.bn1 = nn.BatchNorm2d(out_channels)


        in_channels = out_channels
        out_channels = out_channels*2

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size= k_size, stride= stride_, padding=padding_)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool1 = nn.MaxPool2d(kernel_size=pool_k_size, stride=pool_stride)

        in_channels = out_channels
        out_channels = out_channels*2

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride_, padding=padding_)
        self.bn3 = nn.BatchNorm2d(out_channels)

        in_channels = out_channels
        out_channels = out_channels*2

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride_, padding=padding_)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.pool2 = nn.MaxPool2d(kernel_size=pool_k_size, stride=pool_stride)
        



        # Calculate the size of the output from the last pooling layer
        def calc_output_dim(input_dim, kernel_size, stride, padding):
            return (input_dim - kernel_size + 2 * padding) // stride + 1
        
        #Initial dimension of the data is 64
        dim = 64
        # After conv1
        dim = calc_output_dim(dim, k_size, stride_, padding_)      
        # After conv2
        dim = calc_output_dim(dim, k_size, stride_, padding_)
        # After pool1
        dim = calc_output_dim(dim, pool_k_size, pool_stride, pool_padding)   
        # After conv3
        dim = calc_output_dim(dim, k_size, stride_, padding_)
        # After conv4
        dim = calc_output_dim(dim, k_size, stride_, padding_)

        # After pool2
        dim = calc_output_dim(dim, pool_k_size, pool_stride, pool_padding)          

        self.dropout = nn.Dropout(dropout_rate)
    
        self.fc = nn.Linear(in_features= out_channels*dim*dim, out_features=1024)
        
        #out_features is the number of classes we want to predict, here Cat and Dog so 2 classses
        self.fc2 = nn.Linear(in_features=1024 , out_features=2)
        

    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = (self.fc2(x))
        return x



# LOGGING

wandb.login(key="b15da3ba051c5858226f1d6b28aee6534682d044")
run = wandb.init(
    project="authomathon Deep Fake Detection Otho Local",
)
# ENTRAINEMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

loss_fn = nn.MSELoss()
model = EnhancedCNN4_3D().to(device)
#model = DeepfakeDetector().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#epochs = 5
epochs = 1
loader = DataLoader(experimental_dataset, batch_size=2, shuffle=True)

for epoch in range(epochs):
    for sample in tqdm(loader, desc="Epoch {}".format(epoch), ncols=0):
        optimizer.zero_grad()
        X, label, ID = sample
        X = X.permute(0, 2, 1, 3, 4).to(device)  # Adjusting dimension order and moving to device
        X = X.to(device)
        label = label.to(device)
        #X = X.cuda()
        #label = label.cuda()
        label_pred = model(X)
        label=torch.unsqueeze(label,dim=1)
        loss = loss_fn(label, label_pred)
        loss.backward()
        optimizer.step()
        run.log({"loss": loss.item(), "epoch": epoch})

## TEST

loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = model.to(device)
ids = []
labels = []
print("Testing...")
for sample in tqdm(loader):
    X, ID = sample
    #ID = ID[0]
    X = X.to(device)
    label_pred = model(X)
    ids.extend(list(ID))
    pred = (label_pred > 0.5).long()
    pred = pred.cpu().detach().numpy().tolist()
    labels.extend(pred)

### ENREGISTREMENT
print("Saving...")
tests = ["id,label\n"] + [f"{ID},{label_pred[0]}\n" for ID, label_pred in zip(ids, labels)]
with open("submissionCNN2D.csv", "w") as file:
    file.writelines(tests)
