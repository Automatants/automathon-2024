import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from tqdm import tqdm
import csv
import timm
import wandb
import time
 
from PIL import Image
import torchvision.transforms as transforms
 
import matplotlib.pyplot as plt
 
def display_image(img):
    img = img.permute(1,2,0)
    plt.imshow(img)
def extract_frames(video_path, nb_frames=10, delta=1, timeit=False):
    # use time to measure the time it takes to resize a video
    t1 = time.time()
    reader = io.VideoReader(video_path)
    # take 10 frames uniformly sampled from the video
    frames = []
    for i in range(nb_frames):
        reader.seek(delta)
        frame = next(reader)
        frames.append(frame['data'])
    t2 = time.time()     
    video = torch.stack(frames)
    if timeit:
        print(f"read: {t2-t1}")
    return video
 
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
    # Prends un tensor de shape [...,C,H,W] et le resize en [...,C,new_height,new_width]
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
 
dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
 
nb_frames = 10
 
class VideoDataset(Dataset):
    """
    This Dataset takes a video and returns a tensor of shape [10, 3, 256, 256]
    That is 10 colored frames of 256x256 pixels.
    """
    def __init__(self, root_dir, dataset_choice="train", nb_frames=10):
        super().__init__()
        self.dataset_choice = dataset_choice
        if  self.dataset_choice == "test":
            self.root_dir = os.path.join(root_dir, "test_dataset")
        elif  self.dataset_choice == "experimental":
            self.root_dir = os.path.join(root_dir, "train")
        else:
            raise ValueError("choice must be 'test' or 'experimental'")
 
        with open(os.path.join(root_dir, "dataset.csv"), 'r') as file:
            reader = csv.reader(file)
            # read dataset.csv with id,label columns to create
            # a dict which associated label: id
            self.ids = {row[1] : row[0] for row in reader}
 
        if self.dataset_choice == "test":
            self.data = None
        else:
            with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
                self.data= json.load(file)
                self.data = {k : (torch.tensor(float(1)) if v == 'fake' else torch.tensor(float(0))) for k, v in self.data.items()}
 
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        #self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]
 
    def __len__(self):
        return len(self.video_files)
 
    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        #video, audio, info = io.read_video(video_path, pts_unit='sec')
        #video = torch.load(video_path)
        video = extract_frames(video_path)
        #video = video.permute(0,3,1,2)
        #length = video.shape[0]
        #video = video[[i*(length//(nb_frames)) for i in range(nb_frames)]]
        # resize the data into a reglar shape of 256x256 and normalize it
        video = smart_resize(video, 256) / 255
        #video = video / 255
 
        ID = self.ids[self.video_files[idx]]
        if self.dataset_choice == "test":
            return video, ID
        else:
            label = self.data[self.video_files[idx]]
            return video, label, ID
 
 
train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)
 
video, label, ID = experimental_dataset[10]
img = video[0]
 
display_image(img)
print(label)
print(video.shape)
 
video, label, ID = experimental_dataset[0]
img=video[0]
 
img=smart_resize(img, 256)
print(img.shape)
display_image(img)


# MODELE

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Encoder (utilise ResNet50 pré-entraîné)
        resnet = timm.create_model('resnet50', pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            UNetBlock(2048, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            UNetBlock(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            UNetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            UNetBlock(128, 128)
        )
        
        # Classification binaire
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Encoder
        x = self.encoder(x[:, :, 0])
        # Decoder
        x = self.decoder(x)
        # Classification binaire
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


# LOGGING

wandb.login(key="b15da3ba051c5858226f1d6b28aee6534682d044")
run = wandb.init(
    project="authomathon Deep Fake Detection Otho Local",
)


# ENTRAINEMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
loss_fn = nn.MSELoss()
model = UNet(1).to(device)
print("Training model:")
summary(model, input_size=(batch_size, 3, 10, 256, 256))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#loader = DataLoader(experimental_dataset, batch_size=2, shuffle=True)

print("Training...")
for epoch in range(epochs):
    for sample in tqdm(loader):
        optimizer.zero_grad()
        X, label, ID = sample
        X = X.to(device)
        label = label.to(device)
        label_pred = model(X)
        label = torch.unsqueeze(label,dim=1)
        loss = loss_fn(label, label_pred)
        loss.backward()
        optimizer.step()
        run.log({"loss": loss.item(), "epoch": epoch})

## TEST

loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#model = model.to(device)
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
with open("submissionUNET_003.csv", "w") as file:
    file.writelines(tests)
