#create dataset and dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np 
from PIL import Image


class Cifar_10(Dataset):
    def __init__(self, root, phase, transform="None"):
        self.root = root
        self.transform = transform
        classes = os.path.join(root, "train")
        self.classes = [path for path in os.listdir(classes)]
        if phase == "train":
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.image_path = []
        self.labels = []
        for class_id, class_name in enumerate(self.classes):
            sub_foler_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(sub_foler_path):
                image_path = os.path.join(sub_foler_path, image_name)
                self.image_path.append(image_path)
                self.labels.append(class_id)
                
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label 



if __name__== "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  
    ])
    dataset = Cifar_10(root="/Users/user1/Downloads/data", phase="train", transform=transform)
    training_dataloader = DataLoader(
        dataset=dataset,
        batch_size=8, 
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    for image, label in training_dataloader:
        print(label)