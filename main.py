from nips2012 import AlexNet
from data import Cifar_10
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.optim import SGD
import torch

#note the difference between torch.tensor and numpy.tensor

def train():
    data_path = "/Users/user1/Downloads/data"
    batch_size = 8
    lr = 1e-2
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  
    ])
    
    train_dataset = Cifar_10(root=data_path, phase="train", transform=transform)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, 
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    num_iters = len(dataloader)
    
    for epoch in range(num_epochs):
        for iteration, (images, labels) in enumerate(dataloader):
            #forward
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss_value = criterion(output, labels)
            #backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Iter {iteration+1}/{num_iters}, Loss {loss_value.item():.4f}")
    torch.save(model.state_dict(), "best.pt")

def test():
    datapath = "/Users/user1/Downloads/data"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    #wwhy do this step in training and testing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    test_dataset = Cifar_10(root=datapath, phase="test", transform=transform)
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, 
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    avg_loss = test_loss/len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    

if __name__ == "__main__":
    train()
    test()
    
