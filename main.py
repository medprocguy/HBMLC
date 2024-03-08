import numpy as np
import torch, random
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataloader import build

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

# define configs and create config
config = {
    # optimization configs
    'seed': 0,
    'epoch': 100,  # set to 100
    'num_log_iter': 50,    # set to 256
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 32,
    'eval_batch_size': 32,
    'test_batch_size': 1,

    # dataset configs
    'dataset': 'ChestMNIST',
    'num_classes': 14,
}

# fix the seed for reproducibility
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_transform = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(degrees=(-10, 10)),
                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                      transforms.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0)),
                      transforms.ToTensor()
                  ])
eval_transform = transforms.Compose([transforms.Resize(384),
                                      transforms.ToTensor()
                 ])                  
                 
train_dataset = build('train', config, train_transform)
val_dataset = build('val', config, eval_transform)

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], shuffle=False)

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.fc = nn.Linear(2048, config['num_classes'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_val_loss = 9999
for epoch in range(config['epoch']):
    model.train()
    tr_loss, val_loss = 0.0, 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = nn.functional.binary_cross_entropy(outputs.sigmoid(), labels)
        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{config['epoch']}, Loss: {tr_loss/len(train_loader)}")    
    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs.float())
            loss = nn.functional.binary_cross_entropy(outputs.sigmoid(), labels)
            val_loss += loss.item() 
    
    val_loss = val_loss / len(val_loader)  
    
    save_on_master({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': config,
        'val_acc': val_loss,
    }, 'saved_data/latest_model.pth')       
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss    
        save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': config,
            'val_acc': val_loss,
        }, 'saved_data/best_model.pth')        

    

