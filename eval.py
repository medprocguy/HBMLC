import numpy as np
import torch, random
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataloader import build
from sklearn.metrics import hamming_loss

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

# define configs and create config
config = {
    # optimization configs
    'seed': 2024,
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

eval_transform = transforms.Compose([transforms.Resize(384),
                                      transforms.ToTensor()
                 ])                  

val_dataset = build('val', config, eval_transform)                 
test_dataset = build('test', config, eval_transform)

# Create DataLoader instances for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.fc = nn.Linear(2048, config['num_classes'])
model.to(device)

checkpoint_path = 'saved_data/best_model.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()

hl, idx = 0, 0
pred_vals, gt_vals = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        idx += 1
        print(idx, '/', len(val_loader), '>')
        inputs, labels = inputs.to(device), labels.to(device)
            
        outputs = model(inputs.float())
        logits = torch.where(outputs.sigmoid() > 0.5, 1, 0) 
        pred_vals.append(outputs.squeeze().cpu().data.numpy())
        gt_vals.append(labels.squeeze().cpu().data.numpy())
        hl += hamming_loss(labels.cpu().data.numpy(), logits.cpu().data.numpy())

acc = 1 - (hl/len(val_loader))
print('Accuracy:',acc)
print(hl, len(val_loader))
print(np.array(pred_vals).shape, np.array(gt_vals).shape)
np.save('ChestMNIST_raw_s2024/val_384_pred_s2024.npy', np.array(pred_vals))
np.save('ChestMNIST_raw_s2024/val_384_gt_s2024.npy', np.array(gt_vals)) 
