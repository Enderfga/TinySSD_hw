import os
import pandas as pd
import torch
import torchvision
def read_data(is_train=True):
    data_dir = 'data/detection'
    csv_fname = os.path.join(data_dir, 'sysu_train' if is_train else 'sysu_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images,targets =[],[]
    for img_name, img_target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir, 'sysu_train' if is_train else 'sysu_val', 'images', f'{img_name}')))
        targets.append(list(img_target))
    return images,torch.tensor(targets).unsqueeze(1)/256
class Dataset(torch.utils.data.Dataset):
    """Dataset."""
    def __init__(self, is_train, transform=None):
        self.features,self.labels = read_data(is_train)
        self.transform = transform
        print('read ' + str(len(self.features)) +(f' train examples' if is_train else f'validation examples'))
    def __getitem__(self, idx):
        #if self.transform:
        #   self.features[idx] = self.transform(self.features[idx])
        return self.features[idx].float(), self.labels[idx]
    def __len__(self):
        return len(self.features)
# 数据增强
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(15)
])
def load_data(batch_size):
    """Load the data and return a data loader."""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True,transform=transform), batch_size=batch_size, shuffle=True)
    #val_iter = torch.utils.data.DataLoader(Dataset(is_train=False), batch_size=batch_size, shuffle=False)
    return train_iter #, val_iter
