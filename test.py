import torch
import torchvision
from model.TinySSD import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = TinySSD(num_classes=1)
net = net.to(device)

# 加载模型参数
net.load_state_dict(torch.load('model/checkpoints/net_50.pkl', map_location=torch.device(device)))

      
name = 'data/detection/test/1.jpg'
X = torchvision.io.read_image(name).unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

output = predict(X,net,device)    
display(img, output.cpu(), threshold=0.3)