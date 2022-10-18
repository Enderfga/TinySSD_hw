from importlib.resources import path
import torch
from data.dataloader import load_data
import torchvision
from model.TinySSD import *
from utils.utils import *
import argparse

def train(net,device,batch_size,epochs,path):
    train_iter = load_data(batch_size)
    if path:
        net.load_state_dict(torch.load(path, map_location=torch.device(device)))

    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    num_epochs = epochs
    for epoch in range(num_epochs):
        print('epoch: ', epoch+1)
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                        bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                        bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


        # 保存模型参数
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), './model/checkpoints/net_' + str(epoch+1) + '.pkl')

def test(net,device,threshold,path):
    net.load_state_dict(torch.load(path, map_location=torch.device(device)))

      
    name = 'data/detection/test/1.jpg'
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X,net,device)    
    display(img, output.cpu(), threshold=threshold)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TinySSD(num_classes=1)
    net = net.to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--threshold',type=float,default=0.5,help='threshold')
    parser.add_argument('--path',type=str,help='path of the checkpoint')
    args = parser.parse_args()
    if args.mode == 'train':
        train(net,device,args.batch_size,args.epochs,args.path)
    elif args.mode == 'test':
        test(net,device,args.threshold,args.path)