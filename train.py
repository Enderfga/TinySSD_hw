from data.dataloader import load_data
import torch
from model.TinySSD import *
from utils.utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = TinySSD(num_classes=1)
net = net.to(device)

batch_size = 32
train_iter = load_data(batch_size)


trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

num_epochs = 50
for epoch in range(num_epochs):
    print('epoch: ', epoch)
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
        torch.save(net.state_dict(), 'model\checkpoints/net_' + str(epoch+1) + '.pkl')