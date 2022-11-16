from .bbn_dataset import BBN_Dataset
from .bbn_model import BBN_ResNet50
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from pycm import ConfusionMatrix
import torch
# 初始化模型
model=BBN_ResNet50()


# 构建数据集
batch=64
txt_path="./dataset.txt"
train_set = BBN_Dataset(txt_path=txt_path,mode="train",size=[224,224])
val_set = BBN_Dataset(txt_path=txt_path,mode="val",size=[224,224])

# 构建数据集加载器
train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=batch,
    num_workers=4,
    shuffle=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=val_set,
    batch_size=batch,
    num_workers=4,
)

# 开始训练
optimizer=None
lr_scheduler=None
criterion=nn.CrossEntropyLoss()
Epochs=100

for epoch in range(Epochs):
    optimizer.zero_grad()

    for batch_idx, (
            imgs,
            labels,
            imgs_path,
            imgs2,
            labels2,
            imgs_path2,
    ) in enumerate(tqdm(train_dataloader)):
        model.train()
        # 正常采样分布
        imgs, labels = imgs.cuda(), labels.cuda()
        # 逆向采样分布
        imgs2, labels2 = imgs2.cuda(), labels2.cuda()

        l = 1 - ((epoch - 1) / Epochs) ** 2  # parabolic decay抛物线
        params = {"imgs1": imgs, "imgs2": imgs2, "l": l}
        output = model(params)
        loss = l * criterion(output, labels) + (1 - l) * criterion(
            output, labels2
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    lr_scheduler.step()
    # 评估模型
    preds_list, labels_list = [], []
    for batch_idx, (imgs, labels, imgs_path) in enumerate(tqdm(val_dataloader)):
        imgs, labels = imgs.cuda(), labels.cuda()
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)

        preds_list.append(preds)
        labels_list.append(labels)
    preds_list = torch.cat(preds_list, dim=0).cpu().numpy()
    labels_list = torch.cat(labels_list, dim=0).cpu().numpy()
    acc=ConfusionMatrix(labels_list, preds_list).Overall_ACC
    print("val acc:",acc)