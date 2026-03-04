import os
import json
import torch
import random
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from LeNet5_34 import LeNet5, device

dataset_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # 对RGB图像进行标准化处理
])


def setup_seed(seed: int):
    np.random.seed(seed)  # 设置numpy随机数种子
    random.seed(seed)  # 设置Python随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python的哈希种子
    torch.manual_seed(seed)  # 设置torch的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置torch cuda 随机数种子
        torch.cuda.manual_seed_all(seed)  #
        torch.backends.cudnn.benchmark = False  # 关闭cudnn加速
        torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性算法


# 固定随机数种子  选择设备
setup_seed(0)
print("Current Device is [%s]" % str(device))


class CharData(Dataset):
    def __init__(self, dir_path, transform=None):
        self.transform = transform
        self.image_and_label = []  # 存储所有图片路径和标签
        self.class_char = []
        # 支持的文件后缀类型
        supported = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

        # 遍历所有子目录
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            chars_class = []
            if os.path.isdir(subdir_path):
                for cla in os.listdir(subdir_path):
                    if os.path.isdir(os.path.join(subdir_path, cla)):
                        self.class_char.append(cla)
                        chars_class.append(cla)

                chars_class.sort()  # 排序操作

                # 生成类别名称以及对应的数字索引
                class_indices = dict((k, v) for v, k in enumerate(self.class_char))
                json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
                with open('class_indices.json', 'w') as json_file:
                    json_file.write(json_str)
                for per_class in chars_class:
                    class_path = os.path.join(subdir_path, per_class)
                    for filename in os.listdir(class_path):
                        if os.path.splitext(filename)[-1] in supported:
                            tmp_path = os.path.join(class_path, filename)  # 获得图片路径
                            tmp_label = class_indices[per_class]  # 获得对应的类别
                            self.image_and_label.append((tmp_path, tmp_label))

    def __len__(self):
        # 返回数据集长度
        return len(self.image_and_label)

    def __getitem__(self, idx):
        img_path, label = self.image_and_label[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label


# 实例化 数据集
full_dataset = CharData(dir_path="./datatset", transform=dataset_transform)

# 划分数据集
train_ratio = 0.8   # 设置训练集的比例
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 创建对应的dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义损失函数 和 优化器
model = LeNet5().to(device)
loss_func = nn.CrossEntropyLoss()       # sum(p(x)logq(x))  # 通过概率分布q(x)表示概率分布p(x)的困难程度
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # w = w - gt

train_losses = []
test_losses = []
test_accuracies = []
# 开始迭代
epoch_nums = 100
for epoch in range(epoch_nums):
    # 设置训练模型
    model.train()
    most_acc = 0        # 最优准确率
    train_total_loss = 0

    # 迭代数据集
    for batch_idx, train_data in enumerate(train_loader):
        imgs = train_data[0].to(device)
        labels = train_data[1].to(device)

        # 前戏传播
        outputs = model(imgs)
        loss = loss_func(outputs, labels)   # 计算损失
        train_total_loss += loss

        # 反向传播
        optimizer.zero_grad()       # 梯度清零
        loss.backward()
        optimizer.step()
        # print(f"Epoch: {epoch + 1}/{epoch_nums}, Iter: {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}")

    train_avg_loss = train_total_loss / len(train_loader)
    train_losses.append(train_avg_loss.item())
    print(f"Epoch: {epoch+1}/{epoch_nums}, Loss: {train_avg_loss:.4f}")

    # 模型评估测试
    model.eval()  # 设置评估模式
    total_num = 0       # 总的数量
    correct_num = 0      # 正确数量
    test_total_loss = 0     # 测试的损失
    with torch.no_grad():  # 禁用自动梯度计算  --> 1.减少内存的占用  2.加速计算 3.防止梯度累加
        for test_idx, test_data in enumerate(test_loader):
            test_imgs = test_data[0].to(device)
            test_labels = test_data[1].to(device)
            total_num += test_labels.size(0)

            test_outputs = model(test_imgs)
            # print(test_outputs.shape)       # 看一下输出
            test_loss = loss_func(test_outputs, test_labels)
            test_total_loss += test_loss
            pred_dix = torch.argmax(test_outputs, 1)        # 取列维度上 最大的那个

            # 统计正确率
            correct_num += (pred_dix == test_labels).sum().item()     # sum统计数量 item转为python数据类型
        # 每个epoch统计一次平均损失
        test_avg_loss = test_total_loss / len(test_loader)
        acc = correct_num / total_num
        test_losses.append(test_avg_loss.item())
        test_accuracies.append(acc)
        print(f"Test Data: Epoch [{epoch + 1} / {epoch_nums}], Loss {test_avg_loss:.4f}, Acc {acc * 100}%")

    if acc > most_acc:
        torch.save(model.state_dict(), f"./save_model/model_best.pth")
        most_acc = acc

    # 每隔10个epoch 进行保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./save_model/model_{epoch + 1}.pth")
    if epoch ==50:
        break
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='green')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)  # 固定y轴范围便于比较
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./training_curves.png')  # 保存图像
plt.show()  # 显示图像（可选）