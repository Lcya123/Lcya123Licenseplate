import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 因为 LeNet5 想要的输入为32*32   ---> 但 mnist数据集大小为 28*28
# conv -> pool ->  conv -> pool -> fc -> fc -> output


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 输入28*28  k_s=5*5, s=1 不添加padding的话 输出为24  需要额外padding=2  即两边各补2个0
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=2)
        self.pool1 =  nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=5*5*16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=65)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 转换形状
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # 交叉熵会自动 softmax
        return x

    # 真正的参数 只有input_img
    def predict(self, input_img):
        with open(r"class_indices.json", "r") as fd_js:  # 读class_indices.json,从中获取标签信息class_indict和标签数量num_classes
            class_indict = json.load(fd_js)  # json转为python对象
        # 是否为 实例  type torch.Tensor
        if not isinstance(input_img, torch.Tensor):  # 如果不为tensor对象 则转变为tensor对象
            input_img = torch.tensor(input_img, dtype=torch.float32).to(device)  # 转换为 tensor对象

        # H, W ,C  现在图像为二值化图像 28, 28
        if len(input_img.shape) != 3:
            # unsqueeze  ---> 表示增加
            input_img = input_img.unsqueeze(0)  # 如果维度不为 3则增加一个 bs维度
            # 1, 28, 28

        # 返回最大索引值, 参2: 表在某个维度(在列维度找最大)
        # 10*10 ---> 1*10 [[5, 7, 8, 9],
        #                   [1, 8, 6, 3]
        # 2 ---> argmax [0, 1, 0, 0]
        #
        output = torch.argmax(self.forward(input_img), 1)  # 返回最大索引值 (前向传播)

        return class_indict[str(output.item())]  # 转换为Python对象


# 实例化的一个类
model = LeNet5().to(device)  # 有了一个模型
model.load_state_dict(torch.load("./save_model/model_best.pth", weights_only=True))
