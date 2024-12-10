import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union, Tuple
from sklearn.metrics import confusion_matrix

"""
修改日志：
    7.31
    加入双向扫描机制
    加上融合注意力机制
    整改意见：在对特征进行融合的时候采用了适当的激活函数
    发现问题，对第五类样本十分不敏感，测试卷积神经网络表现，进行适当修改，对第三类样本容易分类成第二类样本
    8.2 
    数据增强对于模型很有效，需要制定特殊方案筛选样本
    数据增强对于模型分类准确率提升有比较大的帮助
    下一步考虑给矩阵A加上更多的学习属性，毕竟靠一个矩阵A记住所有历史信息比较困难。
"""


@dataclass
class ModelArgs:
    # basic parameter
    classes: int = 5
    batch_size: int = 25
    image_size: Tuple[int, int, int] = (150, 32, 32)
    bias: bool = True

    # mamba parameter
    spectral_dimension: int = image_size[1]
    spectral_piece_size: Tuple[int, int] = (image_size[0], image_size[2])
    spectral_patch_size: Tuple[int, int] = (3, 32)
    dimension_expand: int = 2
    hidden_dimension: int = 4
    mamba_expand: int = 2
    mamba_layer: int = 2
    state_space: int = 24
    seq_conv: int = 5
    dt_rank: Union[int, str] = 'auto'
    mamba_conv_bias: bool = True

    # ResNet34.py parameter
    image_channel: int = image_size[0]
    cnn_expand: int = 2
    cnn_layer: int = 2

    def __post_init__(self):
        self.hidden_dimension = self.dimension_expand * self.dimension_expand
        self.spectral_dimension_expand = int(self.hidden_dimension * self.spectral_dimension)
        self.d_inner = int(self.mamba_expand * self.spectral_dimension_expand)
        self.seq_len = int(self.spectral_piece_size[0] // self.spectral_patch_size[0])

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.spectral_dimension_expand / 16)


class ResNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super(ResNet, self).__init__()

        self.channel = args.image_channel
        self.factor = args.cnn_expand
        self.layer = args.cnn_layer

        self.b1 = nn.Sequential(nn.Conv2d(in_channels=self.channel,
                                          out_channels=self.channel * self.factor,
                                          kernel_size=4,
                                          stride=2,
                                          padding=2,
                                          groups=150),
                                nn.BatchNorm2d(self.channel * self.factor),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(self.channel * self.factor,
                                                   self.channel * self.factor,
                                                   self.layer,
                                                   first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(self.channel * self.factor,
                                                   self.channel * self.factor * self.factor,
                                                   self.layer))
        self.out_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        y = self.b1(x)
        y = self.b2(y)
        y = self.b3(y)
        output = self.out_layer(y).flatten(2)
        return output


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides, groups=150)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, groups=150)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, groups=150)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        output = x + y
        return F.relu(output)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super(Mamba, self).__init__()

        #  通用信息
        self.args = args
        self.norm_f = RMSNorm(args.spectral_dimension_expand)

        # ResNet信息
        self.resnet = ResNet(args)
        self.res_len = args.image_channel * args.cnn_expand * args.cnn_expand * args.cnn_expand
        self.res_fusion = ResNet_Feature_Extract(args)

        # Mamba信息
        self.embedding = Embedding(args)
        self.layers_1 = nn.ModuleList([ResidualBlock(args) for _ in range(args.mamba_layer)])
        self.layers_2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.mamba_layer)])

        self.feature_extract_1 = nn.Conv1d(args.seq_len, args.seq_len,
                                           kernel_size=args.hidden_dimension,
                                           stride=args.hidden_dimension,
                                           groups=args.seq_len)
        self.mam_fusion_1 = Mamba_Feature_Extract(args)

        self.feature_extract_2 = nn.Conv1d(args.seq_len, args.seq_len,
                                           kernel_size=args.hidden_dimension,
                                           stride=args.hidden_dimension,
                                           groups=args.seq_len)
        self.mam_fusion_2 = Mamba_Feature_Extract(args)

        self.out_linear = nn.Linear(int(6 * args.seq_len * args.spectral_dimension / 8 + self.res_len), args.classes,
                                    bias=args.bias)

    def forward(self, input_image):
        """
        Args:
            input_image (long tensor): shape (b, in_c, c, w, h)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, classes)

        """
        y_mamba_1, y_mamba_2 = self.embedding(input_image)
        y_res = self.resnet(input_image)
        y_res = self.res_fusion(y_res)

        for layer in self.layers_1:
            y_mamba_1 = layer(y_mamba_1)

        for layer in self.layers_2:
            y_mamba_2 = layer(y_mamba_2)

        y_mamba_1 = y_mamba_1[:, 1:, :]
        y_mamba_1 = self.norm_f(y_mamba_1)
        y_mamba_1 = self.feature_extract_1(y_mamba_1)
        y_mamba_1 = self.mam_fusion_1(y_mamba_1)

        y_mamba_2 = y_mamba_2[:, 1:, :]
        y_mamba_2 = self.norm_f(y_mamba_2)
        y_mamba_2 = self.feature_extract_2(y_mamba_2)
        y_mamba_2 = self.mam_fusion_2(y_mamba_2)

        y = torch.cat([y_res, y_mamba_1, y_mamba_2], dim=1)

        output = F.softmax(self.out_linear(y), dim=1)

        return output


class ResNet_Feature_Extract(nn.Module):
    def __init__(self, args: ModelArgs):
        super(ResNet_Feature_Extract, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=args.cnn_expand, kernel_size=3, stride=3)
        self.conv_2 = nn.Conv1d(in_channels=1, out_channels=args.cnn_expand, kernel_size=6, stride=6)
        self.conv_3 = nn.Conv1d(in_channels=1, out_channels=args.cnn_expand, kernel_size=12, stride=12)

    def forward(self, input_feature):

        x = rearrange(input_feature, 'b c l -> b l c')

        y_1 = self.conv_1(x)
        y_2 = self.conv_2(x)
        y_3 = self.conv_3(x)

        y_2 = torch.cat([y_2, y_2], dim=2)
        y_3 = torch.cat([y_3, y_3, y_3, y_3], dim=2)

        # output = F.silu(torch.cat([y_1, y_2, y_3], dim=1).flatten(1))
        output = torch.cat([y_1, y_2, y_3], dim=1).flatten(1)

        return output


class Mamba_Feature_Extract(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Mamba_Feature_Extract, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=args.seq_len, out_channels=args.seq_len, kernel_size=8, stride=8, groups=args.seq_len)
        self.conv_2 = nn.Conv1d(in_channels=args.seq_len, out_channels=args.seq_len, kernel_size=16, stride=16, groups=args.seq_len)
        self.conv_3 = nn.Conv1d(in_channels=args.seq_len, out_channels=args.seq_len, kernel_size=32, stride=32, groups=args.seq_len)

    def forward(self, input_feature):

        y_1 = self.conv_1(input_feature)
        y_2 = self.conv_2(input_feature)
        y_3 = self.conv_3(input_feature)

        y_2 = torch.cat([y_2, y_2], dim=2)
        y_3 = torch.cat([y_3, y_3, y_3, y_3], dim=2)

        # output = F.silu(torch.cat([y_1, y_2, y_3], dim=1).flatten(1))
        output = torch.cat([y_1, y_2, y_3], dim=1).flatten(1)

        return output


class Embedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Embedding, self).__init__()
        self.segmentation = nn.Sequential(nn.Conv2d(in_channels=args.spectral_dimension,
                                                    out_channels=args.spectral_dimension_expand,
                                                    kernel_size=(args.spectral_patch_size[0], args.spectral_patch_size[1]),
                                                    stride=(args.spectral_patch_size[0], args.spectral_patch_size[1]),
                                                    groups=args.spectral_dimension))
        self.embedding_1 = nn.Parameter(torch.zeros(1, args.spectral_dimension_expand, 1))
        self.embedding_2 = nn.Parameter(torch.zeros(1, args.spectral_dimension_expand, 1))

    def forward(self, x):
        """
        处理思路：将第一个维度切片
        Args:
            x : shape (b, c, h, w)

        Returns:
            sequence : shape (b, c, l)

        """
        x = rearrange(x, 'b c h w -> b h c w')

        y = self.segmentation(x).flatten(2)

        y_1 = torch.cat([self.embedding_1.expand(y.size(0), -1, -1), y], dim=2)

        y_2 = torch.cat([y, self.embedding_2.expand(y.size(0), -1, -1)], dim=2)
        y_2 = torch.flip(y_2, dims=[2])

        y_1 = rearrange(y_1, 'b c l -> b l c')
        y_2 = rearrange(y_2, 'b c l -> b l c')

        return y_1, y_2


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.spectral_dimension_expand)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.spectral_dimension_expand, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.mamba_conv_bias,
            kernel_size=args.seq_conv,
            groups=args.d_inner,
            padding=int((args.seq_conv - 1) / 2)
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.state_space * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.state_space + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.spectral_dimension_expand, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        """

        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


model_args = ModelArgs()
net = Mamba(model_args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_confusion_matrix(net, test_data):
    all_preds = []
    all_labels = []
    for i, (X, y) in enumerate(test_data):
        X, y = X.to(device).float(), y.to(device).long()
        y_hat = net(X)
        predict_y = torch.argmax(y_hat, dim=1)
        all_preds.extend(predict_y.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return confusion_matrix(all_labels, all_preds)


def calculate_class_accuracy(confusion_matrix):
    class_accuracy = []
    for i in range(confusion_matrix.shape[0]):
        class_total = sum(confusion_matrix[i, :])
        if class_total == 0:
            class_accuracy.append(0)  # 分母为0时保持正确率为0
        else:
            class_accuracy.append(confusion_matrix[i, i] / class_total)
    return class_accuracy


# 保存地址
save_path = './ResNet-Mamba-v2.1.pkl'
# 是否采用新的网络
new_model = False
save_model = False
BATCH_SIZE = 25
learn_rate = 1e-4
lr_decrease = 0.5
EPOCH = 40
train_step = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 读取三维立方数据
file_path = "D:/hu-dataset/dataset.pt"
dataset = torch.load(file_path)


train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size],
                                                        generator=torch.Generator().manual_seed(0))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 假设我们已经创建了一个Mamba模型实例
total_params = count_parameters(net)
print("Total parameters:", total_params)

net.to(device)  # 实例化网络模型并送入GPU

if not new_model:
    net.load_state_dict(torch.load(save_path))  # 使用上次训练权重接着训练

optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)  # 定义优化器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decrease)  # 定义学习率下降
loss_function = nn.CrossEntropyLoss()  # 多分类问题使用交叉熵损失函数


# 测试函数，传入模型和数据读取接口
def evalute(model, loader):
    # correct为总正确数量，total为总测试数量
    correct = 0
    total = len(loader.dataset)
    # 取测试数据
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).long()
        # validation和test过程不需要反向传播
        model.eval()
        with torch.no_grad():
            out = model(x)  # 计算测试数据的输出logits
            # 计算出out在第一维度上最大值对应编号，得模型的预测值
            prediction = out.argmax(dim=1)
        # 预测正确的数量correct
        correct += torch.eq(prediction, y).float().sum().item()
    # 最终返回正确率
    return correct / total


best_acc, best_epoch = 0.0, 0  # 最好准确度，出现的轮数
global_step = 0  # 全局的step步数，用于画图
for epoch in range(EPOCH):
    running_loss = 0.0  # 一次epoch的总损失
    net.train()  # 开始训练
    for step, (images, labels) in enumerate(train_dataloader, start=0):
        images, labels = images.to(device).float(), labels.to(device).long()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # 将一个epoch的损失累加
        # 打印输出当前训练的进度
        rate = (step + 1) / len(train_dataloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\repoch: {} train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch + 1, int(rate * 100), a, b, running_loss), end="")
        # 每次记录之后将横轴x的值加一
        global_step += 1

    # 在每一个epoch结束，做一次test
    if epoch % 1 == 0:
        # 使用上面定义的evalute函数，测试正确率，传入测试模型net，测试数据集test_dataloader
        test_acc = evalute(net, test_dataloader)
        # val_acc = evalute(net, val_dataloader)
        print("  epoch{} test acc:{}".format(epoch + 1, test_acc))
        # 根据目前epoch计算所得的acc，看看是否需要保存当前状态（即当前的各项参数值）以及迭代周期epoch作为最好情况
        if (test_acc > 0.7 and test_acc > best_acc) or ((epoch + 1) % train_step == 0 and test_acc < 0.7):
            best_acc = test_acc
            # 显示当前模型状态
            val_acc = evalute(net, val_dataloader)
            if test_acc > val_acc:
                val_acc = test_acc
                conf_mat = evaluate_confusion_matrix(net, test_dataloader)
            else:
                conf_mat = evaluate_confusion_matrix(net, val_dataloader)
            print("val_acc{}".format(val_acc))
            class_accuracy = calculate_class_accuracy(conf_mat)
            print("Confusion Matrix:")
            print(conf_mat)
            print('每一类的分类正确率：', class_accuracy)

            if save_model:
                torch.save(net.state_dict(), save_path)
                print('Model save!')

    if epoch % train_step == 0 and epoch != 0:
        scheduler.step()


print("Finish !")
