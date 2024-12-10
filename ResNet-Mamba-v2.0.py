import math
import time
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
    8.3
    对编码层的卷积操作更细致
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

        self.args = args

        self.resnet = ResNet(args)
        self.res_len = args.image_channel * args.cnn_expand * args.cnn_expand * args.cnn_expand

        self.embedding = Embedding(args)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.mamba_layer)])

        self.norm_f = RMSNorm(args.spectral_dimension_expand)

        self.feature_extract = nn.Conv1d(args.seq_len, args.seq_len,
                                         kernel_size=args.hidden_dimension,
                                         stride=args.hidden_dimension,
                                         groups=args.seq_len)

        self.res_fusion = ResNet_Feature_Extract(args)
        self.mam_fusion = Mamba_Feature_Extract(args)

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

        for layer in self.layers:
            y_mamba_1 = layer(y_mamba_1)
            y_mamba_2 = layer(y_mamba_2)

        y_mamba_1 = y_mamba_1[:, 1:, :]
        y_mamba_2 = y_mamba_2[:, 1:, :]

        y_mamba_1 = self.norm_f(y_mamba_1)
        y_mamba_2 = self.norm_f(y_mamba_2)

        y_mamba_1 = self.feature_extract(y_mamba_1)
        y_mamba_2 = self.feature_extract(y_mamba_2)

        y_res = self.res_fusion(y_res)
        y_mamba_1 = self.mam_fusion(y_mamba_1)
        y_mamba_2 = self.mam_fusion(y_mamba_2)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_args = ModelArgs()
net = Mamba(model_args)

# 假设我们已经创建了一个Mamba模型实例
total_params = count_parameters(net)
print("Total parameters:", total_params)

exit()

# 读取三维立方数据
file_path = "C:/Users/CaierSheng/PycharmProjects/pythonProject/真菌显微高光谱分类/dataset.pt"
dataset = torch.load(file_path)

train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size],
                                                        generator=torch.Generator().manual_seed(0))
train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=25, shuffle=True)
val_iter = DataLoader(val_dataset, batch_size=25, shuffle=True)

# 保存地址
save_path = './ResNet-Mamba-v2.0.pkl'
# 是否采用新的网络
new_model = True
save_model = True
learn_rate = 1e-2
lr_decrease = 0.8
num_epochs = 20
train_step = 2


def accuracy(y_hat, y):
    predict_y = torch.argmax(y_hat, dim=1)
    y_equal = torch.eq(predict_y, y)
    y_count = torch.sum(y_equal).item()
    return y_count


def evaluate_accuracy(net, test_data, device):
    equal_counter = 0
    counter = 0
    for i, (X, y) in enumerate(test_data):
        X, y = X.to(device).float(), y.to(device).long()
        y_hat = net(X)
        equal_counter += accuracy(y_hat, y)
        counter += len(X)
    return equal_counter / counter


def evaluate_confusion_matrix(net, test_data, device):
    all_preds = []
    all_labels = []
    for i, (X, y) in enumerate(test_data):
        X, y = X.to(device).float(), y.to(device).long()
        y_hat = net(X)
        predict_y = torch.argmax(y_hat, dim=1)
        all_preds.extend(predict_y.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return confusion_matrix(all_labels, all_preds)


def train_ch6(net, train_iter, test_iter, val_iter):
    if torch.cuda.is_available():
        print('train start on GPU:')
        device = torch.device('cuda')
    else:
        print('train start on CPU:')
        device = torch.device('cpu')

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化网络参数
    net.apply(init_weights)
    net.to(device)
    # 优化器和学习率
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
    # 优化器衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decrease)
    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 记录最好的训练模型
    best_acc = 0.0

    if not new_model:
        # 引入预训练好的模型
        net.load_state_dict(torch.load(save_path))
        test_acc = evaluate_accuracy(net, test_iter, device)
        best_acc = test_acc
        print(f'Loading pretrained net ...')
        print(f'Test acc {test_acc:.3f}')

    # 迭代训练
    for epoch in range(num_epochs):

        # 训练损失之和，训练准确率之和，样本数
        net.train()

        # 计时器
        train_start = time.time()

        # 统计损失值
        total_loss = 0.0

        # 分批次训练
        for i, (X, y) in enumerate(train_iter):

            # 优化器梯度归零
            optimizer.zero_grad()
            # 转移到GPU
            X, y = X.to(device).float(), y.to(device).long()
            # 计算预测值
            y_hat = net(X)
            # 计算损失值
            l = loss(y_hat, y)
            total_loss += l.item()
            # 反向传播误差
            l.backward()
            optimizer.step()

        # 计时结束
        train_end = time.time()

        train_time = train_end - train_start

        test_start = time.time()

        net.eval()

        test_acc = evaluate_accuracy(net, test_iter, device)

        test_end = time.time()

        test_time = test_end - test_start

        print(
            f'epoch {epoch}, test acc {test_acc:.3f}, '
            f'train sec {train_time:.3f} sec, '
            f'test sec {test_time:.3f} sec, '
            f'total loss {total_loss}')

        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型参数
            print("Find better model ...")
            conf_mat = evaluate_confusion_matrix(net, val_iter, device)
            val_acc = evaluate_accuracy(net, val_iter, device)
            print("Confusion Matrix:")
            print(conf_mat)
            print(f"Val acc : {val_acc:.3f} !")
            if save_model and test_acc > 0.75:
                torch.save(net.state_dict(), save_path)
                print(f'Saving best model ...')

        if epoch % train_step == 0 and epoch != 0:
            scheduler.step()


train_ch6(net, train_iter, test_iter, val_iter)
