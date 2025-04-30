import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

# 将数据转化为Variable类型
def to_var(x, requires_grad=True):
    return Variable(x, requires_grad=requires_grad)

# 归一化函数，将预测值映射到0和1之间
def norY(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y

# 定义一个简单的sigmoid网络
def sigmoid_net(X, W):
    return torch.sigmoid(torch.matmul(X, W))

# 定义元模块类，包含许多用于参数访问和更新的辅助函数
class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

# 定义一个MetaLinear类，用于创建元学习中的线性模型
class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        # self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

class MetaLinear_multi(MetaModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        ignore = nn.Linear(in_features, out_features)

        self.register_buffer('weight', ignore.weight)

    def forward(self, x):
        return F.linear(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

# 定义MetaConv2d类，用于创建元学习中的卷积模型
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 权重初始化函数，用于初始化模型参数
def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

# 随机梯度下降优化算法
def sgd(params, lr, batch_size):  # @save
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 更新器函数，返回一次迭代更新
def updater(batch_size, W, lr):
    return sgd([W], lr, batch_size)

# 定义一个简单的全连接网络
class Net1(MetaModule):
    def __init__(self):
        super(Net1, self).__init__()
        # self.in_planes = 16
        self.linear = MetaLinear(1000, 1, bias=False)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

class Net2(MetaModule):
    def __init__(self, features):
        super(Net2, self).__init__()
        # self.in_planes = 16
        self.linear = MetaLinear(features, 1, bias=False)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

class Net2_multi(MetaModule):
    def __init__(self, features, num_classes):
        super(Net2_multi, self).__init__()
        self.linear = MetaLinear(features, num_classes, bias=False)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        return F.softmax(out, dim=1)  # 使用 softmax 激活函数进行多分类

# 计算加权损失函数
def calculate_weight_loss2(model, vnet, x, y):
    inputs, targets = x, y
    outputs = model(inputs).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, targets)  #
    cost_v = torch.reshape(cost, (len(cost), 1))

    with torch.no_grad():
        w_new = vnet(cost_v)
        w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

    return (cost_v * w_new), w_new

def calculate_weight_loss2_mul(model, vnet, x, y):
    inputs, targets = x, y
    outputs = model(inputs).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = F.cross_entropy(outputs, targets.long(), reduce=False)
    cost_v = torch.reshape(cost, (len(cost), 1))

    with torch.no_grad():
        w_new = vnet(cost_v)
        w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

# 定义VNet类，包含一个简单的全连接神经网络
class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)
    
class VNet1(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

