from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter, ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch



class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


################################################  Attention  ################################################ 


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.global_avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.global_avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIR, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIRSE, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    SEModule(out_channels, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x) if self.identity != 1 else x
        res = self.res_layer(x)
        return res + shortcut


class BasicResBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(2, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    ReLU(inplace=True),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class MaskModule(Module):
    def __init__(self, down_sample_times, out_channels, r, net_mode='ir'):
        super(MaskModule, self).__init__()
        assert net_mode in ('ir', 'basic', 'irse')
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}

        self.max_pool_layers = ModuleList()
        for i in range(down_sample_times):
            self.max_pool_layers.append(MaxPool2d(2, 2))

        self.prev_res_layers = ModuleList()
        for i in range(down_sample_times):
            tmp_prev_res_block_layers = []
            for j in range(r):
                tmp_prev_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
            self.prev_res_layers.append(Sequential(*tmp_prev_res_block_layers))

        self.mid_res_layers = None
        self.post_res_layers = None
        if down_sample_times > 1:
            self.mid_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                self.mid_res_layers.append(func[net_mode](out_channels, out_channels, 1))

            self.post_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                tmp_post_res_block_layers = []
                for j in range(r):
                    tmp_post_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
                self.post_res_layers.append(Sequential(*tmp_post_res_block_layers))

        self.r = r
        self.out_channels = out_channels
        self.down_sample_times = down_sample_times

    def mask_branch(self, x, cur_layers, down_sample_times):
        h = x.shape[2]
        w = x.shape[3]

        cur_layers.append(self.max_pool_layers[self.down_sample_times - down_sample_times](x))

        cur_layers.append(self.prev_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
        # down_sample_times -= 1
        if down_sample_times - 1 <= 0:

            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear'))
            return cur_layers[-1]
        else:
            cur_layers.append(self.mid_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))

            shortcut_layer = cur_layers[-1]
            v = self.mask_branch(cur_layers[-1], cur_layers, down_sample_times - 1)
            cur_layers.append(shortcut_layer + v)

            cur_layers.append(self.post_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear'))
            return cur_layers[-1]

    def forward(self, x):
        cur_layers = []
        return self.mask_branch(x, cur_layers, self.down_sample_times)


class AttentionModule(Module):
    def __init__(self, in_channels, out_channels, input_spatial_dim, p=1, t=2, r=1, net_mode='ir'):
        super(AttentionModule, self).__init__()
        self.func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}

        # start branch
        self.start_branch = ModuleList()
        self.start_branch.append(self.func[net_mode](in_channels, out_channels, 1))
        for i in range(p - 1):
            self.start_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # trunk branch
        self.trunk_branch = ModuleList()
        for i in range(t):
            self.trunk_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # mask branch
        # 1st, determine how many down-sample operations should be executed.
        num_down_sample_times = 0
        resolution = input_spatial_dim
        while resolution > 4 and resolution not in (8, 7, 6, 5):
            num_down_sample_times += 1
            resolution = (resolution - 2) // 2 + 1
        self.num_down_sample_times = min(num_down_sample_times, 100)
        self.mask_branch = MaskModule(num_down_sample_times, out_channels, r, net_mode)

        self.mask_helper = Sequential(Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      Sigmoid())
        # output branch
        self.out_branch = ModuleList()
        for i in range(p):
            self.out_branch.append(self.func[net_mode](out_channels, out_channels, 1))
        self.p = p
        self.t = t
        self.r = r

    def forward(self, x):
        for i in range(self.p):
            x = self.start_branch[i](x)
        y = x
        for i in range(self.t):
            x = self.trunk_branch[i](x)

        trunk = x
        mask = self.mask_branch(y)
        mask = self.mask_helper(mask)
        out = trunk * (mask + 1)
        for i in range(self.p):
            out = self.out_branch[i](out)
        return out


class AttentionNet(Module):
    def __init__(self, in_channels=3, p=1, t=2, r=1, net_mode='basic', attention_stages=(1, 1, 1),dim = 512):
        super(AttentionNet, self).__init__()
        final_res_block = 3
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}
        self.input_layer = Sequential(Conv2d(in_channels, 64, 3, 1, 1),
                                      BatchNorm2d(64),
                                      ReLU(inplace=True),
                                      func[net_mode](64, 64, 2))
        input_spatial_dim = (144 - 1) // 2 + 1
        modules = []

        # stage 1
        for i in range(attention_stages[0]):
            modules.append(AttentionModule(64, 64, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](64, 128, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 2
        for i in range(attention_stages[1]):
            modules.append(AttentionModule(128, 128, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](128, 256, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 3
        for i in range(attention_stages[2]):
            modules.append(AttentionModule(256, 256, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](256, 512, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        for i in range(final_res_block):
            modules.append(func[net_mode](512, 512, 1))

        self.body = Sequential(*modules)
        self.output_layer = Sequential(Flatten(),
                                       Linear(512 * input_spatial_dim * input_spatial_dim, dim, False),
                                       BatchNorm1d(512))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return F.normalize(x)



################################################  MobileFaceNet  ################################################ 
    

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, dim):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(9, 9), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, dim, bias=False)
        self.bn = BatchNorm1d(dim)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return F.normalize(out)
