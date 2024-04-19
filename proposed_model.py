import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class JMNet(nn.Module):
    def __init__(self, shape):
        super(JMNet, self).__init__()
        self.first_freq = 1
        self.last_freq = 40
        self.filter_size = 150
        self.reduction_ratio = 4
        self.freq_len = [[0, 8], [8, 16], [16, 24], [24, 32], [32, 40]]
        self.filter_n = (self.freq_len[0][1] - self.freq_len[0][0])
        self.num_branch = len(self.freq_len)
        self.num_ch = shape[2]
        self.C1 = 32
        self.C2 = 64
        self.t1 = 15
        self.t2 = 15
        self.cnn_list = nn.ModuleList()
        self.wkn_list = nn.ModuleList()
        self.freq_selection = torch.tensor(1)
        self.omega = torch.tensor(1)

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(self.C1*self.num_branch, self.C2, kernel_size=(1, self.t2), stride=(1, 2)),
            nn.BatchNorm2d(self.C2),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        for i in range(self.num_branch):
            self.wkn_list.append(CWConv(self.freq_len[i][0]+1, self.freq_len[i][1], self.filter_n, self.filter_size, 1))
            self.cnn_list.append(ConvBlock(self.num_ch, self.C1, self.C2, self.filter_n))
        self.linear = nn.Sequential(
            nn.Linear(in_features=448 * self.num_branch + 448, out_features=4)
        )

    def get_wkn_info(self):
        print("\n==== [WKN INFO] ====\n")
        freq = []
        omega = []
        print("=====> Frequency\n")
        for i in range(self.num_branch):
            print(f"branch{i}:     ", end="")
            freq.append(self.wkn_list[i].get_freq_info())
            print(freq[-1])
        print("\n=====> Omega\n")
        for i in range(self.num_branch):
            print(f"branch{i}:     ", end="")
            omega.append(self.wkn_list[i].get_omega_info())
            print(omega[-1])
        print("\n")
        self.freq_selection = torch.stack(freq, dim=0)
        self.omega = torch.stack(omega, dim=0)


    def forward(self, x):
        batch_size, _, ch, tp = x.shape
        out = torch.reshape(x, (batch_size * ch, 1, -1))
        low_feat = []
        high_feat = []
        wt_img = []
        for i in range(0, self.num_branch):
            tmp = self.wkn_list[i](out)
            tmp = tmp.view(batch_size, ch, self.filter_n, -1)  # 36,22,filter_n,time_n
            wt_img.append(tmp)
            tmp, o, score = self.cnn_list[i](tmp)
            low_feat.append(o[0])
            high_feat.append(tmp)

        # low level global feature embedding
        low_out = torch.cat(low_feat, dim=1)
        low_out = self.Conv_1(low_out)
        low_out, _ = self.se(low_out)

        # high level global feature embedding
        high_out = torch.cat(high_feat, dim=1)

        out = torch.cat([low_out, high_out], dim=1)
        out = out.view(batch_size, -1)
        out = self.linear(out)

        return out, wt_img


class ConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n_freq):
        super(ConvBlock, self).__init__()
        self.in_dim = in_ch
        self.out_dim = out_ch
        self.mid_dim = mid_ch
        self.n_freq = 15

        self.Conv_0 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(n_freq, 15), stride=(1, 2)),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=(1, 15), stride=(1, 2)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(0.25)
        )

        self.chan_atten1 = SEBlock(self.mid_dim)
        self.chan_atten2 = SEBlock(self.out_dim)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.Conv_0(x)
        out, s1 = self.chan_atten1(out)
        low = out
        out = self.Conv_1(out)
        out, s2 = self.chan_atten2(out)
        return out, [low, None], s1


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=(1, 1), padding=(0, 0), pooling=(1, 1), attention=True):
        super(Conv, self).__init__()
        self.in_dim = in_ch
        self.out_dim = out_ch
        self.n_freq = 15
        self.att = attention

        self.Conv_0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=(1, 2)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        self.attention = SEBlock(self.out_ch)

    def forward(self, x):
        out = self.Conv_0(x)
        return self.attention(out) if self.att else out


class CWConv(nn.Module):
    def __init__(self, first_freq, last_freq, filter_n, kernel_size, in_channels=1):
        super(CWConv, self).__init__()
        if in_channels != 1:
            msg = "only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.kernel_size = kernel_size
        self.filter_n = filter_n
        self.omega = 5.15
        self.a_ = nn.parameter.Parameter(torch.tensor([float(x/100) for x in range(first_freq, last_freq+1)]).view(-1, 1))
        self.b_ = torch.tensor(self.omega)

    def get_freq_info(self):
        return torch.clamp(torch.tensor(list(map(lambda x: float(x * 100.0), self.a_))), min=1e-7)

    def get_omega_info(self):
        return torch.tensor(self.b_)

    def forward(self, waveforms):
        device = waveforms.device
        M = self.kernel_size
        x = (torch.arange(0, M) - (M - 1.0) / 2).to(device)
        s = (2.5 * self.b_) / (torch.clamp(self.a_, min=1e-7) * 2 * math.pi)
        x = x / s
        wavelet = (torch.cos(self.b_ * x) * torch.exp(-0.5 * x ** 2) * math.pi ** (-0.25))
        output = (torch.sqrt(1 / s) * wavelet)
        Morlet_filter = output
        self.filters = (Morlet_filter).view(self.filter_n, 1, self.kernel_size)
        out = F.conv1d(waveforms, self.filters, stride=1, padding=(self.kernel_size-1)//2, dilation=1, bias=None, groups=1)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction_ratio=4, dilation=1):
        super(SEBlock, self).__init__()
        self.hid_channel = int(in_channel // reduction_ratio)
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)
        Mc = self.sigmoid(avgOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        score = Mc
        # print(score[0])
        Mf1 = Mc * x
        out = Mf1 + x
        out = self.relu(out)
        return out, score