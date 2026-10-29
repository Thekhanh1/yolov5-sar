import torch
from torch import nn


class LConv(nn.Module):
    def __init__(self, inputCh, outputCh, kernelSize=3, ratio=2, stride=1):
        super().__init__()
        self.m = outputCh // ratio

        self.expensive_conv = nn.Conv2d(
            inputCh, self.m, kernel_size=kernelSize, padding=kernelSize // 2, bias=False, stride=stride
        )

        self.cheap_conv = nn.Conv2d(
            self.m, self.m * (ratio - 1), kernel_size=3, padding=1, groups=self.m, bias=False, stride=1
        )

    def forward(self, x):
        y_prime = self.expensive_conv(x)
        cheap_features = self.cheap_conv(y_prime)
        out = torch.cat([y_prime, cheap_features], dim=1)
        return out


class LBL(nn.Module):
    def __init__(self, inputCh, outputCh, kernelSize, stride=1):
        super().__init__()
        self.conv = LConv(inputCh, outputCh, kernelSize, stride=stride)
        self.BNorm = nn.BatchNorm2d(outputCh)
        self.activate = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.BNorm(x)
        x = self.activate(x)
        return x


class lBottleneck(nn.Module):
    def __init__(self, inputCh):
        super().__init__()
        self.lbl1 = LBL(inputCh, inputCh, 1)
        self.lbl2 = LBL(inputCh, inputCh, 3)

    def forward(self, x):
        x = self.lbl1(x)
        residual = x
        x = self.lbl2(x)
        x = x + residual
        return x


class LCSP(nn.Module):
    def __init__(self, inputCh, outputCh, n=1):
        super().__init__()
        self.branch1 = LBL(inputCh, inputCh // 2, 1)
        self.branch2 = LBL(inputCh, inputCh // 2, 1)
        self.bottleneck = nn.Sequential(*[lBottleneck(inputCh // 2) for _ in range(n)])
        self.final = LBL(inputCh, outputCh, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x2 = self.bottleneck(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.final(x)
        return x


# print(LCSP(128, 128, n=3)(torch.randn(1, 128, 40, 40)).shape)


class se(nn.Module):
    def __init__(self, inputCh, r=16):
        super().__init__()
        self.glPool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=inputCh, out_channels=inputCh // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inputCh // r, out_channels=inputCh, kernel_size=1, bias=False),
        )
        self.Hsig = nn.Hardsigmoid()

    def forward(self, x):
        y = self.glPool(x)
        y = self.bottleNeck(y)
        y = self.Hsig(y)

        return x * y


class cbamMlp(nn.Module):
    def __init__(self, inputCh, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=inputCh, out_channels=inputCh // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inputCh // r, out_channels=inputCh, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.mlp(x)


class cbamCa(nn.Module):
    def __init__(self, inputCh, r=16):
        super().__init__()
        self.maxPath = nn.AdaptiveMaxPool2d((1, 1))
        self.avgPath = nn.AdaptiveAvgPool2d((1, 1))

        self.MLP = cbamMlp(inputCh=inputCh, r=r)

        self.Hsig = nn.Hardsigmoid()

    def forward(self, x):
        x1 = self.MLP(self.maxPath(x))
        x2 = self.MLP(self.avgPath(x))

        x = self.Hsig(x1 + x2)

        return x


class cbamSa(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv7 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k // 2, bias=False)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        a = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)

        a = self.conv7(a)
        a = self.hardsigmoid(a)

        return x * a


class cbam(nn.Module):
    def __init__(self, inputCh, k=7, r=16):
        super().__init__()
        self.mc = cbamCa(inputCh=inputCh, r=r)
        self.ms = cbamSa(k)

    def forward(self, x):
        x1 = self.mc(x) * x
        return self.ms(x1) * x1


# print(cbam(128, 7, 16)(torch.randn(1, 128, 40, 40)).shape)


class coordinateAttention(nn.Module):
    def __init__(self, inputCh, r=32):
        super().__init__()
        self.maxW = nn.AdaptiveAvgPool2d((1, None))
        self.maxH = nn.AdaptiveAvgPool2d((None, 1))

        self.pwConv = nn.Conv2d(in_channels=inputCh, out_channels=inputCh // r, kernel_size=1)

        self.batchNorm = nn.BatchNorm2d(inputCh // r)
        self.act = nn.Hardsigmoid()
        self.convH = nn.Conv2d(in_channels=inputCh // r, out_channels=inputCh, kernel_size=1)
        self.convW = nn.Conv2d(in_channels=inputCh // r, out_channels=inputCh, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape

        xW = self.maxW(x)
        xH = self.maxH(x)

        xW = xW.permute(0, 1, 3, 2)

        x_cat = torch.concat([xH, xW], dim=2)

        x_cat = self.pwConv(x_cat)
        x_cat = self.batchNorm(x_cat)
        x_cat = self.act(x_cat)

        xHn, xWn = torch.split(x_cat, [h, w], dim=2)

        xWn = xWn.permute(0, 1, 3, 2)

        gH = self.act(self.convH(xHn))
        gW = self.act(self.convW(xWn))

        y = gH * gW * x
        return y


print(coordinateAttention(128, 32)(torch.randn(1, 128, 40, 40)).shape)
