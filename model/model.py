import torch
from torch import nn


def upsample_module(input_channle, output_channle, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channle, output_channle, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(output_channle),
        nn.ReLU(inplace=True)
    )


def conv2d_module(input_channle, output_channle, kernel_size, stride, padding, BN=True):
    if BN:
        return nn.Sequential(
            nn.Conv2d(input_channle, output_channle, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channle),
            nn.LeakyReLU(0.2, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channle, output_channle, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )


class Generator(nn.Module):
    def __init__(self, input_size, ngf, output_channels=3):
        super(Generator, self).__init__()
        # # 假定输入为一张1*1*nz维的数据
        self.net = nn.Sequential(
            # input: [b, input_size, 1, 1] => output: [b, ngf * 8, 4, 4]
            upsample_module(input_size, ngf * 8, kernel_size=4, stride=1, padding=0),
            # input: [b, ngf * 8, 4, 4] => output: [b, ngf * 4, 8, 8]
            upsample_module(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            # input: [b, ngf * 4, 8, 8] => output: [b, ngf * 2, 16, 16]
            upsample_module(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            # input: [b, ngf * 2, 16, 16] => output: [b, ngf, 32, 32]
            upsample_module(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            # input: [b, ngf, 32, 32] => output: [b, 3, 96, 96]
            nn.ConvTranspose2d(ngf, output_channels, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input: [b, input_size, 96, 196] => output: [b, ngf, 32, 32]
            conv2d_module(in_channels, ndf, kernel_size=5, stride=3, padding=1, BN=False),
            # input: [b, ngf, 32, 32] => output: [b, ngf*2, 16, 16]
            conv2d_module(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, BN=True),
            # input: [b, ngf*2, 16, 16] => output: [b, ngf*4, 8, 8]
            conv2d_module(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, BN=True),
            # input: [b, ngf*4, 8, 8] => output: [b, ngf*8, 4, 4]
            conv2d_module(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, BN=True),
            # input: [b, ngf*8, 4, 4] => output: [b, 1, 1, 1]
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(-1)


if __name__ == "__main__":
    x = torch.rand((1, 100, 1, 1))
    model = Generator(100, 64)
    net = Discriminator(3, 64)
    y = model(x)
    z = net(y)
    print()