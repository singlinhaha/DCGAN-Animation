import os
import torch
import shutil
import albumentations as A
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from utils.DataReader import DataReader
from model.model import Generator, Discriminator
from utils.general import plot_result
from tensorboardX import SummaryWriter


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def train(cfg):
    if isinstance(cfg.img_size, int):
        img_size = (cfg.img_size, cfg.img_size)
    else:
        img_size = cfg.img_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkponit"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "img_show"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "log"), exist_ok=True)

    transform = A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 将图片放缩到-1~1
    ])
    # dataset
    dataset = DataReader(cfg.root_path, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

    # model
    net_g = Generator(cfg.input_size, cfg.ndg, cfg.out_channels)
    net_d = Discriminator(cfg.out_channels, cfg.ndf)
    net_g.train().to(device)
    net_d.train().to(device)

    # optimize
    optimize_g = Adam(net_g.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    optimize_d = Adam(net_d.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    # loss
    criterions = nn.BCELoss().to(device)
    epoch_item = len(dataloader)
    true_label = torch.ones(cfg.batch_size).float().to(device)
    fake_label = torch.zeros(cfg.batch_size).float().to(device)
    optimize_g.zero_grad()
    optimize_d.zero_grad()

    if cfg.resume and os.path.exists(os.path.join(cfg.output_dir, "checkponit", "last.pt")):
        state_dict = torch.load(os.path.join(cfg.output_dir, "checkponit", "last.pt"))
        net_d.load_state_dict(state_dict["D_state_dict"])
        net_g.load_state_dict(state_dict["G_state_dict"])
        optimize_d.load_state_dict(state_dict["D_optimizer_state_dict"])
        optimize_g.load_state_dict(state_dict["G_optimizer_state_dict"])
        start_epoch = state_dict["epoch"] + 1
    else:
        # 清空目录
        shutil.rmtree(os.path.join(cfg.output_dir, "log"))
        os.mkdir(os.path.join(cfg.output_dir, "log"))
        start_epoch = 0

    write = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "log"), comment='loss')
    noise = torch.randn((cfg.batch_size, cfg.input_size, 1, 1)).to(device)
    fixed_noise = torch.randn((64, cfg.input_size, 1, 1), device=device)     # 创建潜在向量，我们将用它来可视化生成器的进程
    D_loss = -1
    G_loss = -1
    D_x = -1
    G_D_z1 = -1
    G_D_z2 = -1
    for epoch in range(start_epoch, cfg.epochs):
        with tqdm(dataloader, desc="epoch {}".format(epoch)) as pbar:
            for i, real_img in enumerate(pbar):
                real_img = real_img.to(device)

                if ((epoch * epoch_item) + (i + 1)) % cfg.d_item == 0:
                    ########## 训练判别器 ###########
                    optimize_d.zero_grad()
                    # net_d.train()
                    # net_g.eval()

                    # 用真实图片训练判别器
                    output = net_d(real_img)
                    D_real_loss = criterions(output, true_label)
                    D_real_loss.backward()
                    D_x = output.mean().item()

                    # 用随机生成的假图训练判别器
                    noise.data.copy_(torch.randn((cfg.batch_size, cfg.input_size, 1, 1)))
                    fake_img = net_g(noise).detach()
                    output = net_d(fake_img)
                    D_fake_loss = criterions(output, fake_label)
                    D_fake_loss.backward()
                    optimize_d.step()
                    D_loss =  D_real_loss.item()+D_fake_loss.item()
                    G_D_z1 = output.mean().item()

                    # 写入日志
                    write.all_writers("Discriminator_loss", D_loss)

                if ((epoch * epoch_item) + (i + 1)) % cfg.g_item == 0:
                    ########## 训练生成器 ###########
                    optimize_g.zero_grad()
                    # net_g.train()
                    # net_d.eval()

                    noise.data.copy_(torch.randn((cfg.batch_size, cfg.input_size, 1, 1)))
                    fake_img = net_g(noise)
                    output = net_d(fake_img)
                    G_loss = criterions(output, true_label)
                    G_loss.backward()
                    optimize_g.step()
                    G_loss = G_loss.item()
                    G_D_z2 = output.mean().item()
                    write.all_writers("Generator_loss", G_loss)

                pbar.set_postfix_str("loss_D: {:.5f} Loss_G: {:.5f} D(x): {:.5f} D(G(z)): {:.5f}/{:.5f}".format(
                    D_loss, G_loss, D_x, G_D_z1, G_D_z2
                ))

                if ((epoch * epoch_item) + (i + 1)) % cfg.show_item == 0:
                    with torch.no_grad():
                        # net_g.eval()
                        fix_fake_image = net_g(fixed_noise)
                        torchvision.utils.save_image(fix_fake_image, os.path.join(cfg.output_dir, "img_show",
                                                                                  "epoch-{}_item-{}.png".format(epoch, i+1)),
                                                     normalize=True, value_range=(-1, 1))
                        # fix_fake_image = denorm(fix_fake_image).detach().cpu().numpy()
                        # fix_fake_image = (fix_fake_image * 255).astype(np.uint8)
                        # plot_result(fix_fake_image, num_epoch=epoch+1, num_item=i+1, save=True,
                        #             save_dir=os.path.join(cfg.output_dir, "img_show"), fig_size=(8, 8))

        # epoch_save = {"G_state_dict": net_g.state_dict(),
        #               "D_state_dict": net_d.state_dict(),
        #               "D_optimizer_state_dict": optimize_d.state_dict(),
        #               "G_optimizer_state_dict": optimize_g.state_dict(),
        #               "epoch": epoch}
        #
        # torch.save(epoch_save, os.path.join(cfg.output_dir, "checkponit", "last.pt"))
        torch.save(net_g, os.path.join(cfg.output_dir, "checkponit", "net_g-{}.pt".format(epoch)))
        torch.save(net_d, os.path.join(cfg.output_dir, "checkponit", "net_d-{}.pt".format(epoch)))


if __name__ == "__main__":
    class CFG:
        def __init__(self):
            self.root_path = r"E:\code\dataset\faces\faces"      # 数据集存放路径
            self.img_size = 96  # 图片尺寸
            self.batch_size = 256
            self.CUDA = "0"
            self.output_dir = r"output/exp1"    # 输出目录

            # model set
            self.input_size = 100   # 噪声维度
            self.ndg = 64   # 判别器feature map数
            self.ndf = 64   # 生成器feature map数
            self.out_channels = 3   # 图片通道数
            self.lr_g = 2e-4    # 生成器的学习率
            self.lr_d = 2e-4    # 判别器的学习率

            # train
            self.epochs = 20    # 训练轮数
            self.d_item = 1     # 每1个batch训练一次判别器
            self.g_item = 1     # 每1个batch训练一次生成器
            self.show_item = 50     # 隔多少个batch生成一次示例图
            self.resume = False     # 是否断点续训


    cfg = CFG()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA
    train(cfg)