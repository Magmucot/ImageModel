
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Генератор DCGAN для изображений 128x128x3
    """
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # latent_dim x 1 x 1 -> ngf*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # ngf*8 x 4 x 4 -> ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # ngf*4 x 8 x 8 -> ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # ngf*2 x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf x 32 x 32 -> ngf//2 x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # ngf//2 x 64 x 64 -> ngf//4 x 128 x 128
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),

            # ngf//4 x 128 x 128 -> 3 x 128 x 128
            nn.ConvTranspose2d(ngf // 4, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    Дискриминатор DCGAN для изображений 128x128x3
    """
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 3 x 128 x 128 -> ndf x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 64 x 64 -> ndf*2 x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*2 x 32 x 32 -> ndf*4 x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*4 x 16 x 16 -> ndf*8 x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 x 8 x 8 -> ndf*8 x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 x 4 x 4 -> ndf*8 x 2 x 2
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 x 2 x 2 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)


def weights_init(m):
    """
    Инициализация весов для DCGAN
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # Тестирование моделей
    latent_dim = 100
    batch_size = 4

    # Генератор
    G = Generator(latent_dim=latent_dim)
    z = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = G(z)
    print(f"Generator output shape: {fake_images.shape}")

    # Дискриминатор
    D = Discriminator()
    output = D(fake_images)
    print(f"Discriminator output shape: {output.shape}")

    # Подсчет параметров
    num_params_G = sum(p.numel() for p in G.parameters())
    num_params_D = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters: {num_params_G:,}")
    print(f"Discriminator parameters: {num_params_D:,}")