import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import ReplayBuffer
from .datasets import ImageDataset, ValDataset, PredictDataset
from Model.CycleGan import *
from Model import networks
import torch
from .utils import Resize, ToTensor, smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine, ToPILImage
from .transformer import Transformer_2D
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2
from .Perceptual import PerceptualLoss


class US_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        """生成器G、F"""
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
        # self.gpu_ids = '0'
        # self.netG_A2B = networks.define_G(1, 1, 64, 'unet_256', 'instance',
        #                                   not 'store_true', 'normal', 0.02, [0])
        # self.netG_B2A = networks.define_G(1, 1, 64, 'unet_256', 'instance',
        #                                   not 'store_true', 'normal', 0.02, [0])

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                            lr=config['lr'], betas=(0.5, 0.999))
        """判别器 fakeB和fake fakeB加 recovered_A """
        self.netD_fake_B = Discriminator(config['input_nc']).cuda()
        # self.netD_fake_B = networks.define_D(1, 64, 'basic', 3, 'instance', 'normal', 0.02, [0])
        self.optimizer_D_fake_B = torch.optim.Adam(self.netD_fake_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        # self.netD_fake_fake_B = Discriminator(config['input_nc']).cuda()
        # self.netD_fake_fake_B = networks.define_D(1, 64, 'basic', 3, 'instance', 'normal', 0.02, [0])
        # self.optimizer_D_fake_fake_B = torch.optim.Adam(self.netD_fake_fake_B.parameters(),
        #                                                 lr=config['lr'], betas=(0.5, 0.999))
        # self.netD_recovered_A = networks.define_D(1, 64, 'basic', 3, 'instance', 'normal', 0.02, [0])
        self.netD_recovered_A = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_recovered_A = torch.optim.Adam(self.netD_recovered_A.parameters(),
                                                        lr=config['lr'], betas=(0.5, 0.999))
        """配准网络"""
        self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
        self.spatial_transform = Transformer_2D().cuda()
        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        """损失函数定义"""
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pct_loss = PerceptualLoss(nn.MSELoss().cuda(), [3, 8, 15, 22], device)

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level

        transforms_1 = [ToPILImage(),
                        # RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                        #              scale=[1 - 0.02 * level, 1 + 0.02 * level], fillcolor=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                        # RandomAffine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02], fillcolor=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2,
                         unaligned=False, ),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        self.predict_data = DataLoader(PredictDataset(config['test_dataroot'], transforms_=val_transforms),
                                       batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

    def train(self):
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                """Set model input"""
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                """GAN loss 生成器G、F和配准网络R的损失函数构建"""
                # 生成器和配准网络的优化器
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                # 分别对应G(x)、F(G(x))、G(F(G(x)))
                fake_B = self.netG_A2B(real_A)
                recovered_A = self.netG_B2A(fake_B)
                fake_fake_B = self.netG_A2B(recovered_A)
                # 计算身份一致identity
                idt_B = self.netG_A2B(real_B)
                idt_A = self.netG_B2A(real_A)
                # Register_fake_B对应R(G(x),y)
                Trans_fake_B = self.R_A(fake_B, real_B)
                Register_fake_B = self.spatial_transform(fake_B, Trans_fake_B)
                # Register_fake_fake_B对应R(G(F(G(x))),y)
                # Trans_fake_fake_B = self.R_A(fake_fake_B, real_B)
                # Register_fake_fake_B = self.spatial_transform(fake_fake_B, Trans_fake_fake_B)

                # Cycle-consistency loss 计算x和F(G(x))循环一致损失
                # loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)
                # 生成器欺骗判别器loss
                pred_fake_B = self.netD_fake_B(fake_B)
                loss_GAN_fake_B = self.MSE_loss(pred_fake_B, self.target_real)

                pred_recovered_A = self.netD_recovered_A(recovered_A)
                loss_GAN_recovered_A = self.MSE_loss(pred_recovered_A, self.target_real)
                # 配准网络损失
                SR_fB_loss = self.config['Corr_lamda'] * self.pct_loss(Register_fake_B.repeat(1, 3, 1, 1), real_B.repeat(1, 3, 1, 1))
                SM_fB_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans_fake_B)
                # SR_ffB_loss = self.config['Corr_lamda'] * self.L1_loss(Register_fake_fake_B, real_B)
                # SM_ffB_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans_fake_fake_B)
                # identity Loss
                idt_B_loss = self.config['idt_lamda'] * self.L1_loss(idt_B, real_B)
                idt_A_loss = self.config['idt_lamda'] * self.L1_loss(idt_A, real_A)
                # G(F(G(x))) 和 y的L1损失Progressive Loss
                progressive_loss = self.config['Progressive_lamda'] * self.L1_loss(fake_fake_B, real_B)
                # 总损失
                loss_Total = (loss_GAN_fake_B + SR_fB_loss + SM_fB_loss + loss_GAN_recovered_A + progressive_loss +
                              idt_B_loss + idt_A_loss)
                loss_Total.backward()
                self.optimizer_G.step()
                self.optimizer_R_A.step()

                """Discriminator fake_B"""
                self.optimizer_D_fake_B.zero_grad()
                # Real loss
                pred_real = self.netD_fake_B(real_B)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                # Fake loss
                fake_B = self.fake_A_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_fake_B(fake_B.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                # Total loss
                loss_D_fake_B = (loss_D_real + loss_D_fake)
                loss_D_fake_B.backward()
                self.optimizer_D_fake_B.step()

                # """Discriminator fake_fake_B"""
                # self.optimizer_D_fake_fake_B.zero_grad()
                # # Real loss
                # pred_real = self.netD_fake_fake_B(real_B)
                # loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                # # Fake loss
                # fake_fake_B = self.fake_B_buffer.push_and_pop(fake_fake_B)
                # pred_fake = self.netD_fake_fake_B(fake_fake_B.detach())
                # loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                # Total loss
                # loss_D_fake_fake_B = (loss_D_real + loss_D_fake)
                # loss_D_fake_fake_B.backward()
                # self.optimizer_D_fake_fake_B.step()

                """Discriminator recovered_A"""
                self.optimizer_D_recovered_A.zero_grad()
                # Real loss
                pred_real = self.netD_recovered_A(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                # Fake loss
                recovered_A = self.fake_B_buffer.push_and_pop(recovered_A)
                pred_fake = self.netD_recovered_A(recovered_A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                # Total loss
                loss_D_recovered_A = (loss_D_real + loss_D_fake)
                loss_D_recovered_A.backward()
                self.optimizer_D_recovered_A.step()

                self.logger.log({'loss_D_fake_B': loss_D_fake_B, 'loss_D_recovered_A': loss_D_recovered_A,
                                 "loss_Total": loss_Total},
                                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B,
                                        'fake_fake_B': fake_fake_B})

            """Save models checkpoints"""
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')

            """val"""
            with torch.no_grad():
                MAE = 0
                PSNR = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    psnr = self.PSNR(fake_B, real_B)
                    mae = self.MAE(fake_B, real_B)
                    PSNR += psnr
                    MAE += mae
                    num += 1
                print('PSNR:', PSNR / num)
                print('MAE:', MAE / num)

    def test(self, ):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                fake_B = self.netG_A2B(real_A)
                fake_B = fake_B.detach().cpu().numpy().squeeze()
                psnr = self.PSNR(fake_B, real_B)
                ssim = compare_ssim(fake_B, real_B)
                mae = self.MAE(fake_B, real_B)
                PSNR += psnr
                SSIM += ssim
                MAE += mae
                num += 1
            print('PSNR:', PSNR / num)
            print('SSIM:', SSIM / num)
            print('MAE:', MAE / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mae = np.abs(fake[x, y] - real[x, y]).mean()
        return mae / 2  # from (-1,1) normaliz  to (0,1)

    def predict(self, ):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():
            for i, batch in enumerate(self.predict_data):
                fake_B = self.netG_A2B(batch[1].cuda())
                self.save_png(fake_B.detach().cpu().numpy().squeeze(),
                              "data/test/result/" + str(batch[0][0]).split(".")[0] + ".png")

    def save_png(self, fake_B, root):
        fake_B = (fake_B + 1) * 127.5
        cv2.imwrite(root, fake_B)
