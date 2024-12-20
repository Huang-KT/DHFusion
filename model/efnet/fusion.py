import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential
import torchvision.transforms as transforms

from model.pSp.encoders.helpers import get_block, bottleneck_IR, bottleneck_IR_SE
from .stylegan2 import StyleGANv2Generator

from .fusion_modules import *


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt
    
class GradualStyleEncoder(nn.Module):
    def __init__(self, in_size=256, out_size=1024, mode='ir_se'):
        super(GradualStyleEncoder, self).__init__()
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = [
			get_block(in_channel=64, depth=64, num_units=3),     # 256 -> 128
			get_block(in_channel=64, depth=128, num_units=4),    # 128 -> 64*
			get_block(in_channel=128, depth=256, num_units=14),  # 64 -> 32*
			get_block(in_channel=256, depth=512, num_units=3)    # 32 -> 16*
		]
        self.feat_index = [6, 20, 23]

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = int(math.log(out_size, 2)) * 2 - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, x_edit, hm, return_latent=True):
        modulelist = list(self.body._modules.values())
        # encoder feats for x
        x = self.input_layer(x)
        encoder_feats_x = []
        for i, l in enumerate(modulelist):
            x = l(x)
            if i in self.feat_index:
                encoder_feats_x.append(x)
        encoder_feats_x = encoder_feats_x[::-1]

        # encoder feats for x_edit
        if x_edit is None:
            assert hm is None
            encoder_feats = encoder_feats_x
        else:
            assert hm is not None
            x_edit = self.input_layer(x_edit)
            encoder_feats_edit = []
            for i, l in enumerate(modulelist):
                x_edit = l(x_edit)
                if i in self.feat_index:
                    encoder_feats_edit.append(x_edit)
            encoder_feats_edit = encoder_feats_edit[::-1]
            
            ## fusion x and x_edit
            encoder_feats = [encoder_feats_x[i] * (1 - hm[i]) + encoder_feats_edit[i] * hm[i] for i in range(3)]

        if return_latent:
            encoder_feats_latent = encoder_feats
            latents = []
            for j in range(self.coarse_ind):
                latents.append(self.styles[j](encoder_feats_latent[0]))
            
            p2 = self._upsample_add(encoder_feats_latent[0], self.latlayer1(encoder_feats_latent[1]))
            for j in range(self.coarse_ind, self.middle_ind):
                latents.append(self.styles[j](p2))

            p1 = self._upsample_add(p2, self.latlayer2(encoder_feats_latent[2]))
            for j in range(self.middle_ind, self.style_count):
                latents.append(self.styles[j](p1))

            out = torch.stack(latents, dim=1)
            return out, encoder_feats[-1]
        else:
            return encoder_feats[-1]


class PixelAttention_SE(nn.Module):
    def __init__(self, channel):
        super(PixelAttention_SE, self).__init__()
        self.conv = bottleneck_IR_SE(in_channel=channel, depth=channel, stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv(x)) * x
    

class DeghostingModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeghostingModule, self).__init__()
        self.pixelatt = PixelAttention_SE(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, ft):
        return self.conv2(self.act(self.conv1(self.pixelatt(ft))))


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()

        self.in_size = in_size

        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        pSp_ch = {16: 512, 32: 256, 64: 128}
        for c in [64]:
            num_channels = channels[c]
            self.fusion_out.append(
                DeghostingModule(num_channels + pSp_ch[c], num_channels))
            self.fusion_skip.append(
                DeghostingModule(pSp_ch[c] + 3, 3))
            

        self.Upsample = nn.ModuleList()
        for c in [64, 128, 256, 512, 1024]:
            if c == 64:
                in_channels = pSp_ch[c]
            else:
                in_channels = 2 * channels[c]

            if c < out_size:
                out_channels = channels[c * 2]
                self.Upsample.append(
                    nn.Sequential(
                        PixelAttention_SE(in_channels),
                        PixelShufflePack(in_channels, out_channels, 2, 3)))
            else:
                self.Upsample.append(
                    nn.Sequential(
                        PixelAttention_SE(in_channels),
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64, 3, 3, 1, 1)))


class Fusion(nn.Module):
    def __init__(self, in_size, out_size,
                 start_from_latent_avg=True, pretrain=None):
        super(Fusion, self).__init__()

        self.net = Fusion0(in_size, out_size, start_from_latent_avg, pretrain)

    def forward(self, x, x_edit=None, hm=None):

        return self.net(x, x_edit, hm)



class Fusion0(nn.Module):
    def __init__(self, in_size, out_size,
                 start_from_latent_avg=True, pretrain=None):
        super(Fusion0, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.start_from_latent_avg = start_from_latent_avg
        self.pretrain = pretrain
        
        # Encoder
        self.encoder = GradualStyleEncoder(in_size, out_size, 'ir_se')
        # Decoder
        self.decoder = Decoder(in_size=in_size, out_size=out_size)
        # Generator
        self.generator = StyleGANv2Generator(out_size=out_size, style_channels=512,
                                             pretrained=self.pretrain['styleGAN2_weight_url'])
        for p in self.generator.parameters():
            p.requires_grad = False
        
        self.load_weights()

    def load_weights(self):
        checkpoint_path = self.pretrain['pSp']
        print('Loading pSp from checkpoint: {}'.format(checkpoint_path))
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        if self.start_from_latent_avg:
            self.latent_avg = ckpt['latent_avg'].to('cuda')

    def forward(self, x, x_edit=None, hm=None, return_latent=False):

        # encoder
        latent, encoder_feats = self.encoder(x, x_edit, hm)
        if self.start_from_latent_avg:
            latent = latent + self.latent_avg.repeat(latent.shape[0], 1, 1)

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]

        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1
        generator_feats = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):


            if out.size(2) == 64:
                feat = encoder_feats
                
                out = torch.cat([out, feat], dim=1)
                out = self.decoder.fusion_out[0](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.decoder.fusion_skip[0](skip)


            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)


            if out.size(2) > 64:
                generator_feats.append(out)

            _index += 2

        # decoder
        res = encoder_feats
        for i, block in enumerate(self.decoder.Upsample):
            if i > 0:
                res = torch.cat([res, generator_feats[i - 1]], dim=1)
            res = block(res)

        if return_latent:
            return res, latent
        else:
            return res
    