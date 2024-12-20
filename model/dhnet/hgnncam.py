import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from .hgnn.hgnn_cycle import HgnnCycle


class HgnnCAM(nn.Module):
    def __init__(self, num_class, num_cycle, K_neigs, num_edge):
        super(HgnnCAM, self).__init__()

        resnet = list(resnet50(pretrained=True).children())
        hm_size, in_channel = 16, 1024
        self.shared = torch.nn.Sequential(*(resnet[0:7]))
        
        self.num_cycle = num_cycle
        
        self.hgnn = HgnnCycle(in_channel=in_channel, mid_channel=512, in_size=hm_size,
                              num_class=num_class, K_neigs=K_neigs, num_edge=num_edge,
                              num_cycle=num_cycle)

    def forward(self, image):  # [n, 6, h, w]
        inverted = image[:, 0:3, :, :]
        manipulated = image[:, 3:6, :, :]

        feat1, feat2 = self.shared(inverted), self.shared(manipulated)
        feat_delta = feat2 - feat1
        
        return self.hgnn(feat_delta)
