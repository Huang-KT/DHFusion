import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.dhnet.utils.hypergraph_utils as hgut
import model.dhnet.utils.construct_H as conH
from .hgnn import Apply_HypergraphConv
from model.dhnet.eam import EAM


class HgnnCycle(nn.Module):
    def __init__(self, in_channel, mid_channel, in_size, num_class,
                 K_neigs, num_edge, num_cycle, drop_path=0.1):
        super(HgnnCycle, self).__init__()

        assert num_edge == in_size*in_size

        self.num_cycle = num_cycle
        self.K_neigs = K_neigs
        self.alpha = 0.9

        if in_channel == mid_channel:
            self.align = nn.Identity()
        else:
            self.align = nn.Sequential(
                nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.PReLU(),
                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel)
            )

        self.construct_H_fun0 = conH.construct_H_with_KNN
        self.construct_H_fun1 = conH.LearningH(in_features=mid_channel, out_features=mid_channel,
                                               features_height=in_size, features_width=in_size,
                                               edges=in_size*in_size, filters=mid_channel)
        
        self.apply_hypergraphConv = Apply_HypergraphConv(in_features=mid_channel, out_features=mid_channel,
                                                         drop_path=drop_path)
        self.act = nn.ELU()
        self.att = EAM(ft_in_ch=mid_channel, num_class=num_class,
                       d_model=mid_channel, nhead=4, dim_feedforward=1024)
        self.prediction = nn.Linear(mid_channel, num_class)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x: torch.Tensor):
        
        z = self.align(x)
        hm_list, hm_cls_list = [], []
        for i in range(self.num_cycle):
            H_S = self.construct_H_fun0(z.detach(), K_neigs=self.K_neigs)  # Short-Term H
            H_L = self.construct_H_fun1(z)  # Long-Term H
            H = self.alpha * H_S + (1 - self.alpha) * H_L
            HyperG = hgut.generate_G_from_H(H)
            z = self.apply_hypergraphConv(z, HyperG)  # construct from z
            z = self.act(z)

            z, [hm, hm_cls] = self.att(z)
            hm_list.append(hm)
            hm_cls_list.append(hm_cls)

        y = F.adaptive_avg_pool2d(z, 1)
        y = y.squeeze()
        y = self.prediction(y)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        
        return y, [hm_list, hm_cls_list]
