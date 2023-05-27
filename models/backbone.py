# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    

    
class MJ_CONV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 36, kernel_size = 11, padding = 5)
        self.batch_1 = nn.BatchNorm1d(36)
        self.pool_1 = nn.MaxPool1d(kernel_size = 1)

        self.conv_2 = nn.Conv1d(36, 36, kernel_size = 11, padding = 5)
        self.batch_2 = nn.BatchNorm1d(36)
        self.pool_2 = nn.MaxPool1d(kernel_size = 1)

        self.conv_3 = nn.Conv1d(36, 36, kernel_size = 11, padding = 5)
        self.batch_3 = nn.BatchNorm1d(36)
        self.pool_3 = nn.MaxPool1d(kernel_size = 1)

        self.num_channels = 108
    def forward(self, tensor_list: NestedTensor):
        x_1 = output = self.pool_1(self.batch_1(F.relu(self.conv_1(tensor_list.tensors))))
        x_2 = output = self.pool_2(self.batch_2(F.relu(self.conv_2(x_1))))
        x_3 = output = self.pool_3(self.batch_3(F.relu(self.conv_3(x_2))))

        #print(x_1.shape)
        #print(x_2.shape)
        #print(x_3.shape)
        features = torch.cat((x_1, x_2, x_3), 1)

        #xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        
        x = features[:,:,None,:]
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-1:]).to(torch.bool)[0]
        out['CNN_OUT'] = NestedTensor(x, mask)
        return out


class building_block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size = 1, pool_k_size = 1,conv_padd = 1):
        super(building_block, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = k_size, padding = conv_padd)
        self.batch = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size = pool_k_size)

    def forward(self, x):
        output = self.pool(self.batch(F.relu(self.conv(x))))

        return output


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.masks
    #backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone2 = MJ_CONV2()
    model = Joiner(backbone2, position_embedding)
    #model.num_channels = backbone.num_channels
    model.num_channels = 108
    return model
