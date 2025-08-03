from utils import *
from os import path
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.Resnet import resnet18

import torch.nn as nn
import torch
from model.Model3D import InceptionI3d

class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        pretrained_dict = torch.load('your pretrained pt file')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        pretrained_dict = torch.load('your pretrained pt file')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        pretrained_dict = torch.load('your pretrained pt file')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048

class RGBClsModel(nn.Module):
    def __init__(self, config):
        super(RGBClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)

        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        rgb = x1
        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)
        return result_r


class OFClsModel(nn.Module):
    def __init__(self, config):
        super(OFClsModel, self).__init__()
        self.of_encoder = OFEncoder(config)

        self.hidden_dim = 1024
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        of = x1
        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)
        return result_o

class DepthClsModel(nn.Module):
    def __init__(self, config):
        super(DepthClsModel, self).__init__()
        self.depth_encoder = DepthEncoder(config)

        self.hidden_dim = 1024
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        depth = x1
        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)
        return result_d






class JointGBShareReinClsModel_LMM(nn.Module):
    def __init__(self, config):
        super(JointGBShareReinClsModel_LMM, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.learnd_encoder = RGBEncoder(config)
        self.hidden_dim = 1024

        self.learn_modal = nn.Parameter(torch.randn(1, 3, 64, 224, 224), requires_grad=True)

        self.emb_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.emb_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.emb_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.emb_l = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']
        self.fc_out = nn.Linear(256, self.num_class)

        self.additional_layers_r = nn.ModuleList()
        self.additional_layers_o = nn.ModuleList()
        self.additional_layers_d = nn.ModuleList()
        self.relu = nn.ReLU()

        self.rein_network1 = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        of_feat = self.of_encoder(of)
        depth_feat = self.depth_encoder(depth)
        learn_feat = self.learnd_encoder(self.learn_modal)

        return rgb_feat, of_feat, depth_feat, learn_feat

    def add_layer(self, is_i=''):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)
        if is_i == 'rgb':

            self.additional_layers_r.append(new_layer)
        elif is_i == 'of':

            self.additional_layers_o.append(new_layer)
        else:

            self.additional_layers_d.append(new_layer)

    def classfier(self, x, hide_l, w, is_i=''):
        if is_i=='rgb':
            result_r = self.emb_r(x)
            result = result_r + w * hide_l
            r = torch.mean(result, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            i = 0
            layerlen = len(self.additional_layers_r)
            for layer in self.additional_layers_r:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                i = i + 1
                if i < layerlen:
                    o_fea = feature
        elif is_i=='of':
            result_o = self.emb_o(x)
            result = result_o + w * hide_l
            r = torch.mean(result, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            j = 0
            layerlen = len(self.additional_layers_o)
            for layer in self.additional_layers_o:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                j = j + 1
                if j < layerlen:
                    o_fea = feature
        else:
            result_d = self.emb_d(x)
            result = result_d + w * hide_l
            r = torch.mean(result, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            z = 0
            layerlen = len(self.additional_layers_d)
            for layer in self.additional_layers_d:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                z = z + 1
                if z < layerlen:
                    o_fea = feature
        return feature, r, o_fea, add_fea
