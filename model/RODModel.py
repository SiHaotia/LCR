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
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
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
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/flow_imagenet.pt')
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
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
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

class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)

        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)


        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)

        return result_r, result_o, result_d

class JointShareClsModel(nn.Module):
    def __init__(self, config):
        super(JointShareClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
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
        self.num_class = config['setting']['num_class']
        self.fc_out = nn.Linear(256, self.num_class)

        self.weight_r_ = nn.Linear(config['setting']['num_layers'], 1, bias=False)
        self.weight_r = self.weight_r_.weight
        self.weight_o_ = nn.Linear(config['setting']['num_layers'], 1, bias=False)
        self.weight_o = self.weight_o_.weight
        self.weight_d_ = nn.Linear(config['setting']['num_layers'], 1, bias=False)
        self.weight_d = self.weight_d_.weight

        self.additional_layers_r = nn.ModuleList()
        self.additional_layers_o = nn.ModuleList()
        self.additional_layers_d = nn.ModuleList()
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        of_feat = self.of_encoder(of)
        depth_feat = self.depth_encoder(depth)

        return rgb_feat, of_feat, depth_feat

    def add_layer(self, is_i=''):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()

        if is_i == 'rgb':
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_o[0].weight.data+self.emb_d[0].weight.data)*0.5)
                new_layer.bias.copy_((self.emb_o[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_r.append(new_layer)
        elif is_i == 'of':
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_r[0].weight.data+self.emb_d[0].weight.data)*0.5)
                new_layer.bias.copy_((self.emb_r[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_o.append(new_layer)
        else:
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_r[0].weight.data + self.emb_o[0].weight.data) * 0.5)
                new_layer.bias.copy_((self.emb_r[0].bias.data + self.emb_o[0].bias.data) * 0.5)
            self.additional_layers_d.append(new_layer)

    def classfier(self, x, is_i=''):
        if is_i=='rgb':
            result_r = self.emb_r(x)
            r = torch.mean(result_r, 0, True)
            feature = self.fc_out(result_r)
            i = 0
            for layer in self.additional_layers_r:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + self.weight_r[0][i] * self.fc_out(addf)
                i = i + 1
            feature = feature/(i+1)
        elif is_i=='of':
            result_o = self.emb_o(x)
            r = torch.mean(result_o, 0, True)
            feature = self.fc_out(result_o)
            j = 0
            for layer in self.additional_layers_o:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + self.weight_o[0][j] * self.fc_out(addf)
                j = j + 1
            feature = feature / (j+1)
        else:
            result_d = self.emb_d(x)
            r = torch.mean(result_d, 0, True)
            feature = self.fc_out(result_d)
            z = 0
            for layer in self.additional_layers_d:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + self.weight_d[0][z] * self.fc_out(addf)
                z = z + 1
            feature = feature / (z+1)
        return feature, r

class JointShareReinClsModel(nn.Module):
    def __init__(self, config):
        super(JointShareReinClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
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
            nn.Linear(64, config['setting']['num_layers']),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        rgb = x1
        of = x2
        depth = x3

        rgb_feat = self.rgb_encoder(rgb)
        of_feat = self.of_encoder(of)
        depth_feat = self.depth_encoder(depth)

        return rgb_feat, of_feat, depth_feat

    def add_layer(self, is_i=''):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()

        if is_i == 'rgb':
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_o[0].weight.data+self.emb_d[0].weight.data)*0.5)
                new_layer.bias.copy_((self.emb_o[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_r.append(new_layer)
        elif is_i == 'of':
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_r[0].weight.data+self.emb_d[0].weight.data)*0.5)
                new_layer.bias.copy_((self.emb_r[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_o.append(new_layer)
        else:
            with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
                new_layer.weight.copy_((self.emb_r[0].weight.data + self.emb_o[0].weight.data) * 0.5)
                new_layer.bias.copy_((self.emb_r[0].bias.data + self.emb_o[0].bias.data) * 0.5)
            self.additional_layers_d.append(new_layer)

    def classfier(self, x, w, is_i=''):
        if is_i=='rgb':
            result_r = self.emb_r(x)
            r = torch.mean(result_r, 0, True)
            feature = self.fc_out(result_r)
            i = 0
            for layer in self.additional_layers_r:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + w[i] * self.fc_out(addf)
                i = i + 1
            feature = feature/(i+1)
        elif is_i=='of':
            result_o = self.emb_o(x)
            r = torch.mean(result_o, 0, True)
            feature = self.fc_out(result_o)
            j = 0
            for layer in self.additional_layers_o:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + w[j] * self.fc_out(addf)
                j = j + 1
            feature = feature / (j+1)
        else:
            result_d = self.emb_d(x)
            r = torch.mean(result_d, 0, True)
            feature = self.fc_out(result_d)
            z = 0
            for layer in self.additional_layers_d:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                feature = feature + w[z] * self.fc_out(addf)
                z = z + 1
            feature = feature / (z+1)
        return feature, r

class JointGBShareReinClsModel(nn.Module):
    def __init__(self, config):
        super(JointGBShareReinClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
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

        return rgb_feat, of_feat, depth_feat

    def add_layer(self, is_i=''):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)
        if is_i == 'rgb':
            # with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
            #     new_layer.weight.copy_((self.emb_o[0].weight.data+self.emb_d[0].weight.data)*0.5)
            #     new_layer.bias.copy_((self.emb_o[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_r.append(new_layer)
        elif is_i == 'of':
            # with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
            #     new_layer.weight.copy_((self.emb_r[0].weight.data+self.emb_d[0].weight.data)*0.5)
            #     new_layer.bias.copy_((self.emb_r[0].bias.data+self.emb_d[0].bias.data)*0.5)
            self.additional_layers_o.append(new_layer)
        else:
            # with torch.no_grad():  # �����ݶȼ����Ա��ⲻ��Ҫ�ļ���
            #     new_layer.weight.copy_((self.emb_r[0].weight.data + self.emb_o[0].weight.data) * 0.5)
            #     new_layer.bias.copy_((self.emb_r[0].bias.data + self.emb_o[0].bias.data) * 0.5)
            self.additional_layers_d.append(new_layer)

    def classfier(self, x, w, is_i=''):
        if is_i=='rgb':
            result_r = self.emb_r(x)
            r = torch.mean(result_r, 0, True)
            feature = self.fc_out(result_r)
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
            r = torch.mean(result_o, 0, True)
            feature = self.fc_out(result_o)
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
            r = torch.mean(result_d, 0, True)
            feature = self.fc_out(result_d)
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