from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from .Resnet import resnet18, resnet34, resnet50
import copy

class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')
        elif config['text']["name"] == 'resnet34':
            self.audio_net = resnet34(modality='audio')
        elif config['text']["name"] == 'resnet50':
            self.audio_net = resnet50(modality='audio')

        # self.audio_net = resnet18(modality='audio')
        # self.norm = nn.Sequential(
        #     nn.BatchNorm1d(512), #-----------添加
        #     nn.GELU(),#-----------添加
        # )

    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        elif config['visual']["name"] == 'resnet34':
            self.video_net = resnet34(modality='visual')
        elif config['visual']["name"] == 'resnet50':
            self.video_net = resnet50(modality='visual')
        # self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v

class ImageEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(ImageEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.image_net = resnet18(modality='image')
        elif config['visual']["name"] == 'resnet34':
            self.image_net = resnet34(modality='image')
        elif config['visual']["name"] == 'resnet50':
            self.image_net = resnet50(modality='image')
        # self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.image_net(video)
        v = F.adaptive_avg_pool2d(v, 1)
        v = torch.flatten(v, 1)
        return v

class AudioClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AudioClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)

        self.hidden_dim = 512
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 64),
            nn.Linear(256, config['setting']['num_class'])
        )

    def forward(self, audio):
        a_feature = self.audio_encoder(audio)
        # a_feature1 = self.linear1(a_feature)
        # a_feature2 = self.linear2(a_feature)
        # a_feature3 = self.linear3(a_feature)
        # a_feature4 = self.linear4(a_feature)
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2)) / 2.0
        # result_a = (self.cls_a(a_feature1) + self.cls_a(a_feature2) + self.cls_a(a_feature3) + self.cls_a(a_feature4)) / 4.0
        result_a = self.cls_a(a_feature)
        return result_a


class VideoClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(VideoClassifier, self).__init__()
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)

        self.hidden_dim = 512
        if config['visual']["name"] == 'resnet50':
            self.hidden_dim = 2048
        # self.linear1 = nn.Linear(self.hidden_dim, 256)
        # self.linear2 = nn.Linear(self.hidden_dim, 256)
        # self.linear3 = nn.Linear(self.hidden_dim, 256)
        # self.linear4 = nn.Linear(self.hidden_dim, 256)
        self.cls_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 64),
            nn.Linear(256, config['setting']['num_class'])
        )

    def forward(self, video):
        v_feature = self.video_encoder(video)
        # v_feature1 = self.linear1(v_feature)
        # v_feature2 = self.linear2(v_feature)
        # v_feature3 = self.linear3(v_feature)
        # v_feature4 = self.linear4(v_feature)
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2)) / 2.0
        # result_v = (self.cls_v(v_feature1) + self.cls_v(v_feature2) + self.cls_v(v_feature3) + self.cls_v(v_feature4)) / 4.0
        result_v = self.cls_v(v_feature)
        return result_v



class AVEClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVEClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 1024
        if config['visual']["name"] == 'resnet50':
            self.hidden_dim = 2048 * 2
        self.cls = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        feature = torch.cat((a_feature, v_feature), dim=1)
        result = self.cls(feature)
        return result



class AVGBShareClassifier_newModal(nn.Module):
    def __init__(self, config, mask_model=1):
        super(AVGBShareClassifier_newModal, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.learn_encoder = ImageEncoder(config, mask_model)
        self.hidden_dim = 512

        self.learn_modal = nn.Parameter(torch.randn(1, 3, 224, 224), requires_grad=True)

        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_l = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )

        self.num_class = config['setting']['num_class']

        self.embedding_la = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.ReLU(),
        )
        self.embedding_lv = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.ReLU(),
        )
        self.embedding_back = nn.Sequential(
            nn.Linear(self.hidden_dim+4, self.hidden_dim),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(256, self.num_class)
        self.additional_layers_a = nn.ModuleList()
        self.additional_layers_v = nn.ModuleList()

        self.relu = nn.ReLU()
        self.rein_network1 = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, video):
        B = audio.size(0)
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)

        # la_feature = self.embedding_la(a_feature)
        # la_feature = torch.mean(la_feature, dim=0).unsqueeze(0)
        # lv_feature = self.embedding_lv(v_feature)
        # lv_feature = torch.mean(lv_feature, dim=0).unsqueeze(0)

        l_feature = self.learn_encoder(self.learn_modal)

        # print(l_feature.shape, la_feature.shape, lv_feature.shape)
        # l_feature = torch.cat((l_feature, la_feature, lv_feature), dim=1)
        # l_feature = self.embedding_back(l_feature)
        return a_feature, v_feature, l_feature

    def fusion_feature(self, a_feature, v_feature):

        a = torch.tensor(a_feature)
        v = torch.tensor(v_feature)
        f_feature = self.fusion(torch.cat((a, v), dim=1))
        return f_feature


    def classfier(self, x, hide_f, w, is_a=None):
        if is_a == 'a':
            result_a = self.embedding_a(x)
            # print(result_a.shape)
            # result = w * (result_a + hide_f)
            result = result_a + w * hide_f
            # result = result_a + 0.5 * hide_f
            # result = result_a
            r = torch.mean(result_a, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            i = 0
            layerlen = len(self.additional_layers_a)
            # new_x = torch.tensor(x)
            for layer in self.additional_layers_a:
                addf = self.relu(layer(x))
                addf = addf + hide_f
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                i=i+1
                if i < layerlen:
                    o_fea = feature
        elif is_a == 'v':
            result_v = self.embedding_v(x)
            # print(result_v.shape)
            # result = w * (result_v + hide_f)
            result = result_v + w * hide_f
            # result = result_v + 0.5 * hide_f
            # result = result_v
            r = torch.mean(result_v, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            j = 0
            layerlen = len(self.additional_layers_v)
            # new_x = torch.tensor(x)
            for layer in self.additional_layers_v:
                addf = self.relu(layer(x))
                addf = addf + hide_f
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                j=j+1
                if j < layerlen:
                    o_fea = feature

        return feature, result, o_fea, add_fea

    def forward_feature(self, x, w, is_a=True):
        if is_a:
            result_a = self.embedding_a(x)
            r = result_a
            i = 0
            for layer in self.additional_layers_a:
                addf = self.relu(layer(x))
                # r += w * addf
                r = torch.cat((r, w * addf), dim=1)
                i=i+1
        else:
            result_v = self.embedding_v(x)
            r = result_v
            j = 0
            for layer in self.additional_layers_v:
                addf = self.relu(layer(x))
                # r += w * addf
                r = torch.cat((r, w * addf), dim=1)
                j=j+1
        return r

