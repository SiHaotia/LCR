from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict



class VTGBShareModel_newModal(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config['text']["name"] == 'bert-base':
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if config['visual']["name"] == 'resnet50':
            self.visual_encoder = torchvision.models.resnet50()
            checkpoint = torch.load('your pretrained vision pth file')
            self.visual_encoder.load_state_dict(checkpoint)

        self.learned_encoder = torchvision.models.resnet50()
        checkpoint = torch.load(('your pretrained vision pth file'))
        self.visual_encoder.load_state_dict(checkpoint)

        self.learned_modal = nn.Parameter(torch.randn(1, 3, 224, 224), requires_grad=True)

        self.i2t = nn.Linear(self.visual_encoder.fc.out_features, self.text_encoder.config.hidden_size)
        self.l2t = nn.Linear(self.learned_encoder.fc.out_features, self.text_encoder.config.hidden_size)

        self.embedding_t = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 256),
            nn.ReLU(),
        )

        self.embedding_i = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 256),
            nn.ReLU(),
        )

        self.embedding_l = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 256),
            nn.ReLU(),
        )
        self.num_class = config['setting']['num_class']

        self.fc_out = nn.Linear(256, self.num_class)
        
        self.relu = nn.ReLU()
        self.rein_network1 = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, image, text):
        # text_input = self.tokenizer(text, padding='longest', max_length=30, return_tensors="pt")
        image_embeds = self.visual_encoder(image)
        image_embeds = self.i2t(image_embeds)
        text_embeds = self.text_encoder(text.input_ids,
                                             attention_mask=text.attention_mask,
                                             return_dict=True
                                             ).last_hidden_state[:,0,:]
        learned_embeds = self.learned_encoder(self.learned_modal)
        learned_embeds = self.l2t(learned_embeds)

        return image_embeds, text_embeds, learned_embeds

    

    def classfier(self, x, hide_l, w, is_i=True):
        if is_i:
            result_i = self.embedding_i(x)
            # result = result_i + w * hide_l
            result = result_i + w * hide_l
            r = torch.mean(result, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            i = 0
            layerlen = len(self.additional_layers_i)
            for layer in self.additional_layers_i:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                i = i + 1
                if i < layerlen:
                    o_fea = feature
        else:
            result_t = self.embedding_t(x)
            # result = result_t + w * hide_l
            result = result_t + w * hide_l
            r = torch.mean(result, 0, True)
            feature = self.fc_out(result)
            o_fea = feature
            add_fea = None
            j = 0
            layerlen = len(self.additional_layers_t)
            for layer in self.additional_layers_t:
                addf = self.relu(layer(x))
                r += torch.mean(addf, 0, True)
                add_fea = w * self.fc_out(addf)
                feature = feature + add_fea
                j = j + 1
                if j < layerlen:
                    o_fea = feature
        return feature, r, o_fea, add_fea
