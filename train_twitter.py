#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
import re
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from transformers import BertTokenizer
from dataset import create_dataset
from model.VTModel import VTShareModel, VTGBShareModel_newModal
from utils.loss import mixup_criterion
from utils.utils import (
    create_logger,
    Averager,
    shot_acc,
    deep_update_dict,
    get_optimizer,
    get_scheduler,
    pre_compute_class_ratio,
    freeze_backbone,
    mixup_data,
    lr_reset,
    param_count,
    mixup_data_av
)
from utils.tools import GSPlugin, weight_init
from utils.loss import exp_map_poincare, compute_pfc_loss, triplet_cosine_loss, sim_loss

def train_text_image(epoch, train_loader, model, optimizer, logger, gs_plugin=None, merge_alpha=0.5):
    model.train()
    global s_i
    global s_t
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()
    tv = Averager()
    di = Averager()
    dt = Averager()
    dit = Averager()
    d_inter = Averager()
    d_intra = Averager()
    len_dataloader = len(train_loader)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    score_t = 0.0
    score_i = 0.0
    alpha_t = 0.0 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for step, (image, text, y) in enumerate(train_loader):
        image = image.cuda()
        y = y.cuda()
        S_y = y @ y.T

        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)

        optimizer.zero_grad()
        with torch.no_grad():
            w_ = torch.Tensor([[float(s_t)], [float(s_i)]]).cuda()
            w = model.rein_network1(w_.unsqueeze(0))


        o_i, o_t, o_l = model(image, text_input)
        e_i = exp_map_poincare(o_i)
        e_t = exp_map_poincare(o_t)
        e_l = exp_map_poincare(o_l)
        d_il = compute_pfc_loss(e_i, e_l)
        d_tl = compute_pfc_loss(e_t, e_l)
        d_it = triplet_cosine_loss(o_i, o_t)
        hide_l = model.embedding_l(o_l)
        out_i, r_i, o_fea, add_fea = model.classfier(o_i, hide_l, w[:, 0][0], is_i=True)
        if add_fea is None:
            loss_i = criterion(out_i, y).mean()
        else:
            kl = y * o_fea.detach().softmax(1)
            loss_i = criterion(out_i, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()



        out_t, r_t, o_fea, add_fea = model.classfier(o_t, hide_l, w[:, 1][0], is_i=False)
        if add_fea is None:
            loss_t = criterion(out_t, y).mean()
        else:
            kl = y * o_fea.detach().softmax(1)
            loss_t = criterion(out_t, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()

        inter = sim_loss(r_i, r_t, S_y) + sim_loss(r_t, r_i, S_y)
        intra = sim_loss(r_i, r_i, S_y) + sim_loss(r_t, r_t, S_y)

        if score_t>=score_i:
            alpha_t = 0.7
            loss = loss_i * merge_alpha + loss_t * (1 - merge_alpha) + alpha_t * d_tl + (1 - alpha_t) * d_il  + 0.001 * (d_it) + 0.01 * (inter + intra)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tl.add(loss.item())
        ta.add(loss_i.item())
        tv.add(loss_t.item())
        di.add(d_il.item())
        dt.add(d_tl.item())
        dit.add(d_it.item())
        d_inter.add(inter.item())
        d_intra.add(inter.item())

        tmp_v = sum([F.softmax(out_t)[i][torch.argmax(y[i])] for i in range(out_t.size(0))])
        tmp_a = sum([F.softmax(out_i)[i][torch.argmax(y[i])] for i in range(out_i.size(0))])
        score_t += tmp_v
        score_i += tmp_a
        s_i = (tmp_a * (1.0 / (step + 1)) + score_i * (step / (step + 1))) / (
                tmp_v * (1.0 / (step + 1)) + score_t * (step / (step + 1)))
        s_t = 1.0 / s_i
        w = model.rein_network1(w_.unsqueeze(0))
        criterion2 = nn.MSELoss()
        if s_i > 1.0:
            # a_list = [1]*max(len(model.additional_layers_a)-1, 0) + [0]*(min(10-len(model.additional_layers_a)+1, 10))
            # v_list = [1] * len(model.additional_layers_v) + [0] * (10 - len(model.additional_layers_v))
            i_list = [0.07] * 1
            t_list = [0.1] * 1
            w_label = torch.Tensor([i_list, t_list]).unsqueeze(0).cuda()
            loss_w = criterion2(w, w_label)
        else:
            t_list = [0.07] * 1
            i_list = [0.1] * 1
            w_label = torch.Tensor([i_list, t_list]).unsqueeze(0).cuda()
            loss_w = criterion2(w, w_label)
        loss_w = loss_w.mean()
        loss_w.backward()
        optimizer.step()
        optimizer.zero_grad()
        for n, p in model.named_parameters():
            if p.grad != None:
                del p.grad

        if step % cfg['print_inteval'] == 0:
            logger.info((
                'Epoch:{epoch}, Trainnig Loss:{train_loss:.3f}, Training Loss_a:{loss_a:.3f}, Training Loss_v:{loss_v:.3f}, distance_il:{dis_il:.2f}, distance_tl:{dis_tl:.2f}, ').format(
                epoch=epoch, train_loss=loss.item(), loss_a=loss_i.item(), loss_v=loss_t.item(), dis_il=d_il.item(), dis_tl=d_tl.item()))

    ratio_a = score_i / score_t
    loss_ave = tl.item()
    loss_a_ave = ta.item()
    loss_v_ave = tv.item()
    dis_il = di.item()
    dis_tl = dt.item()
    dis_it = dit.item()
    dis_inter = d_inter.item()
    dis_intra = d_intra.item()


    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info((
        'Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Loss_a:{loss_a_ave:.2f}, Average Training Loss_v:{loss_v_ave:.2f}, distance_il:{dis_il:.2f}, distance_tl:{dis_tl:.2f},Average dis_it:{dis_it:.2f},inter_sim:{inter_sim:.2f}, intra_sim:{intra_sim:.2f}').format(
        epoch=epoch, loss_ave=loss_ave, loss_a_ave=loss_a_ave, loss_v_ave=loss_v_ave, dis_il = dis_il, dis_tl = dis_tl,
        dis_it=dis_it, inter_sim=dis_inter, intra_sim=dis_intra))

    return model, ratio_a


def val(epoch, val_loader, model, logger, merge_alpha=0.5):
    global s_i
    global s_t
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []

    # criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for step, (image, text, y) in enumerate(val_loader):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            image = image.cuda()
            y = y.cuda()
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
            w_ = torch.Tensor([[float(s_i)], [float(s_t)]]).cuda()
            w = model.rein_network1(w_.unsqueeze(0))
            o_a, o_v, o_l = model(image, text_input)
            hide_l = model.embedding_l(o_l)
            out_a, _, _, _ = model.classfier(o_a, hide_l, w[:, 0][0], is_i=True)
            out_v, _, _, _ = model.classfier(o_v, hide_l, w[:, 1][0], is_i=False)
            out = merge_alpha * F.softmax(out_a, dim=1) + (1 - merge_alpha) * F.softmax(out_v, dim=1)
            # out = merge_alpha * out_a + (1 - merge_alpha) * out_v
            soft_pred_a = soft_pred_a + (F.softmax(out_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(out_v, dim=1)).tolist()
            soft_pred = soft_pred + out.tolist()
            pred = out.argmax(dim=1)
            pred_a = (F.softmax(out_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(out_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct=np.sum(np.array(label_list)==np.array(pred_list))
        correct_a=np.sum(np.array(label_list)==np.array(pred_list_a))
        correct_v=np.sum(np.array(label_list)==np.array(pred_list_v))
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f}').format(epoch=epoch, f1=f1, acc=acc,
                                                                                                                                    f1_a=f1_a, acc_a=acc_a,
                                                                                                                                     f1_v=f1_v, acc_v=acc_v,))
    logger.info(('s_i:{s_a:.3f}, s_t:{s_v:.3f}, w:{w}').format(s_a=s_i, s_v=s_t, w=w))
    return acc

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='./config/twitter.json')
    parser.add_argument('--lam', default=1.0, type=float, help='GPU ids')
    parser.add_argument('--merge_alpha', default=0.4, type=float, help='2 modal fusion alpha in GS')

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset, test_dataset = create_dataset('twitter', config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    global s_i
    global s_t
    s_i = 0.0
    s_t = 0.0
    # ----- MODEL -----
    model = VTGBShareModel_newModal(config=cfg)
    model = model.cuda()

    gs = GSPlugin()
    lr_adjust = config['train']['optimizer']['lr']

    # optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
    #                       momentum=config['train']['optimizer']['momentum'],
    #                       weight_decay=config['train']['optimizer']['wc'])
    rein_param_list = list(map(id, model.rein_network1.parameters()))
    rein_params = filter(lambda p: id(p) in rein_param_list, model.parameters())
    visual_param_list = list(map(id, model.visual_encoder.parameters()))
    visual_params = filter(lambda p: id(p) in visual_param_list, model.parameters())
    text_param_list = list(map(id, model.text_encoder.parameters()))
    text_params = filter(lambda p: id(p) in text_param_list, model.parameters())
    base_params = filter(lambda p: id(p) not in rein_param_list and id(p) not in text_param_list and id(p) not in visual_param_list, model.parameters())
    params = [{'params': visual_params, 'lr': lr_adjust},
              {'params': text_params, 'lr': lr_adjust},
              {'params': base_params, 'lr': lr_adjust},
              {'params': rein_params, 'lr': lr_adjust / 10}
              ]
    optimizer = optim.Adam(params, lr=lr_adjust, weight_decay=config['train']['optimizer']['wc'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0

    if cfg['train']['epoch_dict'] > 10:
        check = int(cfg['train']['epoch_dict'] / 10)
    else:
        check = 1
        logger.info(('seed 48 统一权重'))
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model, ratio_a = train_text_image(epoch, train_loader, model, optimizer, logger, gs, args.merge_alpha)

        acc = val(epoch, test_loader, model, logger, args.merge_alpha)
        logger.info(('ratio: {ratio_i:.3f}').format(ratio_i=ratio_a))
        if acc > best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            logger.info('Find a better model and save it!')
            m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
            torch.save(model.state_dict(), 'twitter_best_model.pth')
