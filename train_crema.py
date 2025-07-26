#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
import re
from collections import defaultdict
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.Samplers import ClassAwareSampler
from dataset.ClassPrioritySampler import ClassPrioritySampler
from dataset.CREMA import CramedDataset
from dataset.KS import VADataset
from dataset.nvGesture import NvGestureDataset
from dataset.VGGSoundDataset import VGGSound
from dataset import create_dataset
from model.AudioVideo import  AVGBShareClassifierLearnemb, AVGBShareClassifier_newModal, AVGBShareClassifier_attention
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
import csv
import time





def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def train_audio_video(epoch, train_loader, model, optimizer, logger, gs_plugin=None, merge_alpha=0.5):
    model.train()
    global s_a
    global s_v
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()
    tv = Averager()
    da = Averager()
    dv = Averager()
    dav = Averager()
    d_inter = Averager()
    d_intra = Averager()
    w_a = Averager()
    w_v = Averager()
    len_dataloader = len(train_loader)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    score_v = 0.0
    score_a = 0.0

    for step, (spectrogram, image, y) in enumerate(train_loader):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        # print(y)
        # print(y.shape)
        S_y = y @ y.T

        with torch.no_grad():
            w_ = torch.Tensor([[float(s_a)], [float(s_v)]]).cuda()
            w = model.rein_network1(w_.unsqueeze(0))


        o_a, o_v, o_f = model(spectrogram, image)
        # print(type(o_a),type(o_v),type(o_f))
        e_a = exp_map_poincare(o_a)
        e_v = exp_map_poincare(o_v)
        e_f = exp_map_poincare(o_f)
        d_af = compute_pfc_loss(e_a, e_f)
        d_vf = compute_pfc_loss(e_v, e_f)
        d_av = triplet_cosine_loss(o_a, o_v)
        # print(o_a.shape, o_v.shape, o_f.shape)
        # d_af = torch.sum(torch.nn.functional.cosine_similarity(o_a, o_f))
        # d_vf = torch.sum(torch.nn.functional.cosine_similarity(o_v, o_f))
        # d_af = torch.norm(o_a - o_f)
        # d_vf = torch.norm(o_v - o_f)
        # print(f"distance of av is {d_af}, and distance of vf is {d_vf}")

        hide_f = model.embedding_l(o_f)

        # audio
        out_a, r_a, o_fea, add_fea = model.classfier(o_a, hide_f, w[:, 0][0], is_a='a')

        if add_fea is None:
            loss_a = criterion(out_a, y).mean()
        else:
            kl = y*o_fea.detach().softmax(1)
            loss_a = 1.0*criterion(out_a, y).mean() + 1.0*criterion(o_fea, y).mean() + 1.0*criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()
            # loss_a = criterion(out_a, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y-0.2*kl).mean()


        gs_plugin.exp_count += 1

        # video
        out_v, r_v, o_fea, add_fea = model.classfier(o_v, hide_f, w[:, 1][0], is_a='v')
        if add_fea is None:
            loss_v = criterion(out_v, y).mean()
        else:
            kl = y*o_fea.detach().softmax(1)
            loss_v = 1.0*criterion(out_v, y).mean() + 1.0*criterion(o_fea, y).mean() + 1.0*criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()
            # loss_v = criterion(out_v, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y-0.2*kl).mean()

        inter = sim_loss(r_a, r_v, S_y) + sim_loss(r_v, r_a, S_y)
        intra = sim_loss(r_a, r_a, S_y) + sim_loss(r_v, r_v, S_y)

        gs_plugin.exp_count += 1
        if epoch + 1 <=70:
            loss = loss_a * merge_alpha + loss_v * (1 - merge_alpha) + 0.1 * (d_af + d_vf) + 0.01 * (intra + inter)
        else:
            loss = loss_a * merge_alpha + loss_v * (1 - merge_alpha) + 0.1 * (d_af + d_vf) + 0.0015 * (d_av) + 0.01 * (intra + inter)
        # loss = loss_a * merge_alpha + loss_v * (1 - merge_alpha)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tl.add(loss.item())
        ta.add(loss_a.item())
        tv.add(loss_v.item())
        da.add(d_af.item())
        dv.add(d_vf.item())
        dav.add(d_av.item())
        d_inter.add(inter.item())
        d_intra.add(inter.item())

        # print(torch.argmax(y[0]))
        # input()
        tmp_v = sum([F.softmax(out_v)[i][torch.argmax(y[i])] for i in range(out_v.size(0))])
        tmp_a = sum([F.softmax(out_a)[i][torch.argmax(y[i])] for i in range(out_a.size(0))])

        score_v += tmp_v
        score_a += tmp_a

        s_a = (tmp_a * (1.0 / (step + 1)) + score_a * (step / (step + 1))) / (tmp_v * (1.0 / (step + 1)) + score_v * (step / (step + 1)))
        s_v = 1 / s_a
        # s_a = tmp_a * (1.0 / (step + 1)) + score_a * (step / (step + 1))
        # s_v = tmp_v * (1.0 / (step + 1)) + score_v * (step / (step + 1))
        w = model.rein_network1(w_.unsqueeze(0))

        w_a.add(w[:, 0][0])
        w_v.add(w[:, 1][0])
        ## for cremaD
        criterion2 = nn.MSELoss().cuda()
        if s_a > 1.0:
            a_list = [0.2] * 1
            v_list = [0.8] * 1
            w_label = torch.Tensor([a_list,v_list]).unsqueeze(0).cuda()
            loss_w = criterion2(w, w_label)
        else:
            v_list = [0.2] * 1
            a_list = [0.8] * 1
            w_label = torch.Tensor([a_list, v_list]).unsqueeze(0).cuda()
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
                'Epoch:{epoch}, Trainnig Loss:{train_loss:.3f}, Training Loss_a:{loss_a:.3f}, Training Loss_v:{loss_v:.3f}').format(
                epoch=epoch, train_loss=loss.item(), loss_a=loss_a.item(), loss_v=loss_v.item()))

    ratio_a = score_a / score_v
    loss_ave = tl.item()
    loss_a_ave = ta.item()
    loss_v_ave = tv.item()
    dis_af = da.item()
    dis_vf = dv.item()
    dis_av = dav.item()
    inter_sim = d_inter.item()
    intra_sim = d_intra.item()


    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Loss_a:{loss_a_ave:.2f}, Average Training Loss_v:{loss_v_ave:.2f}, Average dis_af:{dis_a:.2f},Average dis_vf:{dis_v:.2f},Average dis_av:{dis_av:.2f},inter_sim:{inter_sim:.2f}, intra_sim:{intra_sim:.2f} ').format(
        epoch=epoch, loss_ave=loss_ave, loss_a_ave=loss_a_ave, loss_v_ave=loss_v_ave, dis_a=dis_af, dis_v=dis_vf, dis_av=dis_av, inter_sim=inter_sim, intra_sim=intra_sim))

    return model, ratio_a, score_a, score_v, w_a, w_v


def val(epoch, val_loader, model, logger, merge_alpha=0.5):
    global s_a
    global s_v
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []

    score_a = 0.0
    score_v = 0.0
    feature_audio_array = None
    feature_visual_array = None
    label_array = None
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            w_ = torch.Tensor([[float(s_a)], [float(s_v)]]).cuda()
            w = model.rein_network1(w_.unsqueeze(0))
            torch.save(w, "w_crema_GB.pt")
            o_a, o_v, o_f = model(spectrogram, image)
            hide_f = model.embedding_l(o_f)
            out_a, result_a, o_fea_a, _ = model.classfier(o_a, hide_f, w[:, 0][0], is_a='a')
            out_v, result_v, o_fea_v, _ = model.classfier(o_v, hide_f, w[:, 1][0], is_a='v')

            a_numpy = result_a.cpu().detach().numpy()
            v_numpy = result_v.cpu().detach().numpy()
            label_numpy = y.cpu().detach().numpy()


            if feature_audio_array is None:
                feature_audio_array = a_numpy
                feature_visual_array = v_numpy
                label_array = label_numpy
            else:
                # 拼接到 NumPy 数组
                feature_audio_array = np.vstack((feature_audio_array, a_numpy))
                feature_visual_array = np.vstack((feature_visual_array, v_numpy))
                label_array = np.hstack((label_array, label_numpy))

            tmp_v = sum([F.softmax(out_v)[i][torch.argmax(y[i])] for i in range(out_v.size(0))])
            tmp_a = sum([F.softmax(out_a)[i][torch.argmax(y[i])] for i in range(out_a.size(0))])

            score_a+=tmp_a
            score_v+=tmp_v
            # out = merge_alpha * out_a + (1-merge_alpha) * out_v
            out = 0.4 * out_a + 0.6 * out_v
            # criterion(out_a, y)
            soft_pred_a = soft_pred_a + (F.softmax(out_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(out_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(out, dim=1)).tolist()
            pred = (F.softmax(out, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(out_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(out_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()


        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))


    np.save(f'./npy/cremad_audio.npy', feature_audio_array)
    np.save(f'./npy/cremad_video.npy', feature_visual_array)
    np.save(f'./npy/cremad_label.npy', label_array)

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                       f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    logger.info(('s_a:{s_a:.3f}, s_v:{s_v:.3f}, w:{w}').format(s_a=s_a, s_v=s_v, w=w))
    return acc, acc_a, acc_v

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='/data/wfq/sht/MML/project1/data/crema.json')
    parser.add_argument('--lam', default=1.0, type=float, help='lam')
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
    tensorboard_save_path = './tensorboard'
    writer = SummaryWriter(log_dir=f'{tensorboard_save_path}')


    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')
    # train_dataset = VADataset(config, mode='train')
    # test_dataset = VADataset(config, mode='test')
    # train_dataset = VGGSound(config, mode='train')
    # test_dataset = VGGSound(config, mode='test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)
    global s_a
    global s_v
    s_a = 0.0
    s_v = 0.0
    # ----- MODEL -----
    model = AVGBShareClassifier_newModal(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    gs = GSPlugin()
    lr_adjust = config['train']['optimizer']['lr']


    rein_param_list = list(map(id, model.rein_network1.parameters()))
    rein_params = filter(lambda p: id(p) in rein_param_list, model.parameters())
    base_params = filter(lambda p: id(p) not in rein_param_list, model.parameters())
    params = [{'params': base_params, 'lr': lr_adjust},
              {'params': rein_params, 'lr': lr_adjust / 10}
              ]
    optimizer = optim.SGD(params, lr=lr_adjust, momentum=config['train']['optimizer']['momentum'], weight_decay=config['train']['optimizer']['wc'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0

    if cfg['train']['epoch_dict'] > 10:
        check = int(cfg['train']['epoch_dict'] / 10)
    else:
        check = 1
    logger.info(('seed 1 not new'))

    acc_a_list = []
    acc_v_list = []
    acc_list = []
    w_a_list = []
    w_v_list = []
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model, ratio_a, t_a, t_v, w_a, w_v = train_audio_video(epoch, train_loader, model, optimizer, logger, gs, args.merge_alpha)
        w_v_list.append(w_v.item().cpu().detach().numpy())
        w_a_list.append(w_a.item().cpu().detach().numpy())
        writer.add_scalar('weight_audio', w_a.item().cpu().detach().numpy(), epoch)
        writer.add_scalar('weight_video', w_v.item().cpu().detach().numpy(), epoch)

        acc, acc_a, acc_v = val(epoch, test_loader, model, logger, args.merge_alpha)
        acc_list.append(acc)
        acc_a_list.append(acc_a)
        acc_v_list.append(acc_v)

        logger.info(('ratio: {ratio_a:.3f}').format(ratio_a=ratio_a))
        if acc >= best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            logger.info('Find a better model and save it!')
            m_name = cfg['visual']['name'] + '_' + cfg['text']['name']
            # torch.save(model.state_dict(), '/data/sht/MultiModel_imbalance/project1/checkpoint/crema_LCR_best_model.pth')
        torch.cuda.empty_cache()

