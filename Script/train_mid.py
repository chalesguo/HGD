"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""
import shutil
import sys
from torch.utils.tensorboard.writer import SummaryWriter

from torch.utils.data import DataLoader
from Source.eval_func import seg_eval, sim_eval
from Source.model_func import load_model1, GradientDescent, load_model2
from Source.help_func import read_yaml, make_dir, tensor2array, seed_torch
from Source.record_func import Recorder
from Source.data_func import load_list, RegistrationDataSet

import os
import glob
import random
import matplotlib.pyplot as plt
import warnings
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
from torch.optim import Adam
from torchvision import transforms
import csv
import torch.nn.functional as F


def compute_folding_ratio(flow):
    """
    Robust 2D Jacobian folding ratio calculation (batch average, return percentage)
    Input:
        flow: Tensor, shape (B, 2, H, W), (dx, dy) format
    Return:
        fold_ratio_percent: float, average folding ratio (%) within batch
    """
    assert flow.dim() == 4 and flow.size(1) == 2, "flow must be Bx2xHxW"

    device = flow.device
    dtype = flow.dtype

    kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=dtype, device=device)
    ky = kx.permute(0, 1, 3, 2)

    ux = flow[:, 0:1, :, :]
    uy = flow[:, 1:2, :, :]

    ux_x = F.conv2d(F.pad(ux, (1, 1, 0, 0), mode='replicate'), kx)
    uy_x = F.conv2d(F.pad(uy, (1, 1, 0, 0), mode='replicate'), kx)

    ux_y = F.conv2d(F.pad(ux, (0, 0, 1, 1), mode='replicate'), ky)
    uy_y = F.conv2d(F.pad(uy, (0, 0, 1, 1), mode='replicate'), ky)

    J = (1.0 + ux_x) * (1.0 + uy_y) - ux_y * uy_x

    neg = (J < 0).float()
    fold_ratio_per_sample = neg.view(neg.size(0), -1).mean(dim=1)

    fold_ratio_percent = float(fold_ratio_per_sample.mean().item() * 100.0)
    return fold_ratio_percent


def train(updater, train_loader, epoch, iter, device, save_path=None):
    running_loss = 0.
    total_batches = len(train_loader)

    sum_similarity = 0.0
    sum_smoothness = 0.0
    sum_overlap = 0.0
    sum_folding = 0.0

    for batch_idx, batch in enumerate(train_loader, 1):
        fwd = updater.model(batch['Moving']['IMG'].to(updater.device),
                            batch['Fixed']['IMG'].to(updater.device))
        loss_total, loss_info, print_info = updater.update_gradient(fwd)
        running_loss += loss_total

        sum_similarity += loss_info['Similarity']
        sum_smoothness += loss_info['Smoothness']
        sum_overlap += loss_info['OverlapRatio']
        sum_folding += compute_folding_ratio(fwd['Flow'])

    avg_similarity = sum_similarity / total_batches
    avg_smoothness = sum_smoothness / total_batches
    avg_overlap = sum_overlap / total_batches
    avg_folding = sum_folding / total_batches

    running_loss /= total_batches

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_exists = os.path.exists(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'Similarity', 'Smoothness', 'Overlap', 'FoldingRatio'])
            writer.writerow([epoch, avg_similarity, avg_smoothness, avg_overlap, avg_folding])

    print('[Epoch {:3d}] Loss: {:.4f} Similarity: {:.4f} Smoothness: {:.4f} Overlap: {:.4f} Folding: {:.2f}%'.format(
        epoch, running_loss, avg_similarity, avg_smoothness, avg_overlap, avg_folding
    ))

    return running_loss


def val(model, val_loader, epoch):
    running_seg = 0.
    running_sim = 0.
    recorder = Recorder()

    for batch_idx, batch in enumerate(val_loader):
        eval_dict = {}

        fwd = model(batch['Moving']['IMG'].to(model.model_device), batch['Fixed']['IMG'].to(model.model_device))

        moved_img = tensor2array(fwd['Moved'], True)
        fixed_img = tensor2array(fwd['Fixed'], True)

        sim_dict = sim_eval(moved_img, fixed_img)
        eval_dict.update(sim_dict)

        running_sim += sim_dict['MSE']
        recorder.update(eval_dict)

    running_seg /= len(val_loader)
    running_sim /= len(val_loader)

    return recorder.info(), running_sim


def main(path):
    cfg = read_yaml(path)
    device = torch.device('cuda:{}'.format(cfg['GPUId']))

    seed_torch(cfg['Seed'])
    writer = SummaryWriter(os.path.join(cfg['CheckpointsPath'], 'Logs', 'OA', '01m'))

    train_list, val_list, _ = load_list(cfg['TextPath'],
                                        fixed_time=cfg['FixedTimePoint'], moving_time=cfg['MovingTimePoint'])
    make_dir(cfg['CheckpointsPath'])
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Logs'))
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Weights'))
    make_dir(os.path.join(cfg['CheckpointsPath'], 'Results'))
    shutil.copy(path, os.path.join(cfg['CheckpointsPath'], 'Logs', 'Configuration.yaml'))

    model1 = load_model1(cfg).to(device)
    model1_path = os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_oa_low_best.pth.gz')
    model1.load_weight(model1_path)

    model = load_model2(cfg, model1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LearningRate'], weight_decay=cfg['WeightDecay'])
    gradient_descent = GradientDescent(model, optimizer,
                                       sim_loss=cfg['SimLoss'], reg_loss=cfg['RegLoss'], loss_weight=cfg['LossWeight'],
                                       gradient_surgery=cfg['GradientSurgery'])
    if cfg['InitWeight']:
        model.load_weight(cfg['InitWeight'])
        print('>>> Load from', cfg['InitWeight'])

    train_set = RegistrationDataSet(train_list, cfg['DataPath'])
    val_set = RegistrationDataSet(val_list, cfg['DataPath'])

    train_loader = DataLoader(train_set, batch_size=cfg['BatchSize'], num_workers=cfg['NumWorkers'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    loss_list = []
    best_metric = 0.
    best_epoch = 0
    best_sim_metric = 1000.

    for epoch in range(cfg['StartEpoch'], cfg['NumEpoch']+cfg['StartEpoch']):
        model.train()
        log_file = os.path.join(cfg['CheckpointsPath'], 'Logs', 'train_history_mid.csv')
        running_loss = train(gradient_descent, train_loader, epoch, cfg['iter'], device, save_path=log_file)
        print('now Epoch {} >>> Loss: {:.4f}'.format(epoch, running_loss))

        writer.add_scalar("total_loss", running_loss, epoch)
        loss_list.append(running_loss)
        print("all loss are:", loss_list)

        if epoch % cfg['ValFreq'] == 0:
            model.eval()
            running_info, running_metric = val(model, val_loader, epoch)

            print('Epoch {} >>> Loss: {:.4f} Metric: {} Best Epoch: {:03d} with Best Sim_Metric: {:.5f}'.format(
                epoch, running_loss, running_info, best_epoch, best_sim_metric))
            open(os.path.join(cfg['CheckpointsPath'], 'Logs', 'record.txt'), 'a+').write(f'{epoch}-{running_loss}-{running_info}\n')

            if running_metric <= best_sim_metric:
                torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_oa_mid_best.pth.gz'))
                best_sim_metric = running_metric
                best_epoch = epoch

                print("Now best epoch  {} =====> MSE:{:.4f}".format(epoch, best_sim_metric))

        if cfg['EfficientSave']:
            torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_oa_mid_last.pth.gz'))
        else:
            torch.save(model.state_dict(), os.path.join(cfg['CheckpointsPath'], 'Weights', f'model_oa_mid_last_{epoch}.pth.gz'))

    # Optional plotting code is commented out
    # x = range(cfg['StartEpoch'], cfg['NumEpoch']+cfg['StartEpoch'])
    # y = loss_list
    # plt.plot(x, y, '.-')
    # plt.xlabel('Test loss')
    # plt.ylabel('Test loss')
    # plt.savefig(os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_oa_mid.png'))
    # plt.show()


if __name__ == "__main__":
    cfg_path = "./Config/Config.yaml"
    main(cfg_path)
