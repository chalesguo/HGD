import os
import time
import random
import shutil

import cv2
import torch
import nibabel as nib
import numpy as np
import sys
import neurite as ne

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Source.data_func import RegistrationDataSet

sys.path.append('../Source')
from data_func import load_list
from record_func import Plotter
from image_func import intensity_scale
from help_func import read_yaml, make_dir, tensor2array
from model_func import load_model, load_model1
from eval_func import seg_eval, sim_eval, pear_eval


def main(path):
    cfg = read_yaml(path)

    device = torch.device('cuda:{}'.format(cfg['GPUId']))

    torch.manual_seed(cfg['Seed'])
    random.seed(cfg['Seed'])

    _, _, test_list = load_list(cfg['TextPath'], fixed_time=cfg['FixedTimePoint'], moving_time=cfg['MovingTimePoint'])
    test_set = RegistrationDataSet(test_list, cfg['DataPath'])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    if os.path.exists(os.path.join(cfg['CheckpointsPath'], 'Logs', 'log.npy')):
        plotter = Plotter(os.path.join(cfg['CheckpointsPath'], 'Logs'))
        plotter.buffer = np.load(os.path.join(cfg['CheckpointsPath'], 'Logs', 'log.npy'), allow_pickle=True)[()]
        plotter.send()

    time_list = []
    mse_list = []
    ssim_list = []
    pcc_list = []

    for batch_idx, (subj_id) in enumerate(test_loader):
        print('testing...', batch_idx)

        # subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results', subj_id)
        subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results')
        make_dir(subj_save_path)
        start_time = time.time()
        # for slice_id in slice_ids:

        moving_img = subj_id['Moving']['IMG']
        fixed_img = subj_id['Fixed']['IMG']


        # moved_seg = tensor2array(moved_seg, True)
        moving_img = tensor2array(moving_img, True)
        fixed = tensor2array(fixed_img, True)

        # print(fixed_img.size, moved_img.size)
        sim_dict = sim_eval(moving_img, fixed)
        pcc_dict = pear_eval(moving_img, fixed)
        mse_list.append(sim_dict['MSE'])
        ssim_list.append(sim_dict['SSIM'])
        pcc_list.append(pcc_dict)

    print('MSE_Score:', mse_list, 'mean_mse:', np.mean(mse_list))
    print("SSIM_Score:", ssim_list, 'mean_ssim:', np.mean(ssim_list))
    print("PCC_Score:", pcc_list, 'mean_pcc:', np.mean(pcc_list))
    print(np.mean(time_list))
    print(np.std(time_list))


if __name__ == '__main__':
    cfg_path = '../Config/IRLBO.yaml'
    main(cfg_path)
