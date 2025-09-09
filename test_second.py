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

plt.switch_backend('agg')
from torch.utils.data import DataLoader

# from Source.data_func import RegistrationDataSet

# sys.path.append('../Source')

sys.path.append('../Source')
from data_func import load_list, RegistrationDataSet
from record_func import Plotter
from image_func import intensity_scale
from help_func import read_yaml, make_dir, tensor2array
from model_func import load_model1, load_model2, load_model3
from eval_func import seg_eval, sim_eval, pear_eval


def main(path):
    cfg = read_yaml(path)

    device = torch.device('cuda:{}'.format(cfg['GPUId']))

    torch.manual_seed(cfg['Seed'])
    random.seed(cfg['Seed'])

    _, _, test_list = load_list(cfg['TextPath'], fixed_time=cfg['FixedTimePoint'], moving_time=cfg['MovingTimePoint'])
    test_set = RegistrationDataSet(test_list, cfg['DataPath'])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    model1 = load_model1(cfg).to(device)
    model = load_model2(cfg, model1=model1).to(device)

    model_path = os.path.join(cfg['CheckpointsPath'], 'Weights', 'ML', 'clap', 'model_sn_c2_last.pth.gz')
    model.load_weight(model_path)
    model.eval()

    total_params = sum([param.nelement() for param in model.parameters()])
    Up = torch.nn.Upsample(scale_factor=2, mode='bilinear')
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

        subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results')
        make_dir(subj_save_path)
        start_time = time.time()

        moving_img = subj_id['Moving']['IMG']
        fixed = subj_id['Fixed']['IMG']

        # moving_seg = cv2.imread(os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['MovingTimePoint'])),
        #                         cv2.IMREAD_GRAYSCALE)

        # moving_img = intensity_scale(moving_img)[np.newaxis, np.newaxis, ...]
        # fixed_img = intensity_scale(fixed_img)[np.newaxis, np.newaxis, ...]

        # moving_img = torch.FloatTensor(moving_img).unsqueeze(0)
        # # moving_seg = torch.FloatTensor(moving_seg[np.newaxis, np.newaxis, ...])
        # fixed_img = torch.FloatTensor(fixed_img).unsqueeze(0)

        fwt = model(moving_img.to(model.model_device), fixed.to(model.model_device))

        time_list.append(time.time() - start_time)

        moved_img = fwt['Moved']

        moved_img = tensor2array(Up(moved_img), True)
        flow = tensor2array(fwt['Flow'], True)
        fixed = tensor2array(fixed, True)
        # fixed_img = tensor2array(fwt['Fixed'], True)
        moving = tensor2array(fwt['Moving'], True)

        sim_dict = sim_eval(moved_img, fixed)
        pcc_dict = pear_eval(moved_img, fixed)

        mse_list.append(sim_dict['MSE'])
        ssim_list.append(sim_dict['SSIM'])
        pcc_list.append(pcc_dict)

        # moved_img = np.uint8(moved_img * 255)  
        moved_img = np.uint8(moved_img)

        # fixed = np.uint8(fixed_img * 255)
        # fixed = np.uint8(fixed_img)
        # moving = np.uint8(moving * 255)
        moving = np.uint8(moving)

        slice_save_path = os.path.join(subj_save_path)
        os.makedirs(slice_save_path, exist_ok=True)

        # flow = np.transpose(flow, axes=(1, 2, 0))
        # z_flow = np.zeros_like(flow[..., 0])[..., np.newaxis]
        # flow = np.concatenate([flow, z_flow], axis=-1)
        # plt.imshow(flow, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # cv2.imwrite(os.path.join(slice_save_path, 'ML', 'flow', 'flow_{}.png'.format(batch_idx)), flow)
        # cv2.imwrite(os.path.join(slice_save_path, 'ML', 'ali', 'MovedIMG_{}.png'.format(batch_idx)), moved_img)
        # cv2.imwrite(os.path.join(slice_save_path, 'ML', 'OUR', 'FixedIMG_{}.png'.format(batch_idx)), fixed)
        # cv2.imwrite(os.path.join(slice_save_path, 'ML', 'OUR', 'MovingIMG_{}.png'.format(batch_idx)), moving)

        # plt.imshow(flow)
        # plt.axis('off')
        # plt.show()

        # moved_seg = model.stn(batch['Moving']['SEG'].unsqueeze(0).to(model.model_device), fwd['Flow'],
        # mode='nearest')  # 确实是形变相关的，但是不想要分割

        # fixed_seg = tensor2array(batch['Fixed']['SEG'], True)

        # seg_dict = seg_eval(moved_seg, fixed_seg)

    print('MSE_Score:', mse_list, 'mean_mse:', np.mean(mse_list), '******', np.std(mse_list))
    print("SSIM_Score:", ssim_list, 'mean_ssim:', np.mean(ssim_list), '******', np.std(ssim_list))
    print("PCC_Score:", pcc_list, 'mean_pcc:', np.mean(pcc_list), '******', np.std(pcc_list))
    print(np.mean(time_list))
    print(np.std(time_list))
    print("Parameters: %.2fM" % (total_params / 1e6))


if __name__ == '__main__':
    # cfg_path = '../Config/IRLBO.yaml'
    cfg_path = '../Config/IRLBO.yaml'
    main(cfg_path)
