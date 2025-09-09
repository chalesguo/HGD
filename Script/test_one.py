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

    model = load_model(cfg).to(device)

    model.eval()

    total_params = sum([param.nelement() for param in model.parameters()])
    # load_path = os.path.join(cfg['CheckpointsPath'], 'Weights', 'model_oa_nu80_best.pth.gz')
    load_path = os.path.join(cfg['CheckpointsPath'], 'Weights', 'ML', 'n2', 'model_ml_nu_best.pth.gz')
    model.load_weight(load_path)

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

        # subj_path = os.path.join(cfg['DataPath'], subj_id['Moving']['IMG'])
        # slice_ids = os.listdir(subj_path)

        # temp_img = cv2.imread(subj_path, cv2.IMREAD_GRAYSCALE)

        # w, h = temp_img.shape
        #
        # if w != cfg['ImgSize'] or h != cfg['ImgSize']:
        #     continue

        # subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results', subj_id)
        subj_save_path = os.path.join(cfg['CheckpointsPath'], 'Results')
        make_dir(subj_save_path)

        # for slice_id in slice_ids:

        # moving_img = cv2.imread(os.path.join(cfg['DataPath'], subj_id['Moving']['IMG']), cv2.IMREAD_GRAYSCALE)
        # fixed_img = cv2.imread(os.path.join(cfg['DataPath'], subj_id['Fixed']['IMG']), cv2.IMREAD_GRAYSCALE)

        moving_img = subj_id['Moving']['IMG']
        fixed_img = subj_id['Fixed']['IMG']

        # moving_seg = cv2.imread(os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['MovingTimePoint'])),
        #                         cv2.IMREAD_GRAYSCALE)

        # moving_img = intensity_scale(moving_img)[np.newaxis, np.newaxis, ...]
        # fixed_img = intensity_scale(fixed_img)[np.newaxis, np.newaxis, ...]

        # moving_img = torch.FloatTensor(moving_img).unsqueeze(0)
        # # moving_seg = torch.FloatTensor(moving_seg[np.newaxis, np.newaxis, ...])
        # fixed_img = torch.FloatTensor(fixed_img).unsqueeze(0)
        start_time = time.time()
        fwt = model(moving_img.to(model.model_device), fixed_img.to(model.model_device))
        time_list.append(time.time() - start_time)
        moved_img = fwt['Moved']

        # moved_seg = model.stn(moving_seg.to(device), fwt['Flow'], mode='nearest')

        # moved_seg = tensor2array(moved_seg, True)
        moved_img = tensor2array(moved_img, True)
        fixed = tensor2array(fixed_img, True)

        # flow = tensor2array(fwt['Flow'], True)
        flow = fwt['Flow']
        fixed_img = tensor2array(fwt['Fixed'], True)
        # print(fixed_img.size, moved_img.size)
        sim_dict = sim_eval(moved_img, fixed_img)
        pcc_dict = pear_eval(moved_img, fixed_img)
        mse_list.append(sim_dict['MSE'])
        ssim_list.append(sim_dict['SSIM'])
        pcc_list.append(pcc_dict)

        # flow = flow.permute(1, 0, 2, 3)
        # flow1 = torch.squeeze(flow, dim=1)
        # flow1 = flow1.detach().cpu()
        # flow = np.transpose(flow1.numpy(), (1, 2, 0))
        # ne.plot.flow([flow])

        # flow = np.transpose(flow, axes=(1, 2, 0))
        # z_flow = np.zeros_like(flow[..., 0])[..., np.newaxis]
        # flow = np.concatenate([flow, z_flow], axis=-1)
        # plt.imshow(flow, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # moved_img = np.uint8(moved_img * 255)
        moved_img = np.uint8(moved_img)
        # fixed = np.uint8(fixed * 255)
        fixed = np.uint8(fixed)

        slice_save_path = os.path.join(subj_save_path)
        os.makedirs(slice_save_path, exist_ok=True)

        # shutil.copy(src=os.path.join(cfg['DataPath'], subj_id['Moving']['IMG']),
        #             dst=os.path.join(slice_save_path, 'MovingIMG.png'))
        # shutil.copy(src=os.path.join(cfg['DataPath'], subj_id['Fixed']['IMG']),
        #             dst=os.path.join(slice_save_path, 'FixedIMG.png'))
        # shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['MovingTimePoint'])),
        #             dst=os.path.join(slice_save_path, 'MovingSEG.png'))
        # shutil.copy(src=os.path.join(subj_path, str(slice_id), '{}_SEG.png'.format(cfg['FixedTimePoint'])),
        #             dst=os.path.join(slice_save_path, 'FixedSEG.png'))

        # cv2.imwrite(os.path.join(slice_save_path, 'ML', 'flow', 'flow_{}.png'.format(batch_idx)), flow)
        # cv2.imwrite(os.path.join(slice_save_path, 'MovedIMG_{}.png'.format(batch_idx)), moved_img)
        # cv2.imwrite(os.path.join(slice_save_path, 'FixedIMG_{}.png'.format(batch_idx)), fixed)
        # cv2.imwrite(os.path.join(slice_save_path, 'MovedSEG.png'), moved_seg)
        # flow = np.transpose(flow, axes=(1, 2, 0))
        # z_flow = np.zeros_like(flow[..., 0])[..., np.newaxis]
        # flow = np.concatenate([flow, z_flow], axis=-1)
        # flow_nii = nib.Nifti1Image(flow[:, :, np.newaxis, :], affine=np.eye(4))
        # nib.save(flow_nii, os.path.join(slice_save_path, 'Deformation.nii'))
        # time_list.append(time.time() - start_time)

    print('MSE_Score:', mse_list, 'mean_mse:', np.mean(mse_list))
    print("SSIM_Score:", ssim_list, 'mean_ssim:', np.mean(ssim_list))
    print("PCC_Score:", pcc_list, 'mean_pcc:', np.mean(pcc_list))
    print(np.mean(time_list))
    print(np.std(time_list))
    print("Parameters: %.2fM" % (total_params / 1e6))


if __name__ == '__main__':
    cfg_path = '../Config/IRLBO.yaml'
    main(cfg_path)
