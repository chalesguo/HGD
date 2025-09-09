import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# from image_func import intensity_scale


# def load_list(text_path, fixed_time, moving_time):
#
#     train_list = []
#     val_list = []
#     test_list = []
#
#     train_subjects = open(os.path.join(text_path, "Train.txt"), 'r').readlines()
#     val_subjects = open(os.path.join(text_path, "Validation.txt"), 'r').readlines()
#     test_subjects = open(os.path.join(text_path, "Test.txt"), 'r').readlines()
#
#     for subj_list, subj_ids in zip([train_list, val_list], [train_subjects, val_subjects]):
#         for subject in subj_ids:
#             subject = subject.strip()
#             subject_dict = {
#                 "Moving": {'IMG': os.path.join(subject, f'{moving_time}_IMG.png')},
#                 "Fixed": {'IMG': os.path.join(subject, f'{fixed_time}_IMG.png')}
#             }
#             subj_list.append(subject_dict)
#
#     for subject in test_subjects:
#         test_list.append(subject.strip())
#
#     print(f"Train and val on {len(train_list)}, {len(val_list)} pairs, test with {len(test_list)} subjects")
#
#     return train_list, val_list, test_list

def list_data(data_subjects):
    list_ = []
    for subj_ids in data_subjects:
        subject = subj_ids.strip()
        subject_dict = {
            "Moving": {'IMG': subject.split(',')[0]},
            "Fixed": {'IMG': subject.split(',')[1]}
        }
        list_.append(subject_dict)
    return list_


def load_list(text_path, fixed_time, moving_time):
    train_subjects = open(os.path.join(text_path, "Train.txt"), 'r').readlines() 
    val_subjects = open(os.path.join(text_path, "Valid.txt"), 'r').readlines()
    test_subjects = open(os.path.join(text_path, "pic.txt"), 'r').readlines()  

    train_list = list_data(train_subjects)
    test_list = list_data(test_subjects)
    val_list = list_data(val_subjects)

    print(f"Train and val on {len(train_list)}, {len(val_list)} pairs, test with {len(test_list)} subjects")

    return train_list, val_list, test_list


class RegistrationDataSet(Dataset):
    def __init__(self, data_list, data_root):
        self.data_list = data_list
        self.data_root = data_root
        torch.manual_seed(3407)

    def __getitem__(self, item):
        data_item = self.data_list[item]

        moving_path = os.path.join(self.data_root, data_item['Moving']['IMG'])
        fixed_path  = os.path.join(self.data_root, data_item['Fixed']['IMG'])

        moving_img = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
        fixed_img  = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)

        if moving_img is None:
            raise FileNotFoundError(f"Failed to load image: {moving_path}")
        if fixed_img is None:
            raise FileNotFoundError(f"Failed to load image: {fixed_path}")

        moving_img = torch.from_numpy(np.expand_dims(moving_img, axis=0)).float()
        fixed_img  = torch.from_numpy(np.expand_dims(fixed_img, axis=0)).float()

        data_dict = {
            'Moving': {'IMG': moving_img},
            'Fixed':  {'IMG': fixed_img}
        }

        return data_dict

    def __len__(self):
        return len(self.data_list)


class RegistrationDataSet1(Dataset):
    def __init__(self, data_list, data_root):
        self.data_list = data_list
        self.data_root = data_root
        torch.manual_seed(3407)

    def __getitem__(self, item):
        data_item = self.data_list[item]

        moving_img = cv2.imread(os.path.join(self.data_root, data_item['Moving']['IMG']), cv2.IMREAD_GRAYSCALE)
        fixed_img = cv2.imread(os.path.join(self.data_root, data_item['Fixed']['IMG']), cv2.IMREAD_GRAYSCALE)

        moving_img = torch.FloatTensor(moving_img).unsqueeze(0)
        # moving_seg = torch.FloatTensor(moving_seg)

        fixed_img = torch.FloatTensor(fixed_img).unsqueeze(0)
        # fixed_seg = torch.FloatTensor(fixed_seg)

        data_dict = {
            # 'Moving': {'IMG': moving_img, 'SEG': moving_seg},
            # 'Fixed': {'IMG': fixed_img, 'SEG': fixed_seg}
            'Moving': {'IMG': moving_img},
            'Fixed': {'IMG': fixed_img}
        }

        return data_dict

    def __len__(self):
        return len(self.data_list)
