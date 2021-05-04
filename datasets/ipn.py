import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
from datasets.preprocess_ipn import preprocess_ipn_dataset
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
from pathlib import Path

from utils import load_value_file
import pdb


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality in ['RGB', 'flo']:
                return img.convert('RGB')
            elif modality in ['Depth', 'seg']:
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    return pil_loader
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader
    # else:
    #     return pil_loader


def video_loader(video_dir_path, frame_indices, clip_positions_dict, modality, sample_duration, 
                image_loader, use_preprocessing=False):

    video = []
    # contains image patch's position info if use_preprocessing
    # in the format of ((top-left corner), width, height)
    video_meta = []
    if modality in ['RGB', 'flo', 'seg']:
        for i in frame_indices:
            file_name = '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i)
            image_path = os.path.join(video_dir_path, file_name)
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
                if modality == "seg" and use_preprocessing:
                    video_meta.append(clip_positions_dict[file_name])
            else:
                print(image_path, "------- Does not exist")
                return video, video_meta
    elif modality in ['RGB-flo', 'RGB-seg']:
        for i in frame_indices: 
            # index 35 is used to change img to flow
            # seg1CM42_21_R_#156_000076

            file_name = '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i)
            image_path = os.path.join(video_dir_path, file_name)

            if modality.split('-')[1] == 'flo':
                sensor = 'flow'
            elif modality.split('-')[1] == 'seg':
                sensor = 'segment'
            image_path_depth = os.path.join(video_dir_path.replace('frames',sensor), file_name)
            
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')
            if modality == "RGB-seg" and use_preprocessing:
                video_meta.append(clip_positions_dict[file_name])

            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video, video_meta

    # video_meta = [((X, Y), width, height)]
    return video, video_meta

def get_default_video_loader(use_preprocessing=False):
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader,
                            use_preprocessing=use_preprocessing)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, use_preprocessing=False):
    if use_preprocessing:
        root_path, video_dir_path, clip_position_path = preprocess_ipn_dataset(root_path)

    if use_preprocessing and os.path.exists(clip_position_path):
        with open(clip_position_path, "r") as f:
            # clip_positions: filename -> ((top-left corner), width, height)
            clip_positions_dict = json.load(f)
    else:
        clip_positions_dict = {}

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: IPN Dataset - " + subset + " is loading...")
    print("  path: " + video_names[0])
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            continue

        

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class, clip_positions_dict


class IPN(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader,
                 use_preprocessing = False):
        self.data, self.class_names, self.clip_positions_dict = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, use_preprocessing = use_preprocessing)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader(use_preprocessing=use_preprocessing)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']


        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip, clip_meta = self.loader(path, frame_indices, self.clip_positions_dict, self.modality, self.sample_duration)

        oversample_clip = []
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)

        # clip is of shape [channel, sample_duration, image_dim[0], image_dim[1]]
        # target is just a single label index

        # TODO(cyjiang): transform clip_meta into appropriate data form
        gravity_position = np.array([i[0] for i in clip_meta], dtype=np.float32)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, gravity_position, target

    def __len__(self):
        return len(self.data)


