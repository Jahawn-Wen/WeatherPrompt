# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import os
import time
import math
import json
import shutil
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import scipy.io
from ruamel.yaml import YAML

from model import ft_net, two_view_net, three_view_net
from project_utils import load_network
from image_folder import customData, customData_one, customData_style, ImageFolder_iaa
import imgaug.augmenters as iaa


# ------------------------------- CLI ------------------------------- #
parser = argparse.ArgumentParser(description='Feature Extraction / Evaluation')


parser.add_argument('--gpu_ids', default='0', type=str, help='e.g. "0" 或 "0,1"')
parser.add_argument('--which_epoch', default='last', type=str, help='last 或具体数字')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='训练产物目录名（./model/<name>）')


parser.add_argument('--test_dir', default='./data/test', type=str, help='测试数据根目录')
parser.add_argument('--gallery_split', default='gallery_drone', type=str, help='图库子目录名')
parser.add_argument('--query_split',   default='query_satellite', type=str, help='查询子目录名')


parser.add_argument('--h', default=384, type=int)
parser.add_argument('--w', default=384, type=int)
parser.add_argument('--pad', default=0, type=int, help='像素平移幅度（>0 开启 customData 平移）')
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--pool', default='avg', choices=['avg', 'max', 'avg+max'])
parser.add_argument('--views', default=3, type=int, choices=[2, 3], help='2-view 或 3-view 模型')


parser.add_argument('--ms', default='1', type=str, help='多尺度，逗号分隔，如 "1,1.1"')
parser.add_argument('--scale_test', action='store_true', help='对 query_drone 用 customData_one 做多尺度/旋转评测')
parser.add_argument('--iaa', action='store_true', help='对 drone 分支用 IAA 增强（ImageFolder_iaa）')
parser.add_argument('--style', default='none', type=str, help='为 *_drone_style 分支选择风格集（none 表示不启用）')


parser.add_argument('--captions_json', default=None, type=str, help='类别→天气描述 JSON 路径（可选）')
parser.add_argument('--caption_weather_key', default='light', type=str, help='JSON 里取哪种天气描述，如 "light"')
parser.add_argument('--topk', default=10, type=int, help='为每个 query 另存 Top-K 匹配结果')


parser.add_argument('--result_mat', default='pytorch_result.mat', type=str)
parser.add_argument('--topk_txt', default='topk_matches.txt', type=str)

opt = parser.parse_args()



ms = [math.sqrt(float(s)) for s in opt.ms.split(',') if s.strip()]


gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if x.strip()]
if len(gpu_ids) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    cudnn.benchmark = True
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print('Using device:', device)



data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_move_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if opt.iaa:
    iaa_transform = iaa.Sequential([
        iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),
        iaa.Resize({"height": opt.h, "width": opt.w}, interpolation=3),
    ])
    data_transforms_iaa = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = opt.test_dir


image_datasets = {
    'gallery_satellite': datasets.ImageFolder(os.path.join(data_dir, 'gallery_satellite'), data_transforms),
    'gallery_drone':     datasets.ImageFolder(os.path.join(data_dir, 'gallery_drone'),     data_transforms),
    'gallery_street':    datasets.ImageFolder(os.path.join(data_dir, 'gallery_street'),    data_transforms),
}


if opt.scale_test and opt.query_split == 'query_drone':
    image_datasets['query_drone'] = customData_one(os.path.join(data_dir, 'query_drone'), data_transforms, rotate=0, reverse=False)
else:
    for q in ['query_satellite', 'query_drone', 'query_street']:
        if opt.pad > 0:
            image_datasets[q] = customData(os.path.join(data_dir, q), transform_move_list, rotate=0, pad=opt.pad)
        else:
            image_datasets[q] = customData(os.path.join(data_dir, q), data_transforms, rotate=0)


if opt.iaa:
    image_datasets['query_drone'] = ImageFolder_iaa(os.path.join(data_dir, 'query_drone'), data_transforms_iaa, iaa_transform=iaa_transform)
    image_datasets['gallery_drone'] = ImageFolder_iaa(os.path.join(data_dir, 'gallery_drone'), data_transforms_iaa, iaa_transform=iaa_transform)

if opt.style != 'none':
    image_datasets['query_drone_style'] = customData_style(os.path.join(data_dir, 'query_drone_style'), data_transforms, style=opt.style)
    image_datasets['gallery_drone_style'] = customData_style(os.path.join(data_dir, 'gallery_drone_style'), data_transforms, style=opt.style)


need_keys = {opt.gallery_split, opt.query_split}
if opt.style != 'none':
    if 'drone' in opt.query_split:
        need_keys.add('query_drone_style')
    if 'drone' in opt.gallery_split:
        need_keys.add('gallery_drone_style')

dataloaders = {
    k: torch.utils.data.DataLoader(
        image_datasets[k],
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=opt.num_workers
    ) for k in need_keys
}

print('Dataloaders:', list(dataloaders.keys()))



captions_dict: Optional[Dict[str, Dict[str, str]]] = None
if opt.captions_json is not None and os.path.isfile(opt.captions_json):
    with open(opt.captions_json, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)
    print(f'Loaded captions from {opt.captions_json} (key="{opt.caption_weather_key}")')
else:
    print('No captions JSON provided')


def fliplr(img: torch.Tensor) -> torch.Tensor:
    inv_idx = torch.arange(img.size(3) - 1, -1, -1, device=img.device)
    return img.index_select(3, inv_idx)

def which_view(name: str) -> int:
    if 'satellite' in name:
        return 1
    if 'street' in name:
        return 2
    if 'drone' in name:
        return 3
    raise ValueError(f'unknown view for split "{name}"')

def get_id(img_path_list):
    labels, paths = [], []
    for path, _ in img_path_list:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

def extract_feature(model, data_loader, view_index: int, ms: List[float], dataset_classes: List[str],
                    captions: Optional[Dict[str, Dict[str, str]]] = None,
                    weather_key: Optional[str] = None):
    features = torch.FloatTensor().cpu()
    count = 0

    for imgs, labels in tqdm(data_loader, desc=f'Extract v{view_index}'):
        n = imgs.size(0)
        count += n


        batch_captions: Optional[List[str]] = None
        if captions is not None and weather_key is not None and view_index == 3:
            batch_captions = []
            for cls_idx in labels:
                cls_name = dataset_classes[cls_idx.item()]
                wcapt = captions.get(cls_name, {})

                text = wcapt.get(weather_key)
                if text is None and len(wcapt) > 0:
                    text = next(iter(wcapt.values()))
                batch_captions.append(text if text is not None else "")

        ff = torch.zeros(n, 512, device=device)

        for flip in range(2):
            input_img = imgs.to(device, non_blocking=True)
            if flip == 1:
                input_img = fliplr(input_img)

            for s in ms:
                img_scale = input_img
                if abs(s - 1.0) > 1e-6:
                    img_scale = nn.functional.interpolate(input_img, scale_factor=s, mode='bilinear', align_corners=False)

                with torch.no_grad():
                    if opt.views == 3:
                        if view_index == 1:
                            out, _, _ = model(img_scale, None, None)
                        elif view_index == 2:
                            _, out, _ = model(None, img_scale, None)
                        elif view_index == 3:
                            _, _, out = model(None, None, img_scale, captions=batch_captions)
                        else:
                            raise ValueError('view_index must be 1/2/3 for 3-view model')
                    else:  # 2-view
                        if view_index == 1:
                            out, _ = model(img_scale, None)
                        elif view_index == 2:
                            _, out = model(None, img_scale)
                        else:
                            raise ValueError('view_index must be 1/2 for 2-view model')
                ff += out

        # L2 normalize
        ff = ff / (ff.norm(p=2, dim=1, keepdim=True) + 1e-12)

        features = torch.cat((features, ff.detach().cpu()), dim=0)

    return features



print('------- test / extract -------')
model, _, epoch = load_network(opt.name, opt)

if hasattr(model, 'classifier'):
    model.classifier.classifier = nn.Sequential()
model.eval().to(device)



gallery_name = opt.gallery_split
query_name   = opt.query_split
which_gallery = which_view(gallery_name)
which_query   = which_view(query_name)
print(f'{which_query} -> {which_gallery}')


gallery_paths_raw = image_datasets[gallery_name].imgs
query_paths_raw   = image_datasets[query_name].imgs
with open('gallery_name.txt', 'w') as f:
    for p in gallery_paths_raw:
        f.write(p[0] + '\n')
with open('query_name.txt', 'w') as f:
    for p in query_paths_raw:
        f.write(p[0] + '\n')

gallery_label, gallery_path = get_id(gallery_paths_raw)
query_label,   query_path   = get_id(query_paths_raw)


gallery_classes = image_datasets[gallery_name].classes
query_classes   = image_datasets[query_name].classes


since = time.time()
with torch.no_grad():
    qfeat = extract_feature(
        model,
        dataloaders[query_name],
        view_index=which_query,
        ms=ms if len(ms) > 0 else [1.0],
        dataset_classes=query_classes,
        captions=captions_dict,
        weather_key=(opt.caption_weather_key if captions_dict is not None else None)
    )
    gfeat = extract_feature(
        model,
        dataloaders[gallery_name],
        view_index=which_gallery,
        ms=ms if len(ms) > 0 else [1.0],
        dataset_classes=gallery_classes,
        captions=captions_dict,
        weather_key=(opt.caption_weather_key if captions_dict is not None else None)
    )

elapsed = time.time() - since
print('Feature extraction done in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))


result_mat = {
    'gallery_f': gfeat.numpy(),
    'gallery_label': np.array(gallery_label),
    'gallery_path': np.array(gallery_path, dtype=object),
    'query_f': qfeat.numpy(),
    'query_label': np.array(query_label),
    'query_path': np.array(query_path, dtype=object),
}
scipy.io.savemat(opt.result_mat, result_mat)
print(f'Saved features to {opt.result_mat}')


if len(gpu_ids) == 0:
    cmd = f'python evaluate_gpu.py | tee -a ./model/{opt.name}/result.txt'
else:
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_ids[0]} python evaluate_gpu.py | tee -a ./model/{opt.name}/result.txt'
os.makedirs(f'./model/{opt.name}', exist_ok=True)
os.system(cmd)


def topk_matches(qf, gf, g_paths, k=10):
    # qf: [512], gf: [N,512]
    score = torch.matmul(gf, qf.view(-1, 1)).squeeze(1).cpu().numpy()
    idx = np.argsort(score)[::-1][:k]
    return [g_paths[i] for i in idx]

qf = torch.tensor(result_mat['query_f'], device=device).float()
gf = torch.tensor(result_mat['gallery_f'], device=device).float()
all_matches = {}
for i in range(qf.size(0)):
    matches = topk_matches(qf[i], gf, result_mat['gallery_path'], k=opt.topk)
    all_matches[result_mat['query_path'][i]] = matches

with open(opt.topk_txt, 'w', encoding='utf-8') as f:
    for q, ms_ in all_matches.items():
        f.write(f'Query: {q}\n')
        for m in ms_:
            f.write(f'{m}\n')
        f.write('\n')
print(f"Top-{opt.topk} matching results saved to '{opt.topk_txt}'.")
