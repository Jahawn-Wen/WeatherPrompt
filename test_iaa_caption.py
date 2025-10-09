# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net
from project_utils import load_network
from image_folder import customData, customData_one, customData_style, ImageFolder_iaa
import imgaug.augmenters as iaa
import random
import shutil
from PIL import Image
from tqdm import tqdm
import json
from ruamel.yaml import YAML
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--style', default='none', type=str, help='select image style: e.g. night, nightfall, NightLight, shadow, StrongLight, all')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_VIT', action='store_true', help='use VIT' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scale_test', action='store_true', help='scale test' )
parser.add_argument('--iaa', action='store_true', help='iaa image augmentation' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument(
    '--xvlm_config',
    type=str,
    default='/home/wjh/project/MuseNet-master-test-vit/XVLM/X-VLM-master/configs/config_swinB_384.json',
    help="X-VLM vision (JSON) 配置路径"
)
parser.add_argument(
    '--xvlm_text_config',
    type=str,
    default='/home/wjh/project/MuseNet-master-test-vit/XVLM/X-VLM-master/configs/config_bert.json',
    help="X-VLM 文本（BERT/RoBERTa）配置路径"
)
parser.add_argument('--use_swin',        action='store_true', help='在 X-VLM 中启用 Swin Transformer')
parser.add_argument('--use_clip_vit',    action='store_true', help='在 X-VLM 中启用 CLIP-ViT')

parser.add_argument('--use_roberta',     action='store_true')
parser.add_argument('--embed_dim',       type=int,   default=256)
parser.add_argument('--temp',            type=float, default=0.07)
parser.add_argument('--max_tokens',      type=int,   default=256)
parser.add_argument('--use_mlm_loss',    action='store_true')
parser.add_argument('--use_bbox_loss',   action='store_true')


opt = parser.parse_args()
config_xvlm = json.load(open(opt.xvlm_config, 'r'))
config_xvlm['use_swin']       = opt.use_swin
config_xvlm['use_clip_vit']   = opt.use_clip_vit
config_xvlm['vision_config']  = opt.xvlm_config
config_xvlm['image_res']      = opt.h
config_xvlm['patch_size']     = 32


yaml = YAML(typ='safe')
if opt.xvlm_text_config.endswith('.json'):
    config_text = json.load(open(opt.xvlm_text_config, 'r'))
else:
    config_text = yaml.load(open(opt.xvlm_text_config, 'r'))

config_xvlm['use_roberta']    = opt.use_roberta
config_xvlm['text_config']    = opt.xvlm_text_config
config_xvlm['text_encoder']   = 'roberta-base' if opt.use_roberta else '/home/wjh/project/1014/APTM/data/bert-base-uncased'

config_xvlm['embed_dim']      = opt.embed_dim
config_xvlm['temp']           = opt.temp
config_xvlm['max_tokens']     = opt.max_tokens

config_xvlm['use_mlm_loss']   = opt.use_mlm_loss
config_xvlm['use_bbox_loss']  = opt.use_bbox_loss




###load config###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#debug
# opt.iaa = True
# opt.name = 'three_view_long_share_d0.75_256_s1_google_lr0.005_spade_v24.5_210ep_weather_1010000_5std'
# opt.test_dir = '/home/wangtyu/datasets/University-Release/test'
# opt.batchsize = 4

# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
yaml = YAML(typ='safe')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
opt.LPN = config['LPN']
opt.block = config['block']
scale_test = opt.scale_test
style = opt.style
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
print('------------------------------',opt.h)
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    # torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#像素点平移动的transforms
transform_move_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if opt.LPN:
    data_transforms = transforms.Compose([
        # transforms.Resize((384,192), interpolation=3),
        transforms.Resize((opt.h,opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
# using iaa image augmentation
if opt.iaa:
    iaa_transform = iaa.Sequential(
        [
            # rain
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),

            # fog
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),

            # # dark
            # iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            # iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),

            # light
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),  # guobao

            # snow
            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),

            # fog+rain
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),

            # # fog+snow
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),

            # rain+snow
            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),

            # wind
            # iaa.MotionBlur(15, seed=17),

            # unseen easy
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),

            # unseen 
            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),

            # iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
            #                alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            #                density_multiplier=0.5, seed=35),
            # iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            # iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            # iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            # iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),

            #暴风雨
            # iaa.Rain(drop_size=(0.1, 0.2), speed=(0.02, 0.05), seed=808),
            # iaa.Rain(drop_size=(0.15, 0.25), speed=(0.03, 0.06), seed=809),
            # iaa.CloudLayer( intensity_mean=240, intensity_freq_exponent=-2.2, intensity_coarse_scale=1.5, alpha_min=1.0, alpha_multiplier=0.8, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.6, seed=812),
            # iaa.MultiplyAndAddToBrightness(mul=0.5, add=(-30, -10), seed=888),
            # iaa.LinearContrast((0.7, 0.9)),

            #日落
            # iaa.AddToHueAndSaturation((10, 30)),
            # iaa.Multiply((0.8, 0.9)),# 降亮度
            # iaa.Add((5, 20)), 

            #沙尘暴
            # iaa.AddToHueAndSaturation((-15, -5)), # 暖色偏移，显得土
            # iaa.Multiply((0.75, 0.85)),  # 降亮度，显得混浊
            # iaa.CloudLayer( intensity_mean=200, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.8, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.4, seed=88 ),
           
        #   misty_snow_night
            # iaa.Multiply((0.4, 0.6)),   # 模拟夜间昏暗
            # iaa.AddToHueAndSaturation((-30, -10)), # 偏冷色调
            # iaa.Snowflakes(flake_size=(0.4, 0.8), speed=(0.02, 0.05), seed=123), # 较大的雪花
            # iaa.Fog(), # 已在上次修正中移除 alpha 参数
            # iaa.LinearContrast((0.6, 0.8)), # 对比度低

        #   heavy_snowfall
            # iaa.Snowflakes(flake_size=(0.6, 1.0), speed=(0.03, 0.07)), # **修正：flake_size 最大值改为 1.0，移除 density 参数**
            # iaa.Multiply((0.5, 0.7)), # 整体显著变暗，能见度降低
            # iaa.MotionBlur(k=(7, 11)), # 模拟大雪纷飞的模糊感
            # iaa.AddToHueAndSaturation((-20, -10)), # 偏冷色调，模拟雪天的寒冷感
            # iaa.LinearContrast((0.7, 0.9)), # 对比度降低，能见度差

        # windy_snowstorm
            # iaa.Snowflakes(flake_size=(0.8, 1.0), speed=(0.05, 0.1)), # **修正：flake_size 最大值改为 1.0，移除 density 参数**
            # iaa.Multiply((0.4, 0.6)), # 整体非常暗，能见度极低
            # iaa.MotionBlur(k=(9, 15), angle=(-45, 45)), # 强烈的运动模糊和随机角度，模拟风雪横吹
            # iaa.AddToHueAndSaturation((-30, -15)), # 强烈的冷色调
            # iaa.LinearContrast((0.6, 0.8)), # 对比度极低

            iaa.Resize({"height": opt.h, "width": opt.w}, interpolation=3),
        ]
)
    data_transforms_iaa = transforms.Compose([
        # transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_dir = test_dir


if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
elif opt.iaa:
    print('------------------processing images using iaa----------------------')
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x), data_transforms) for x in ['gallery_satellite','query_satellite']}
    for x in ['query_drone', 'gallery_drone']:
        image_datasets[x] = ImageFolder_iaa(os.path.join(data_dir,x), data_transforms_iaa, iaa_transform=iaa_transform)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}

else:
    # image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'query_satellite', 'query_drone', 'query_street']}
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street']}
    # image_datasets = {}
    # for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'gallery_satellite_usa_un']:
    #     image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    if scale_test:
        for x in ['query_drone']:
            print('----------scale test--------------')
            image_datasets[x] = customData_one( os.path.join(data_dir,x) ,data_transforms, rotate=0, reverse=False)
    else:
        for x in ['query_satellite', 'query_drone', 'query_street']:
            if opt.pad > 0:
                print('-----------move pixel test-----------')
                image_datasets[x] = customData( os.path.join(data_dir,x) ,transform_move_list, rotate=0, pad=opt.pad)
            else: 
                print('----------rotation test--------------')   
                image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    if style != 'none':
        for x in ['query_drone_style', 'gallery_drone_style']:
            image_datasets[x] = customData_style( os.path.join(data_dir,x) ,data_transforms, style=style)

    print(image_datasets.keys())
    # image_datasets = {x: customData( os.path.join(data_dir,x) ,data_transforms, rotate=0) for x in ['query_satellite', 'query_drone', 'query_street']}
    if scale_test:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'query_drone']}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'query_satellite', 'query_drone']}
    if style != 'none':
        print('using style is-----------------:', style)
        dataloaders['query_drone_style'] =  torch.utils.data.DataLoader(image_datasets['query_drone_style'], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16)
        dataloaders['gallery_drone_style'] =  torch.utils.data.DataLoader(image_datasets['gallery_drone_style'], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16)
use_gpu = torch.cuda.is_available()

# query_satellite
class_names = image_datasets['gallery_drone'].classes
# query_drone
# class_names = image_datasets['query_drone'].classes


# ——— 4) 写入 JSON ———
caption_path = "/home/wjh/project/MuseNet-master-test-vit/multiweather_captions_test_32B_gallery.json"

# ——— 5) 直接用它来构造测试时的 scene/weather captions 列表 ———
with open(caption_path, 'r', encoding='utf-8') as f:
    captions_dict = json.load(f)  # {class_id: {weather: caption, …}, …}


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def save_augmented_images(img_tensor, save_dir, start_index):
    os.makedirs(save_dir, exist_ok=True)
    img_paths = []
    for i in range(img_tensor.size(0)):
        img = img_tensor[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = transforms.ToPILImage()(img.clamp(0, 1))
        img_path = os.path.join(save_dir, f"aug_{start_index + i}.jpg")
        img.save(img_path)
        img_paths.append(img_path)
    return img_paths

def extract_feature(model, data_loader, view_index=1, ms=[1], captions_dict=None, opt=None):
    features = torch.FloatTensor()
    count = 0
    ptr = 0
    for data in data_loader:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        batch_captions = []
        # batch_captions = None

        if captions_dict is not None and view_index == 3:
            for cls_idx in label:
                cls_name = class_names[cls_idx.item()]
                batch_captions.append(captions_dict[cls_name]['light'])


        # ——————以下是你的原始模型推理代码 (不变)———————— #
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        
        for i in range(2):
            input_img = Variable(img.cuda())
            if i == 1:
                input_img = fliplr(input_img)
            
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                
                outputs = None
                if opt.views == 3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index == 2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index == 3:
                        _, _, outputs = model(None, None, input_img, captions=batch_captions)
                ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
        ptr += n  # 更新指针位置

    return features


# def extract_feature(model,dataloaders, view_index = 1):
#     features = torch.FloatTensor()
#     count = 0
#     for data in dataloaders:
#         img, label = data
#         n, c, h, w = img.size()
#         count += n
#         print(count)
#         ff = torch.FloatTensor(n,512).zero_().cuda()
#         if opt.LPN:
#             # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
#             ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
#         for i in range(2):
#             if(i==1):
#                 img = fliplr(img)
#             input_img = Variable(img.cuda())
#             for scale in ms:
#                 if scale != 1:
#                     # bicubic is only  available in pytorch>= 1.1
#                     input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
#                 if opt.views ==2:
#                     if view_index == 1:
#                         outputs, _ = model(input_img, None) 
#                     elif view_index ==2:
#                         _, outputs = model(None, input_img) 
#                 elif opt.views ==3:
#                     if view_index == 1:
#                         outputs, _, _ = model(input_img, None, None)
#                     elif view_index ==2:
#                         _, outputs, _ = model(None, input_img, None)
#                     elif view_index ==3:
#                         _, _, outputs = model(None, None, input_img)
#                 ff += outputs
#         # norm feature
#         if opt.LPN:
#             # feature size (n,2048,6)
#             # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
#             # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
#             fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block) 
#             ff = ff.div(fnorm.expand_as(ff))
#             ff = ff.view(ff.size(0), -1)
#         else:
#             fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
#             ff = ff.div(fnorm.expand_as(ff))

#         features = torch.cat((features,ff.data.cpu()), 0)
#     return features


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
if opt.LPN:
    print('use LPN')
    # model = three_view_net_test(model)
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()
# print(model)
# Extract feature
since = time.time()

# gallery_name = 'gallery_street'

query_name = 'query_satellite'
gallery_name = 'gallery_drone'

# query_name = 'query_drone'
# gallery_name = 'gallery_satellite'

# query_name = 'query_street'
# gallery_name = 'gallery_street'

# query_name = 'query_drone_style'
# query_name = 'query_drone_one'

# gallery_name = 'gallery_drone_style'
# gallery_name = 'gallery_satellite_usa_un'
which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,
                                        dataloaders['query_satellite'],
                                        view_index=which_query,
                                        ms=[1],
                                        captions_dict=captions_dict,
                                        opt=opt)
        
        gallery_feature = extract_feature(model,
                                          dataloaders['gallery_drone'],
                                          view_index=which_gallery,
                                          ms=[1],
                                          captions_dict=captions_dict,
                                          opt=opt)
        # query_feature = extract_feature(model,
        #                                 dataloaders['query_drone'],
        #                                 view_index=which_query,
        #                                 ms=[1],
        #                                 captions_dict=captions_dict,
        #                                 opt=opt)
        
        # gallery_feature = extract_feature(model,
        #                                   dataloaders['gallery_satellite'],
        #                                   view_index=which_gallery,
        #                                   ms=[1],
        #                                   captions_dict=captions_dict,
        #                                   opt=opt)



    # For street-view image, we use the avg feature as the final feature.
    '''
    if which_query == 2:
        new_query_label = np.unique(query_label)
        new_query_feature = torch.FloatTensor(len(new_query_label) ,512).zero_()
        for i, query_index in enumerate(new_query_label):
            new_query_feature[i,:] = torch.sum(query_feature[query_label == query_index, :], dim=0)
        query_feature = new_query_feature
        fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
        query_feature = query_feature.div(fnorm.expand_as(query_feature))
        query_label   = new_query_label
    elif which_gallery == 2:
        new_gallery_label = np.unique(gallery_label)
        new_gallery_feature = torch.FloatTensor(len(new_gallery_label), 512).zero_()
        for i, gallery_index in enumerate(new_gallery_label):
            new_gallery_feature[i,:] = torch.sum(gallery_feature[gallery_label == gallery_index, :], dim=0)
        gallery_feature = new_gallery_feature
        fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
        gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
        gallery_label   = new_gallery_label
    '''
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat',result)

    print(opt.name)
    result = './model/%s/result.txt'%opt.name
    os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))
    # os.system('python evaluate_gpu.py | tee -a %s'%result)
    #test single part and combination


    import scipy.io
    import torch
    import numpy as np
    import os

    # Evaluate function to get the top 10 matches
    def evaluate_top10_matches(qf, ql, gf, gl, g_paths):
        query = qf.view(-1,1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        index = np.argsort(score)[::-1][:10]  # Get the top 10 indices
        # Extracting the matched image paths
        match_paths = [g_paths[idx] for idx in index]
        return match_paths

    # Load features and labels
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    query_paths = result['query_path']
    gallery_paths = result['gallery_path']

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # Store all query top 10 match results
    all_matches = {}

    for i in range(len(query_label)):
        matches = evaluate_top10_matches(query_feature[i], query_label[i], gallery_feature, gallery_label, gallery_paths)
        all_matches[query_paths[i]] = matches

    # Write the match results into a text file
    with open('top10_matches.txt', 'w') as f:
        for query, matches in all_matches.items():
            f.write(f'Query: {query}\n')
            for match in matches:
                f.write(f'{match}\n')
            f.write('\n')

    print("Matching results have been saved to 'top10_matches.txt'.")





    '''
    # for i in range(7):
    #     if i == 0:
    #         query_feature_ = query_feature[:,0:512]
    #         gallery_feature_ = gallery_feature[:,0:512]
    #         print('-------------- 1 -----------------')
    #     if i == 1:
    #         query_feature_ = query_feature[:,512:1024]
    #         gallery_feature_ = gallery_feature[:,512:1024]
    #         print('-------------- 2 -----------------')
    #     if i == 2:
    #         query_feature_ = query_feature[:,1024:1536]
    #         gallery_feature_ = gallery_feature[:,1024:1536]
    #         print('-------------- 3 -----------------')
    #     if i == 3:
    #         query_feature_ = query_feature[:,1536:2048]
    #         gallery_feature_ = gallery_feature[:,1536:2048]
    #         print('-------------- 4 -----------------')
    #     if i == 4:
    #         query_feature_ = query_feature[:,0:1024]
    #         gallery_feature_ = gallery_feature[:,0:1024]
    #         print('-------------- 1+2 -----------------')
    #     if i == 5:
    #         query_feature_ = query_feature[:,0:1536]
    #         gallery_feature_ = gallery_feature[:,0:1536]
    #         print('-------------- 1+2+3 -----------------')
    #     if i == 6:
    #         query_feature_ = query_feature[:,0:2048]
    #         gallery_feature_ = gallery_feature[:,0:2048]
    #         print('-------------- 1+2+3+4 -----------------')
    #     result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    #     scipy.io.savemat('pytorch_result.mat',result)
    #     print(opt.name)
    #     result = './model/%s/result.txt'%opt.name
    #     os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))
    '''
    # query_feature_ = query_feature[:,0:1536]
    # gallery_feature_ = gallery_feature[:,512:2048]
    # print('-------------- （1+2+3，2+3+4） -----------------')
    # result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    # scipy.io.savemat('pytorch_result.mat',result)
    # print(opt.name)
    # result = './model/%s/result.txt'%opt.name
    # os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))