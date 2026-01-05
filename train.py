from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
import time
import os
import sys
import math
import random
import json
import uuid
import io
import base64
import numpy as np

from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from ruamel.yaml import YAML

from model import two_view_net, three_view_net
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import imgaug.augmenters as iaa

from image_folder import (
    ImageFolder_iaa_selectID,
    ImageFolder_iaa_multi_weather,
    PathFolder_qwen,
    ImageFolder_iaa_multi_weather_single
)
from torch.utils.tensorboard import SummaryWriter
from circle_loss import CircleLoss, convert_label_to_similarity, FocalLoss

from project_utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
from models.xvlm import XVLMBase

version = torch.__version__

# -----------------------------
# Path configuration for open-source release
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent


#   export XVLM_ROOT=/path/to/X-VLM-master
#   export XVLM_CKPT=/path/to/ckpt.th
#   export CAPTION_JSON=/path/to/multiweather_captions.json
#   export TEXT_ENCODER=bert-base-uncased  (或本地目录)
ENV_XVLM_ROOT = os.environ.get("XVLM_ROOT", str(REPO_ROOT / "third_party" / "XVLM" / "X-VLM-master"))

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--xvlm_root", type=str, default=ENV_XVLM_ROOT)
_pre_args, _ = _pre_parser.parse_known_args()

XVLM_ROOT = Path(_pre_args.xvlm_root).expanduser()
DEFAULT_XVLM_CONFIG = os.environ.get("XVLM_CONFIG", str(XVLM_ROOT / "configs" / "config_swinB_384.json"))
DEFAULT_XVLM_TEXT_CONFIG = os.environ.get("XVLM_TEXT_CONFIG", str(XVLM_ROOT / "configs" / "config_bert.json"))
DEFAULT_XVLM_CKPT = os.environ.get(
    "XVLM_CKPT",
    str(REPO_ROOT / "third_party" / "XVLM" / "4m_base_model_state_step_199999.th")
)
DEFAULT_CAPTION_JSON = os.environ.get("CAPTION_JSON", str(REPO_ROOT / "multiweather_captions.json"))
DEFAULT_TEXT_ENCODER = os.environ.get("TEXT_ENCODER", "bert-base-uncased")

# -----------------------------
# Distributed init (keep your original behavior)
# -----------------------------
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:23456',
        world_size=1,
        rank=0
    )

# -----------------------------
# sys.path setup (open-source friendly)
# -----------------------------

proj_root = str(REPO_ROOT)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


xvlm_root = str(XVLM_ROOT)
if xvlm_root not in sys.path:
    sys.path.insert(1, xvlm_root)



os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='two_view', type=str, help='output model name')
parser.add_argument('--experiment_name', default='debug', type=str, help='log dir name')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning')
parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
parser.add_argument('--LPN', action='store_true', help='use LPN')
parser.add_argument('--iaa', action='store_true', help='use iaa')
parser.add_argument('--reptile', action='store_true', help='use Reptile')
parser.add_argument('--qwen', action='store_true', help='use qwen')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--focal', action='store_true', help='use Focal loss')
parser.add_argument('--block', default=6, type=int, help='the num of block')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--norm', default='bn', type=str, help='selecting norm from [bn, ibn, ada-ibn]')
parser.add_argument('--multi_weather', action='store_true', help='use multiple weather')
parser.add_argument('--adain', default='a', type=str, help='the mode of adain: a or b')
parser.add_argument('--conv_norm', default='none', type=str, help='none, in, ln')
parser.add_argument('--style_loss', action='store_true', help='use style loss')
parser.add_argument('--btnk', nargs='+', type=int, default=[1, 0, 1], help='determining the btnk')

# -----------------------------
# Open-source friendly path args
# -----------------------------
parser.add_argument(
    '--xvlm_root',
    type=str,
    default=str(XVLM_ROOT),
    help="Path to X-VLM repo root. Replace with your own path or set env XVLM_ROOT."
)
parser.add_argument(
    '--xvlm_config',
    type=str,
    default=DEFAULT_XVLM_CONFIG,
    help="X-VLM vision config JSON. Replace with your own path or set env XVLM_CONFIG."
)
parser.add_argument(
    '--xvlm_text_config',
    type=str,
    default=DEFAULT_XVLM_TEXT_CONFIG,
    help="X-VLM text config (JSON/YAML). Replace with your own path or set env XVLM_TEXT_CONFIG."
)
parser.add_argument(
    '--xvlm_ckpt',
    type=str,
    default=DEFAULT_XVLM_CKPT,
    help="Path to X-VLM pretrained checkpoint (.th). Replace with your own path or set env XVLM_CKPT."
)
parser.add_argument(
    '--caption_path',
    type=str,
    default=DEFAULT_CAPTION_JSON,
    help="Path to multi-weather captions JSON. Replace with your own path or set env CAPTION_JSON."
)
parser.add_argument(
    '--text_encoder',
    type=str,
    default=DEFAULT_TEXT_ENCODER,
    help="HF name (e.g., bert-base-uncased) or local dir. Replace with your own path or set env TEXT_ENCODER."
)

# X-VLM extra switches
parser.add_argument('--use_swin', action='store_true', help="Enable Swin Transformer in X-VLM")
parser.add_argument('--use_clip_vit', action='store_true', help="Enable CLIP-ViT in X-VLM")
parser.add_argument('--use_roberta', action='store_true', help="Enable RoBERTa text encoder in X-VLM")

opt = parser.parse_args()
args = opt


def _p(s: str) -> str:
    return str(Path(s).expanduser())

args.xvlm_root = _p(args.xvlm_root)
args.xvlm_config = _p(args.xvlm_config)
args.xvlm_text_config = _p(args.xvlm_text_config)
args.xvlm_ckpt = _p(args.xvlm_ckpt)
args.caption_path = _p(args.caption_path)

with open(args.xvlm_config, 'r') as f:
    config_xvlm = json.load(f)


config_xvlm['use_swin'] = args.use_swin
config_xvlm['use_clip_vit'] = args.use_clip_vit
config_xvlm['vision_config'] = args.xvlm_config
config_xvlm['image_res'] = args.h
config_xvlm['patch_size'] = 32
config_xvlm['use_swin'] = True


ruamel_yaml_safe = YAML(typ='safe')
if args.xvlm_text_config.endswith('.json'):
    with open(args.xvlm_text_config, 'r') as f:
        config_text = json.load(f)
else:
    config_text = ruamel_yaml_safe.load(open(args.xvlm_text_config, 'r'))

config_xvlm['use_roberta'] = args.use_roberta
config_xvlm['text_config'] = args.xvlm_text_config


config_xvlm['text_encoder'] = 'roberta-base' if args.use_roberta else args.text_encoder


config_xvlm['embed_dim'] = 256
config_xvlm['temp'] = 0.07
config_xvlm['max_tokens'] = 256


config_xvlm['use_mlm_loss'] = True
config_xvlm['use_bbox_loss'] = True

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

print('btnk:-------------------', opt.btnk)

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if opt.seed > 0:
    print('random seed---------------------:', opt.seed)
    seed_torch(seed=opt.seed)

# set gpu ids
if len(gpu_ids) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    print("gpu_ids:", gpu_ids)
    cudnn.benchmark = True

######################################################################
# Load Data
######################################################################
transform_train_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.iaa:
    print('-----------------using iaa to augment the drone image----------------------------')
    iaa_drone_transform = iaa.Sequential([
        iaa.Resize({"height": opt.h, "width": opt.w}, interpolation=3),
        iaa.Pad(px=opt.pad, pad_mode="edge", keep_size=False),
        iaa.CropToFixedSize(width=opt.w, height=opt.h),
        iaa.Fliplr(0.5),
    ])

    iaa_weather_list = [
        None,
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225,
                intensity_freq_exponent=-2,
                intensity_coarse_scale=2,
                alpha_min=1.0,
                alpha_multiplier=0.9,
                alpha_size_px_max=10,
                alpha_freq_exponent=-2,
                sparsity=0.9,
                density_multiplier=0.5,
                seed=35
            )
        ]),
        iaa.Sequential([
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
        ]),
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
        ]),
        iaa.Sequential([
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]),
        iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)
        ]),
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35
            ),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
        ]),
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35
            ),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
        ]),
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        ]),
        iaa.Sequential([
            iaa.MotionBlur(15, seed=17)
        ])
    ]

    transform_iaa_drone_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list),
    'drone': {
        'transform': transform_iaa_drone_list if opt.iaa else None,
        'iaa_transform': iaa_drone_transform if opt.iaa else None,
        'iaa_weather_list': iaa_weather_list if opt.iaa else None,
        'batchsize': opt.batchsize,
        'shuffle': True,
        'norm': opt.norm
    }
}

# -----------------------------
# Captions path (open-source friendly)
# -----------------------------
caption_path = args.caption_path  # 
with open(caption_path, 'r', encoding='utf-8') as f:
    captions_dict = json.load(f)  # {class_id: {weather: caption, …}, …}

# The order of weather_names must match the order of iaa_weather_list in ImageFolder_iaa_multi_weather
weather_names = ["fog", "rain", "snow", "dark", "light", "fog_rain", "fog_snow", "rain_snow", "wind"]

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_iaa_path = {}

image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms['satellite'])
image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'), data_transforms['train'])

category = []  
if opt.iaa:
    print('-----------------using iaa to augment the drone image----------------------------')
    if opt.multi_weather:
        print('-----------------using multiple weather to augment the drone image----------------------------')
        category = ['drone']
        if opt.reptile:
            category = ["drone", "fog", "rain", "snow", "dark", "light", "fog_rain", "fog_snow", "rain_snow", "wind"]
            for functions, category_name in zip(iaa_weather_list, category):
                image_datasets[category_name] = ImageFolder_iaa_multi_weather_single(
                    os.path.join(data_dir, 'drone'),
                    transform=transform_iaa_drone_list,
                    iaa_transform=iaa_drone_transform,
                    iaa_weather_list=functions,
                    batchsize=opt.batchsize,
                    shuffle=False,
                    norm=opt.norm,
                    select=True
                )

            image_datasets['drone1'] = ImageFolder_iaa_multi_weather(
                os.path.join(data_dir, 'drone'),
                transform=transform_iaa_drone_list,
                iaa_transform=iaa_drone_transform,
                iaa_weather_list=iaa_weather_list,
                batchsize=opt.batchsize,
                shuffle=True,
                norm=opt.norm,
                select=True
            )
    else:
        image_datasets['drone'] = ImageFolder_iaa_selectID(
            os.path.join(data_dir, 'drone'),
            transform=transform_iaa_drone_list,
            iaa_transform=iaa_drone_transform,
            norm=opt.norm
        )
else:
    image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'), data_transforms['train'])

image_datasets['google'] = datasets.ImageFolder(os.path.join(data_dir, 'google'), data_transforms['train'])


loader_keys = ['satellite', 'street', 'google']
if 'drone1' in image_datasets:
    loader_keys.append('drone1')
loader_keys += [k for k in category if k in image_datasets]

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    for x in loader_keys
}

dataset_sizes = {x: len(image_datasets[x]) for x in loader_keys}
class_names = image_datasets['street'].classes
print(dataset_sizes)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Training helpers
######################################################################
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

def one_LPN_output(outputs, labels, criterion, block):
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)
    return preds, loss

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_style_loss(target, input):
    mse_loss = nn.MSELoss()
    assert (len(input) == len(target))
    style_loss = 0
    for i in range(len(input)):
        target_tmp = target[i].clone()
        input_tmp = input[i]
        assert (input_tmp.size() == target_tmp.size())
        input_mean, input_std = calc_mean_std(input_tmp)
        target_mean, target_std = calc_mean_std(target_tmp)
        style_loss = style_loss + (
            mse_loss(input_mean, target_mean.detach()) +
            mse_loss(input_std, target_std.detach())
        )
    return style_loss

def train_model(model, model_test, criterion, optimizer, scheduler, num_epochs=25, opt_pt=None, pt_scheduler=None):
    since = time.time()

    warm_up = 0.1
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch

    criterion_circle = None
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=64)
    if opt.focal:
        criterion_circle = FocalLoss(gamma=2.0, alpha=0.25)

    is_weather_norm = opt.norm in {'ada-ibn', 'spade'}

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            running_loss_ce = None
            running_loss_cir = None
            running_corrects_s_w = None
            running_corrects_d_w = None
            running_weather_loss = None
            running_contrastive_loss = None

            if opt.circle:
                running_loss_ce = 0.0
                running_loss_cir = 0.0
            if is_weather_norm:
                running_corrects_s_w = 0.0
                running_corrects_d_w = 0.0
                running_weather_loss = 0.0
                if opt.style_loss:
                    running_style_loss = 0.0
            if epoch >= 0:
                drone_loader_key = 'drone1' if 'drone1' in dataloaders else 'drone'
                for data, data2, data3, data4 in zip(
                    dataloaders['satellite'],
                    dataloaders['street'],
                    dataloaders[drone_loader_key],
                    dataloaders['google']
                ):
                    inputs, labels = data
                    inputs2, labels2 = data2
                    inputs4, labels4 = data4

                    if is_weather_norm:
                        inputs3, labels3, wlabels3, indeces = data3
                        if opt.qwen:
                            batch_captions = []
                            for cls_idx, w_idx in zip(labels3, wlabels3):
                                cls_name = class_names[cls_idx.item()]
                                w = w_idx.item() if hasattr(w_idx, 'item') else int(w_idx)
                                if 1 <= w <= len(weather_names):
                                    cap = captions_dict[cls_name][weather_names[w - 1]]
                                else:
                                    cap = captions_dict[cls_name]['normal']
                                batch_captions.append(cap)
                        wlabels1 = torch.zeros_like(wlabels3)
                    else:
                        inputs3, labels3 = data3

                    now_batch_size, c, h, w = inputs.shape
                    if now_batch_size < opt.batchsize:
                        continue

                    if use_gpu:
                        inputs = Variable(inputs.cuda().detach())
                        inputs2 = Variable(inputs2.cuda().detach())
                        inputs3 = Variable(inputs3.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                        labels2 = Variable(labels2.cuda().detach())
                        labels3 = Variable(labels3.cuda().detach())
                        if is_weather_norm:
                            wlabels3 = Variable(wlabels3.cuda().detach())
                            wlabels1 = Variable(wlabels1.cuda().detach())
                        if opt.extra_Google:
                            inputs4 = Variable(inputs4.cuda().detach())
                            labels4 = Variable(labels4.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    if opt_pt is not None:
                        opt_pt.zero_grad()

                    if phase == 'val':
                        with torch.no_grad():
                            outputs, outputs2 = model(inputs, inputs2)
                    else:
                        if opt.views == 2:
                            outputs, outputs2 = model(inputs, inputs2)
                        elif opt.views == 3:
                            if opt.extra_Google:
                                if opt.norm == 'ada-ibn':
                                    outputs, outputs2, outputs3, outputs4 = model(
                                        inputs, inputs2, inputs3, inputs4,
                                        scene_captions=scene_captions,
                                        weather_captions=weather_captions
                                    )
                                elif opt.norm == 'spade':
                                    if opt.qwen:
                                        outputs, outputs2, outputs3, outputs4 = model(
                                            inputs, inputs2, inputs3, inputs4,
                                            captions=batch_captions
                                        )
                                    else:
                                        outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs3, inputs4)
                                else:
                                    outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs3, inputs4)
                            else:
                                outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)

                    if not opt.LPN and opt.circle:
                        logits1, ff1 = outputs
                        logits2, ff2 = outputs2
                        logits3, ff3 = outputs3

                        _, preds = torch.max(logits1.data, 1)
                        _, preds2 = torch.max(logits2.data, 1)
                        _, preds3 = torch.max(logits3.data, 1)

                        ff1 = F.normalize(ff1)
                        ff2 = F.normalize(ff2)
                        ff3 = F.normalize(ff3)
                        loss1_ce = criterion(logits1, labels)
                        loss1_cir = criterion_circle(*convert_label_to_similarity(ff1, labels)) / now_batch_size

                        loss2_ce = criterion(logits2, labels2)
                        loss2_cir = criterion_circle(*convert_label_to_similarity(ff2, labels2)) / now_batch_size

                        loss3_ce = criterion(logits3, labels3)
                        loss3_cir = criterion_circle(*convert_label_to_similarity(ff3, labels3)) / now_batch_size

                        loss_ce = loss1_ce + loss2_ce + loss3_ce
                        loss_cir = 2 * (loss1_cir + loss2_cir + loss3_cir)
                        loss = loss_ce + loss_cir
                        if opt.extra_Google:
                            logits4, ff4 = outputs4
                            ff4 = F.normalize(ff4)
                            loss4 = criterion(logits4, labels4) + 2 * criterion_circle(*convert_label_to_similarity(ff4, labels4)) / now_batch_size
                            loss = loss + loss4

                    elif not opt.LPN and not opt.circle:
                        _, preds = torch.max(outputs.data, 1)
                        _, preds2 = torch.max(outputs2.data, 1)

                        if opt.views == 2:
                            loss = criterion(outputs, labels) + criterion(outputs2, labels2)
                        elif opt.views == 3:
                            _, preds3 = torch.max(outputs3.data, 1)
                            loss = criterion(outputs, labels) + 1 * criterion(outputs2, labels2) + criterion(outputs3, labels3)
                            if opt.extra_Google:
                                loss = loss + 1 * criterion(outputs4, labels4)
                            if is_weather_norm:
                                loss_itc = model.loss_itc
                                loss_itm = model.loss_itm
                                loss = loss + loss_itc + loss_itm
                                if opt.style_loss:
                                    style_loss = calc_style_loss(in_s, in_d)
                                    loss = loss + 1 * style_loss
                    else:
                        preds, loss = one_LPN_output(outputs, labels, criterion, opt.block)
                        preds2, loss2 = one_LPN_output(outputs2, labels2, criterion, opt.block)

                        if opt.views == 2:
                            loss = loss + loss2
                        elif opt.views == 3:
                            preds3, loss3 = one_LPN_output(outputs3, labels3, criterion, opt.block)
                            loss = loss + loss2 + loss3
                            if opt.extra_Google:
                                _, loss4 = one_LPN_output(outputs4, labels4, criterion, opt.block)
                                loss = loss + loss4

                    if epoch < opt.warm_epoch and phase == 'train':
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss *= warm_up

                    if phase == 'train':
                        if fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()
                        if opt_pt is not None:
                            opt_pt.step()
                        if opt.moving_avg < 1.0:
                            update_average(model_test, model, opt.moving_avg)

                    if int(version[0]) > 0 or int(version[2]) > 3:
                        running_loss += loss.item() * now_batch_size
                        running_contrastive_loss = 0.0
                        if opt.circle:
                            running_loss_ce += loss_ce.item() * now_batch_size
                            running_loss_cir += loss_cir.item() * now_batch_size
                    else:
                        running_loss += loss.data[0] * now_batch_size

                    running_corrects += float(torch.sum(preds == labels.data))
                    running_corrects2 += float(torch.sum(preds2 == labels2.data))
                    if opt.views == 3:
                        running_corrects3 += float(torch.sum(preds3 == labels3.data))
                    if is_weather_norm:
                        if opt.style_loss:
                            running_style_loss += style_loss.item() * now_batch_size

            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

            if opt.circle:
                epoch_loss_ce = running_loss_ce / dataset_sizes['satellite']
                epoch_loss_cir = running_loss_cir / dataset_sizes['satellite']
            if is_weather_norm:
                if opt.style_loss:
                    epoch_style_loss = running_style_loss / dataset_sizes['satellite']

            if opt.views == 2:
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_acc2
                ))
            elif opt.views == 3:
                epoch_acc3 = running_corrects3 / dataset_sizes['satellite']
                if is_weather_norm:
                    if opt.style_loss:
                        print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} StyleLoss: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc, epoch_acc2, epoch_acc3, epoch_style_loss
                        ))
                    else:
                        print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} '.format(
                            phase, epoch_loss, epoch_acc, epoch_acc2, epoch_acc3
                        ))
                else:
                    print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc, epoch_acc2, epoch_acc3
                    ))

            writer.add_scalar('Train Loss', epoch_loss, epoch + 1)
            if opt.circle:
                writer.add_scalar('Train Loss CE', epoch_loss_ce, epoch + 1)
                writer.add_scalar('Train Loss CIR', epoch_loss_cir, epoch + 1)
            if is_weather_norm:
                if opt.style_loss:
                    writer.add_scalar('style_loss', epoch_style_loss, epoch + 1)

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            if phase == 'train':
                scheduler.step()
                if pt_scheduler is not None:
                    pt_scheduler.step()

            last_model_wts = model.state_dict()
            if epoch > 50 and (epoch + 1) % 10 == 0:
                save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

######################################################################
# Draw Curve
######################################################################
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))

######################################################################
# Build model
######################################################################
if opt.views == 2:
    if opt.LPN:
        model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, LPN=True)
    else:
        model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share)
elif opt.views == 3:
    if opt.LPN:
        model = three_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, LPN=True, block=opt.block)
    else:
        model = three_view_net(
            len(class_names),
            droprate=opt.droprate,
            stride=opt.stride,
            pool=opt.pool,
            share_weight=opt.share,
            norm=opt.norm,
            adain=opt.adain,
            circle=opt.circle,
            btnk=opt.btnk,
            conv_norm=opt.conv_norm,
            config_xvlm=config_xvlm
        )

opt.nclasses = len(class_names)

# -----------------------------
# Load X-VLM pretrained checkpoint (open-source friendly)
# -----------------------------
model.xvlm.load_pretrained(
    args.xvlm_ckpt,
    config_xvlm,
    is_eval=False
)

print(model)

# For resume:
if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

if not opt.LPN:
    if hasattr(model, 'pt_model'):
        ignored_params = list(map(id, model.classifier.parameters()))
        ignored_params += list(map(id, model.pt_model.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        pt_res_params = list(map(id, model.pt_model.model.parameters()))
        pt_res_params += list(map(id, model.pt_model.classifier.parameters()))
        pt_mlp_params = filter(lambda p: id(p) not in pt_res_params, model.pt_model.parameters())

        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.classifier.parameters(), 'lr': opt.lr},
            {'params': model.pt_model.model.parameters(), 'lr': 0.1 * opt.lr},
            {'params': pt_mlp_params, 'lr': opt.lr},
            {'params': model.pt_model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    else:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

else:
    ignored_params = []
    for i in range(opt.block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        ignored_params += list(map(id, c.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optim_params = [{'params': base_params, 'lr': 0.1 * opt.lr}]
    for i in range(opt.block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        optim_params.append({'params': c.parameters(), 'lr': opt.lr})

    optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[120, 180, 210], gamma=0.1)

######################################################################
# Train and evaluate
######################################################################
log_dir = './log/' + opt.experiment_name
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

dir_name = os.path.join('./model', name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # record every run (open-source friendly: avoid relying on current working dir)
    def safe_copy(src: Path, dst: str):
        if src.is_file():
            copyfile(str(src), dst)
        else:
            print(f"[WARN] File not found, skip copying: {src}")

    safe_copy(REPO_ROOT / 'run.sh', os.path.join(dir_name, 'run.sh'))
    safe_copy(REPO_ROOT / 'train.py', os.path.join(dir_name, 'train.py'))
    safe_copy(REPO_ROOT / 'model.py', os.path.join(dir_name, 'model.py'))
    safe_copy(REPO_ROOT / 'resnet_adaibn.py', os.path.join(dir_name, 'resnet_adaibn.py'))

    # save opts
    with open(f'{dir_name}/opts.yaml', 'w') as fp:
        ruamel_yaml = YAML()
        ruamel_yaml.dump(vars(opt), fp)

model = model.to(device)
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()

if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 210

model = train_model(
    model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
    num_epochs=num_epochs, opt_pt=None, pt_scheduler=None
)
writer.close()
