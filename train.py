# -*- coding: utf-8 -*-
# train_xvlm_clean.py

import os
import sys
import time
import json
import copy
import random
import argparse
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ruamel.yaml import YAML

from project_utils import load_network, save_network
from model import two_view_net, three_view_net


proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


DEFAULT_XVLM_ROOT = os.path.join(proj_root, 'XVLM', 'X-VLM-master')
xvlm_root = os.environ.get('XVLM_ROOT', DEFAULT_XVLM_ROOT)
if xvlm_root not in sys.path:
    sys.path.insert(1, xvlm_root)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--name', default='two_view', type=str)
parser.add_argument('--experiment_name', default='debug', type=str)
parser.add_argument('--data_dir', default='./data/train', type=str)
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--stride', default=2, type=int)
parser.add_argument('--pad', default=10, type=int)
parser.add_argument('--h', default=384, type=int)
parser.add_argument('--w', default=384, type=int)
parser.add_argument('--views', default=3, type=int)
parser.add_argument('--erasing_p', default=0.0, type=float)
parser.add_argument('--color_jitter', action='store_true')
parser.add_argument('--DA', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--share', action='store_true')
parser.add_argument('--extra_Google', action='store_true')
parser.add_argument('--LPN', action='store_true')
parser.add_argument('--iaa', action='store_true')
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--norm', default='bn', type=str, choices=['bn', 'ibn', 'ada-ibn', 'spade'])
parser.add_argument('--adain', default='a', type=str)
parser.add_argument('--conv_norm', default='none', type=str, choices=['none', 'in', 'ln'])
parser.add_argument('--btnk', nargs='+', type=int, default=[1, 0, 1])
parser.add_argument('--pool', default='avg', type=str)
parser.add_argument('--droprate', default=0.75, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--warm_epoch', default=0, type=int)
parser.add_argument('--workers', default=8, type=int)

parser.add_argument('--xvlm_config',      type=str, default=os.path.join(xvlm_root, 'configs', 'config_swinB_384.json'))
parser.add_argument('--xvlm_text_config', type=str, default=os.path.join(xvlm_root, 'configs', 'config_bert.json'))
parser.add_argument('--xvlm_ckpt',        type=str, default=os.environ.get('XVLM_CKPT', ''), help='X-VLM pre-ckpt(.th),enviorment XVLM_CKPT')
parser.add_argument('--use_swin', action='store_true')
parser.add_argument('--use_clip_vit', action='store_true')
parser.add_argument('--use_roberta', action='store_true')

opt = parser.parse_args()

# ========= GPU & seed =========
gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if x.strip().isdigit() and int(x) >= 0]
if gpu_ids:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    print("gpu_ids:", gpu_ids)
    cudnn.benchmark = True

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if opt.seed > 0:
    print('[Seed]', opt.seed)
    seed_everything(opt.seed)

# ========= XVLM  =========
config_xvlm = None
if opt.xvlm_config and os.path.isfile(opt.xvlm_config):
    with open(opt.xvlm_config, 'r') as f:
        config_xvlm = json.load(f)

    config_xvlm['use_swin']      = opt.use_swin or True
    config_xvlm['use_clip_vit']  = opt.use_clip_vit
    config_xvlm['vision_config'] = opt.xvlm_config
    config_xvlm['image_res']     = opt.h
    config_xvlm['patch_size']    = 32


    if opt.xvlm_text_config and os.path.isfile(opt.xvlm_text_config):
        if opt.xvlm_text_config.endswith('.json'):
            _ = json.load(open(opt.xvlm_text_config, 'r'))
        else:
            YAML(typ='safe').load(open(opt.xvlm_text_config, 'r'))
        config_xvlm['use_roberta']  = opt.use_roberta
        config_xvlm['text_config']  = opt.xvlm_text_config
        config_xvlm['text_encoder'] = 'roberta-base' if opt.use_roberta else 'bert-base-uncased'
    else:
        config_xvlm['use_roberta']  = opt.use_roberta
        config_xvlm['text_config']  = ''
        config_xvlm['text_encoder'] = 'roberta-base' if opt.use_roberta else 'bert-base-uncased'


    config_xvlm['embed_dim']    = 256
    config_xvlm['temp']         = 0.07
    config_xvlm['max_tokens']   = 256
    config_xvlm['use_mlm_loss'] = False
    config_xvlm['use_bbox_loss']= True
else:
    print(f"[XVLM] No valid xvlm_config provided (currently: {opt.xvlm_config}), skipping X-VLM initialization.")

# ========= data aug =========
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy

transform_train_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]
transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

if opt.erasing_p > 0:
    transform_train_list += [RandomErasing(probability=opt.erasing_p, mean=[0, 0, 0])]
if opt.color_jitter:
    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
    transform_train_list = [jitter] + transform_train_list
    transform_satellite_list = [jitter] + transform_satellite_list
if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'satellite': transforms.Compose(transform_satellite_list),
}

# =========  opt.iaa =========
iaa_drone_transform = None
iaa_weather_list = None
transform_iaa_drone_tensor = None
if opt.iaa:
    import imgaug.augmenters as iaa
    print('[IAA] use iaa to augment drone images')
    iaa_drone_transform = iaa.Sequential([
        iaa.Resize({"height": opt.h, "width": opt.w}, interpolation=3),
        iaa.Pad(px=opt.pad, pad_mode="edge", keep_size=False),
        iaa.CropToFixedSize(width=opt.w, height=opt.h),
        iaa.Fliplr(0.5),
    ])
    iaa_weather_list = [
        None,
        iaa.Sequential([iaa.CloudLayer(
            intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
            alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10,
            alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35
        )]),
        iaa.Sequential([iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35)]),
        iaa.Sequential([
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]),
        iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)]),
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35
            ),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),
        ]),
        iaa.Sequential([iaa.MotionBlur(15, seed=17)]),
    ]
    transform_iaa_drone_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ========= DataLoader =========
from image_folder import (
    ImageFolder_iaa_selectID,
    ImageFolder_iaa_multi_weather,
)

def build_dataloaders(opt):
    image_datasets = {}
    image_datasets['satellite'] = datasets.ImageFolder(
        os.path.join(opt.data_dir, 'satellite'), data_transforms['satellite']
    )
    image_datasets['street'] = datasets.ImageFolder(
        os.path.join(opt.data_dir, 'street'), data_transforms['train']
    )

    google_dir = os.path.join(opt.data_dir, 'google')
    if os.path.isdir(google_dir):
        image_datasets['google'] = datasets.ImageFolder(google_dir, data_transforms['train'])

    # Drone
    if opt.iaa:
        image_datasets['drone1'] = ImageFolder_iaa_multi_weather(
            os.path.join(opt.data_dir, 'drone'),
            transform=transform_iaa_drone_tensor,
            iaa_transform=iaa_drone_transform,
            iaa_weather_list=iaa_weather_list,
            batchsize=opt.batchsize, shuffle=True, norm=opt.norm, select=True
        )
        drone_key = 'drone1'
    else:
        image_datasets['drone'] = datasets.ImageFolder(
            os.path.join(opt.data_dir, 'drone'), data_transforms['train']
        )
        drone_key = 'drone'

    # Dataloaders
    dataloaders = {
        k: torch.utils.data.DataLoader(
            v, batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers, pin_memory=True
        )
        for k, v in image_datasets.items()
    }
    dataset_sizes = {k: len(v) for k, v in image_datasets.items()}
    class_names = image_datasets['street'].classes
    return dataloaders, dataset_sizes, class_names, drone_key

dataloaders, dataset_sizes, class_names, drone_key = build_dataloaders(opt)
print('[dataset_sizes]', dataset_sizes)

# ========= model =========
if opt.views == 2:
    model = two_view_net(
        len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
        share_weight=opt.share, LPN=opt.LPN
    ) if opt.LPN else two_view_net(
        len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
        share_weight=opt.share
    )
else:
    model = three_view_net(
        len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
        share_weight=opt.share, LPN=True, block=6
    ) if opt.LPN else three_view_net(
        len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
        share_weight=opt.share, norm=opt.norm, adain=opt.adain,
        btnk=opt.btnk, conv_norm=opt.conv_norm, config_xvlm=(config_xvlm or {})
    )

# opt：load X-VLM pre
ckpt_path = opt.xvlm_ckpt.strip()
if ckpt_path and os.path.isfile(ckpt_path) and hasattr(model, 'xvlm') and hasattr(model.xvlm, 'load_pretrained'):
    print('[XVLM] load_pretrained:', ckpt_path)
    model.xvlm.load_pretrained(ckpt_path, (config_xvlm or {}), is_eval=False)
else:
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f'[XVLM][Warn] Specified weight does not exist: {ckpt_path}, loading skipped')
    else:
        print('[XVLM] skip pretrained')

# ========= Optimizer & Scheduler =========
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

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[120, 180, 210], gamma=0.1)

# ========= log =========
log_dir = './log/' + opt.experiment_name
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
dir_name = os.path.join('./model', opt.name)
os.makedirs(dir_name, exist_ok=True)

if not opt.resume:
    for src, dst in [
        ('./run.sh', os.path.join(dir_name, 'run.sh')),
        (__file__, os.path.join(dir_name, 'train.py')),
        ('./model.py', os.path.join(dir_name, 'model.py')),
        ('./resnet_adaibn.py', os.path.join(dir_name, 'resnet_adaibn.py')),
    ]:
        try:
            copyfile(src, dst)
        except Exception:
            pass
    with open(f'{dir_name}/opts.yaml', 'w') as fp:
        YAML().dump(vars(opt), fp)


if opt.resume:
    try:
        model, opt, start_epoch = load_network(opt.name, opt)
        print(f"[Resume] epoch={start_epoch} Recover")
    except Exception as e:
        print(f"[Resume][Warn] Recovery failed：{e}")
        start_epoch = 0
else:
    start_epoch = 0
if start_epoch >= 40:
    opt.lr *= 0.1

# ========= Training =========
model = model.to(device)
criterion = nn.CrossEntropyLoss()

def forward_three_views(model, x1, x2, x3, x4=None):
    if opt.views == 2:
        return model(x1, x2)
    if opt.extra_Google and x4 is not None:
        return model(x1, x2, x3, x4)
    return model(x1, x2, x3)

def compute_losses(outputs, labels_tuple):
    labels, labels2, labels3, labels4 = labels_tuple
    def logits_of(x):  #  (logits, feat)
        return x[0] if (isinstance(x, (tuple, list)) and len(x) == 2) else x

    if opt.views == 2:
        out1, out2 = outputs
        l1 = logits_of(out1); l2 = logits_of(out2)
        loss = criterion(l1, labels) + criterion(l2, labels2)
        return loss, [l1.argmax(1), l2.argmax(1)]
    else:
        if opt.extra_Google and len(outputs) == 4:
            out1, out2, out3, out4 = outputs
        else:
            out1, out2, out3 = outputs[:3]
            out4 = None

        l1 = logits_of(out1); l2 = logits_of(out2); l3 = logits_of(out3)
        loss = criterion(l1, labels) + criterion(l2, labels2) + criterion(l3, labels3)
        if out4 is not None:
            l4 = logits_of(out4)
            loss = loss + criterion(l4, labels4)


        if hasattr(model, 'loss_itc') and isinstance(model.loss_itc, torch.Tensor):
            loss = loss + model.loss_itc
        if hasattr(model, 'loss_itm') and isinstance(model.loss_itm, torch.Tensor):
            loss = loss + model.loss_itm

        preds = [l1.argmax(1), l2.argmax(1), l3.argmax(1)]
        return loss, preds

def train_one_epoch(model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    c1 = c2 = c3 = 0.0

    sat_loader = dataloaders['satellite']
    street_loader = dataloaders['street']
    drone_loader = dataloaders[drone_key]
    google_loader = dataloaders.get('google', None)

    for batch in zip(sat_loader, street_loader, drone_loader, google_loader or sat_loader):
        (x1, y1) = batch[0]
        (x2, y2) = batch[1]
        drone_batch = batch[2]
        if len(drone_batch) == 4:
            x3, y3, _, _ = drone_batch
        else:
            x3, y3 = drone_batch
        if google_loader:
            x4, y4 = batch[3]
        else:
            x4, y4 = None, None

        if x1.size(0) < opt.batchsize:
            continue

        x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
        y1 = y1.to(device); y2 = y2.to(device); y3 = y3.to(device)
        if x4 is not None:
            x4 = x4.to(device); y4 = y4.to(device)

        optimizer.zero_grad()
        outputs = forward_three_views(model, x1, x2, x3, x4)
        loss, preds = compute_losses(outputs, (y1, y2, y3, y4))
        loss.backward()
        optimizer.step()

        bs = x1.size(0)
        running_loss += loss.item() * bs
        if len(preds) > 0: c1 += (preds[0] == y1).float().sum().item()
        if len(preds) > 1: c2 += (preds[1] == y2).float().sum().item()
        if len(preds) > 2: c3 += (preds[2] == y3).float().sum().item()

    epoch_loss = running_loss / max(1, dataset_sizes['satellite'])
    acc1 = c1 / max(1, dataset_sizes['satellite'])
    acc2 = c2 / max(1, dataset_sizes['satellite'])
    log = f'Epoch {epoch} | Loss {epoch_loss:.4f}  Sat_Acc {acc1:.4f}  Street_Acc {acc2:.4f}'
    if opt.views == 3:
        acc3 = c3 / max(1, dataset_sizes['satellite'])
        log += f'  Drone_Acc {acc3:.4f}'
        writer.add_scalar('Train/Drone_Acc', acc3, epoch)
    print(log)

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Sat_Acc', acc1, epoch)
    writer.add_scalar('Train/Street_Acc', acc2, epoch)

def train(model, optimizer, scheduler, num_epochs=210):
    t0 = time.time()
    for epoch in range(start_epoch, num_epochs):
        # Simple warmup: linearly increase the temperature by epoch
        if opt.warm_epoch and epoch < opt.warm_epoch:
            scale = float(epoch + 1) / float(opt.warm_epoch)
            for pg in optimizer.param_groups:
                base_lr = pg.get('_base_lr', None)
                if base_lr is None:
                    pg['_base_lr'] = pg['lr']
                    base_lr = pg['_base_lr']
                pg['lr'] = base_lr * scale

        train_one_epoch(model, optimizer, epoch)
        if scheduler is not None:
            scheduler.step()

        if epoch > 180 and (epoch + 1) % 10 == 0:
            save_network(model, opt.name, epoch)

    dt = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(dt // 60, dt % 60))
    return model

# ========= Training Begin =========
model = model.to(device)
num_epochs = 210
model = train(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
writer.close()
