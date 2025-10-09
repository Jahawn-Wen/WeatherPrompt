# -*- coding: utf-8 -*-
# dataset_utils_clean.py

import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import torch.utils.data as Data
from torchvision import transforms
from torchvision import get_image_backend


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')


def has_file_allowed_extension(filename, extensions=IMG_EXTENSIONS):
    return filename.lower().endswith(tuple(extensions))


def find_classes(root_dir):
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(root_dir, class_to_idx, extensions=IMG_EXTENSIONS):
    images = []
    root_dir = os.path.expanduser(root_dir)
    for target in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for r, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(r, fname)
                    images.append((path, class_to_idx[target]))
    return images


def make_dataset_one(root_dir, class_to_idx, extensions=IMG_EXTENSIONS, reverse=False, pick_idx=36):
    images = []
    root_dir = os.path.expanduser(root_dir)
    for target in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for r, _, fnames in sorted(os.walk(d)):
            for i, fname in enumerate(sorted(fnames, reverse=reverse), start=1):
                if has_file_allowed_extension(fname, extensions) and i == pick_idx:
                    images.append((os.path.join(r, fname), class_to_idx[target]))
                    break
    return images


def make_dataset_style(root_dir, class_to_idx, extensions=IMG_EXTENSIONS, style='all'):
    images = []
    root_dir = os.path.expanduser(root_dir)
    for target in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for r, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if not has_file_allowed_extension(fname, extensions):
                    continue
                if style == 'all':
                    images.append((os.path.join(r, fname), class_to_idx[target]))
                else:
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        fstyle = parts[2].split('.')[0]
                        if fstyle == style:
                            images.append((os.path.join(r, fname), class_to_idx[target]))
    return images


def make_dataset_selectID(root_dir, class_to_idx, extensions=IMG_EXTENSIONS):
    buckets = defaultdict(list)
    root_dir = os.path.expanduser(root_dir)
    for target in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue
        for r, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    buckets[class_to_idx[target]].append((os.path.join(r, fname), class_to_idx[target]))
    return buckets



def _pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def _accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except Exception:
        return _pil_loader(path)


def default_loader(path):
    return _accimage_loader(path) if get_image_backend() == 'accimage' else _pil_loader(path)



class customData(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class customData_one(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0, reverse=False):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_one(root, class_to_idx, reverse=reverse)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class customData_style(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0, style='all'):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_style(root, class_to_idx, style=style)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, save_augmented_dir=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.save_augmented_dir = save_augmented_dir
        if save_augmented_dir:
            os.makedirs(save_augmented_dir, exist_ok=True)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.iaa_trans is not None:
            img_np = np.array(img)
            img_np = self.iaa_trans(image=img_np)
            img = Image.fromarray(img_np)

            if self.save_augmented_dir:
                rel_path = os.path.relpath(path, self.root)
                save_path = os.path.join(self.save_augmented_dir, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_save(Data.Dataset):
    def __init__(self, root, iaa_transform=None, save_augmented_dir='./augmented_images', loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.save_augmented_dir = save_augmented_dir

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.iaa_trans is not None:
            img = Image.fromarray(self.iaa_trans(image=np.array(img)))
        rel_path = os.path.relpath(path, self.root)
        out_path = os.path.join(self.save_augmented_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        return out_path, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_selectID(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, norm='bn'):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_selectID(root, class_to_idx)
        if len(imgs.keys()) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")
        self.root = root
        self.imgs = imgs  # dict[class_idx] -> list[(path, class_idx)]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.norm = norm

    def __getitem__(self, index):
        path, target = random.choice(self.imgs[index])
        img = self.loader(path)
        if self.iaa_trans is not None:
            img = self.iaa_trans(image=np.array(img))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.norm in ('ada-ibn', 'spade'):
            return img, target, 1
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_multi_weather(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, iaa_weather_list=None, batchsize=8, shuffle=False,
                 norm='bn', select=False):
        iaa_weather_list = iaa_weather_list or []
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_selectID(root, class_to_idx) if select else make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")

        self.root = root
        self.imgs = imgs  # select=True: dict；False: list
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batch = batchsize
        self.img_num = 0
        self.shuffle = shuffle
        self.norm = norm
        self.select = select

    def __getitem__(self, index):
        if self.select:
            path, target = random.choice(self.imgs[index])
        else:
            path, target = self.imgs[index]
        img = self.loader(path)

        weather_idx = 0
        if self.iaa_weather_list:
            img_np = np.array(img)
            if self.shuffle:
                weather_idx = random.randrange(len(self.iaa_weather_list))
                aug = self.iaa_weather_list[weather_idx]
                if aug is not None:
                    img_np = aug(image=img_np)
            else:
                weather_idx = self.img_num // self.batch % (len(self.iaa_weather_list) + 1)
                if weather_idx != 0:
                    aug = self.iaa_weather_list[weather_idx - 1]
                    if aug is not None:
                        img_np = aug(image=img_np)
                self.img_num = 0 if self.img_num + 1 == len(self) else self.img_num + 1
            img = img_np

        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.norm in ('ada-ibn', 'spade'):
            return img, target, (weather_idx + 1 if self.iaa_weather_list else None), index
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class PathFolder_qwen(Data.Dataset):
    def __init__(self, root, transform=None, iaa_transform=None, loader=default_loader, iaa_weather_list=None,
                 batchsize=8, shuffle=True, norm='bn', select=False, temp_root=None):
        iaa_weather_list = iaa_weather_list or []
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_selectID(root, class_to_idx) if select else make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in: {root}")

        self.root = root
        self.imgs = imgs  # select=True: dict；False: list
        self.transform = transform
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.norm = norm
        self.select = select
        self.loader = loader
        self.current_epoch = 0
        self.img_num = 0

        env_tmp = os.environ.get('QWEN_TMP_DIR', '').strip()
        self.temp_root = temp_root or (env_tmp if env_tmp else os.path.join(root, "_qwen_tmp"))
        os.makedirs(self.temp_root, exist_ok=True)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.img_num = 0 

    def __getitem__(self, index):
        path, target = (random.choice(self.imgs[index]) if self.select else self.imgs[index])
        img = self.loader(path)

        weather_idx = 0
        if self.iaa_weather_list:
            img_np = np.array(img)
            if self.current_epoch < 10:
                weather_idx = index % len(self.iaa_weather_list)
                aug = self.iaa_weather_list[weather_idx]
                if aug is not None:
                    img_np = aug(image=img_np)
            else:
                if self.shuffle:
                    weather_idx = random.randrange(len(self.iaa_weather_list))
                    aug = self.iaa_weather_list[weather_idx]
                    if aug is not None:
                        img_np = aug(image=img_np)
                else:
                    weather_idx = self.img_num // self.batchsize % (len(self.iaa_weather_list) + 1)
                    if weather_idx != 0:
                        aug = self.iaa_weather_list[weather_idx - 1]
                        if aug is not None:
                            img_np = aug(image=img_np)
                    self.img_num = 0 if self.img_num + 1 == len(self) else self.img_num + 1
            img = img_np

        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)


        if isinstance(img, Image.Image):
            img_to_save = img
        else:
            from torchvision.transforms import ToPILImage
            img_to_save = ToPILImage()(img)

        rel_path = os.path.relpath(path, self.root)
        out_path = os.path.join(self.temp_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img_to_save.save(out_path)

        if self.norm in ('ada-ibn', 'spade'):
            return out_path, target, weather_idx, index
        return out_path, target, index

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_multi_weather_single(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, iaa_weather_list=None, batchsize=8, shuffle=False,
                 norm='bn', select=False):
        iaa_weather_list = iaa_weather_list or []
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_selectID(root, class_to_idx) if select else make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {root}\nSupported: {', '.join(IMG_EXTENSIONS)}")

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batch = batchsize
        self.img_num = 0
        self.shuffle = shuffle
        self.norm = norm
        self.select = select

    def __getitem__(self, index):
        path, target = (random.choice(self.imgs[index]) if self.select else self.imgs[index])
        img = self.loader(path)
        img = np.array(img)

        idx = 0
        if self.iaa_weather_list:
            if callable(self.iaa_weather_list):
                img = self.iaa_weather_list(image=img)
            else:
                if self.shuffle:
                    idx = random.randint(0, len(self.iaa_weather_list) - 1)
                    img = self.iaa_weather_list[idx](image=img)
                else:
                    idx = self.img_num // self.batch % (len(self.iaa_weather_list) + 1)
                    if idx != 0:
                        img = self.iaa_weather_list[idx - 1](image=img)
                    self.img_num = 0 if self.img_num + 1 == len(self) else self.img_num + 1

        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.norm in ('ada-ibn', 'spade'):
            return img, target, idx + 1, index
        return img, target, index

    def __len__(self):
        return len(self.imgs)
