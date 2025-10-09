import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import random





def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
#getting one image of a folder.
def make_dataset_one(dir, class_to_idx, extensions, reverse=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            index = 0
            for fname in sorted(fnames, reverse=reverse):
                index += 1
                if has_file_allowed_extension(fname, extensions) and index == 36:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    break

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customData(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

class customData_one(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0, reverse=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_one(root, class_to_idx, IMG_EXTENSIONS, reverse)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)

def make_dataset_style(dir, class_to_idx, extensions, style='all'):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    if style == 'all':
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
                    else:
                        fstyle = fname.split('_')[2].split('.')[0]
                        if fstyle == style:
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)

    return images
class customData_style(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0, style = 'all'):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_style(root, class_to_idx, IMG_EXTENSIONS, style=style)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, save_augmented_dir=None):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.iaa_trans is not None:
                img = np.array(img)
                img = self.iaa_trans(image = img)
                img = Image.fromarray(img)
                # img = Image.fromarray(img)
                # img_aug.save('/home/wangtingyu/%s'%path.split('/')[-1])
            # else:
            #     img = np.array(img)
            #     # img = Image.fromarray(img)
                
                if self.save_augmented_dir:

                    # 计算相对路径： 相对于 root 的子路径
                    rel_path = os.path.relpath(path, self.root)
                    save_path = os.path.join(self.save_augmented_dir, rel_path)
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    img.save(save_path)  # 自动按原格式保存

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self):
        return len(self.imgs)


def make_dataset_selectID(dir, class_to_idx, extensions):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images[class_to_idx[target]].append(item)
    return images


class ImageFolder_iaa_save(Data.Dataset):
    def __init__(self, root, iaa_transform=None, save_augmented_dir='./augmented_images', loader=default_loader):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            img_aug = self.iaa_trans(image=np.array(img))
            
            img_aug = Image.fromarray(img_aug)
        else:
            img_aug = img

        # 保持原始目录结构保存增强图像
        relative_path = os.path.relpath(path, self.root)
        augmented_image_path = os.path.join(self.save_augmented_dir, relative_path)

        # 创建保存目录
        os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)

        # 保存增强后的图像
        img_aug.save(augmented_image_path)

        # 返回增强后图像路径和标签
        return augmented_image_path, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_selectID(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, norm='bn'):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs.keys()) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.norm = norm
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = random.choice(self.imgs[index])
            img = self.loader(path)
            if self.iaa_trans is not None:
                img = np.array(img)
                img = self.iaa_trans(image = img)
                # img_aug = Image.fromarray(img)
                # img_aug.save('/home/wangtyu/test_img/%s'%path.split('/')[-1])
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.norm == 'ada-ibn' or self.norm == 'spade':
                return img, target, 1
            else:
                return img, target

    def __len__(self):
        return len(self.imgs)

# class ImageFolder_iaa_multi_weather(Data.Dataset):
#     def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, iaa_weather_list=[], batchsize=8, shuffle=False, norm='bn', select=False):
#         classes, class_to_idx = find_classes(root)
#         IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
#         if select:
#             imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
#         else:
#             imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#         self.iaa_trans = iaa_transform
#         self.iaa_weather_list = iaa_weather_list
#         self.batch = batchsize
#         self.img_num = 0
#         self.shuffle = shuffle
#         self.norm = norm
#         self.select = select
#         # random.seed(1)
#     def __getitem__(self, index):
#             """
#             index (int): Index
#         Returns:tuple: (image, target) where target is class_index of the target class.
#             """
#             if self.select:
#                 path, target = random.choice(self.imgs[index])
#             else:
#                 path, target = self.imgs[index]
#             img = self.loader(path)
#             # if self.iaa_trans is not None:
#             #     img = np.array(img)
#             #     img = self.iaa_trans(image = img)
#             if self.iaa_weather_list:
#                 img = np.array(img)
#                 if self.shuffle:
#                     idx = random.choice(range(len(self.iaa_weather_list)))
#                     if idx == 0:
#                         img = img
#                     else:
#                         img = self.iaa_weather_list[idx](image=img)
#                         # img_aug = Image.fromarray(img)
#                         # img_aug.save('/home/wangtyu/test_img/%s'%path.split('/')[-1])

#                 else:
#                     idx = self.img_num // self.batch % (len(self.iaa_weather_list)+1)
#                     if idx == 0:
#                         img = img
#                         # img_aug = Image.fromarray(img)
#                         # img_aug.save('/Users/wongtyu/Downloads/University-Release/train/%s'%path.split('/')[-1])
#                     else:
#                         img = self.iaa_weather_list[idx-1](image = img)
#                         # img_aug = Image.fromarray(img)
#                         # img_aug.save('/Users/wongtyu/Downloads/University-Release/train/%s'%path.split('/')[-1])
#                     self.img_num += 1
#                     if self.img_num == len(self):
#                         self.img_num = 0
#             if self.iaa_trans is not None:
#                 img = self.iaa_trans(image = img)
#             if self.transform is not None:
#                 img = self.transform(img)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#             if self.norm == 'ada-ibn' or self.norm == 'spade':
#                 return img, target, idx+1
#             else:
#                 return img, target

#     def __len__(self):
#         return len(self.imgs)

class ImageFolder_iaa_multi_weather(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, iaa_weather_list=[], batchsize=8, shuffle=False,
                 norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if select:
            imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        else:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))
        
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
        # 注意：不在这里创建临时保存路径，caption匹配只需要全局索引

    def __getitem__(self, index):
        """
        index (int): 全局索引，对应于self.imgs中的样本顺序
        返回：
            如果 norm 为 'ada-ibn' 或 'spade'：
                (img, target, weather_index, global_index)
            否则：
                (img, target, global_index)
        """
        if self.select:
            path, target = random.choice(self.imgs[index])
        else:
            path, target = self.imgs[index]
        img = self.loader(path)

        # 天气增强处理：注意这里我们保留原有的计算方法，不做修改
        if self.iaa_weather_list:
            img = np.array(img)
            if self.shuffle:
                weather_idx = random.choice(range(len(self.iaa_weather_list)))
                if weather_idx == 0:
                    img = img
                else:
                    img = self.iaa_weather_list[weather_idx](image=img)
            else:
                weather_idx = self.img_num // self.batch % (len(self.iaa_weather_list) + 1)
                if weather_idx == 0:
                    img = img
                else:
                    img = self.iaa_weather_list[weather_idx - 1](image=img)
                self.img_num += 1
                if self.img_num == len(self):
                    self.img_num = 0
        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # 此处不改变原有的天气索引返回；如果有天气增强，则返回第三个值为天气索引（原代码使用 idx+1，这里用 weather_idx+1）
        if self.norm == 'ada-ibn' or self.norm == 'spade':
            # 如果有天气增强，返回计算得到的天气索引；否则返回 None
            weather_index = (weather_idx + 1) if self.iaa_weather_list else None
            return img, target, weather_index, index  # 新增全局索引 index
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)




### TODO: add the qwen image
# class PathFolder_qwen(Data.Dataset):
#     def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
#                  iaa_transform=None, iaa_weather_list=[], batchsize=8, shuffle=False, norm='bn', select=False):
#         classes, class_to_idx = find_classes(root)
#         IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
#         if select:
#             imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
#         else:
#             imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#         self.iaa_trans = iaa_transform
#         self.iaa_weather_list = iaa_weather_list
#         self.batch = batchsize
#         self.shuffle = shuffle
#         self.norm = norm
#         self.select = select
#         # 定义临时保存的目录
#         self.temp_root = os.path.join(root, "/home/wjh/project/MuseNet-master-test-vit/dataset/qwen")
#         os.makedirs(self.temp_root, exist_ok=True)

#     def __getitem__(self, index):
#         # 使用传入的 index 作为全局索引
#         if self.select:
#             path, target = random.choice(self.imgs[index])
#         else:
#             path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.iaa_weather_list:
#             img = np.array(img)
#             if self.shuffle:
#                 idx_sel = random.choice(range(len(self.iaa_weather_list)))
#                 if idx_sel != 0:
#                     img = self.iaa_weather_list[idx_sel](image=img)
#             else:
#                 idx_sel = index % (len(self.iaa_weather_list) + 1)
#                 if idx_sel != 0:
#                     img = self.iaa_weather_list[idx_sel - 1](image=img)
#         if self.iaa_trans is not None:
#             img = self.iaa_trans(image=img)
#         if self.transform is not None:
#             if isinstance(img, np.ndarray):
#                 img = Image.fromarray(img)
#             img = self.transform(img)

#         # 转换成 PIL Image 保存
#         if not isinstance(img, Image.Image):
#             from torchvision.transforms import ToPILImage
#             to_pil = ToPILImage()
#             img_to_save = to_pil(img)
#         else:
#             img_to_save = img

#         # 根据原始路径生成临时保存路径
#         rel_path = os.path.relpath(path, self.root)
#         temp_save_path = os.path.join(self.temp_root, rel_path)
#         os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
#         img_to_save.save(temp_save_path)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         # 直接返回 index 而不是 index+1
#         if self.norm == 'ada-ibn' or self.norm == 'spade':
#             return temp_save_path, target, index
#         else:
#             return temp_save_path, target

#     def __len__(self):
#         return len(self.imgs)


# class PathFolder_qwen(Data.Dataset):
#     def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
#                  iaa_transform=None, iaa_weather_list=[], batchsize=8, shuffle=False, norm='bn', select=False):
#         classes, class_to_idx = find_classes(root)
#         IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
#         if select:
#             imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
#         else:
#             imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
#         if len(imgs) == 0:
#             raise RuntimeError("Found 0 images in subfolders of: " + root +
#                                "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS))
        
#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#         self.iaa_trans = iaa_transform
#         self.iaa_weather_list = iaa_weather_list
#         self.batch = batchsize
#         self.shuffle = shuffle
#         self.norm = norm
#         self.select = select
#         # 定义临时保存的目录
#         self.temp_root = os.path.join(root, "/home/wjh/project/MuseNet-master-test-vit/dataset/qwen")
#         os.makedirs(self.temp_root, exist_ok=True)

#     def __getitem__(self, index):
#         # 使用传入的 index 作为全局索引
#         if self.select:
#             path, target = random.choice(self.imgs[index])
#         else:
#             path, target = self.imgs[index]
#         img = self.loader(path)

#         # 计算天气增强索引，保持原有逻辑不变
#         weather_index = None
#         if self.iaa_weather_list:
#             img = np.array(img)
#             if self.shuffle:
#                 idx_sel = random.choice(range(len(self.iaa_weather_list)))
#                 if idx_sel != 0:
#                     img = self.iaa_weather_list[idx_sel](image=img)
#                 weather_index = idx_sel  # 这里返回选中的索引（可能为0表示不做变化）
#             else:
#                 idx_sel = index % (len(self.iaa_weather_list) + 1)
#                 if idx_sel != 0:
#                     img = self.iaa_weather_list[idx_sel - 1](image=img)
#                 weather_index = idx_sel
#         # 其他转换操作
#         if self.iaa_trans is not None:
#             img = self.iaa_trans(image=img)
#         if self.transform is not None:
#             # 如果 img 为 numpy 数组，则先转为 PIL Image
#             if isinstance(img, np.ndarray):
#                 img = Image.fromarray(img)
#             img = self.transform(img)

#         # 保存临时图片
#         if not isinstance(img, Image.Image):
#             from torchvision.transforms import ToPILImage
#             to_pil = ToPILImage()
#             img_to_save = to_pil(img)
#         else:
#             img_to_save = img

#         rel_path = os.path.relpath(path, self.root)
#         temp_save_path = os.path.join(self.temp_root, rel_path)
#         os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
#         img_to_save.save(temp_save_path)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         # 根据 norm 类型决定返回值
#         if self.norm == 'ada-ibn' or self.norm == 'spade':
#             # 返回 (temp_save_path, target, weather_index, global_index)
#             return temp_save_path, target, weather_index, index
#         else:
#             return temp_save_path, target, index

#     def __len__(self):
#         return len(self.imgs)

class PathFolder_qwen(Data.Dataset):
    def __init__(self, root, transform=None, iaa_transform=None, loader=default_loader, iaa_weather_list=[],
                 batchsize=8, shuffle=True, norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS) if select else make_dataset(root, class_to_idx, IMG_EXTENSIONS)

        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in: " + root)

        self.root = root
        self.imgs = imgs
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

        # 定义临时保存的目录(保持你的路径不变，注意这里直接用你之前确认正确的路径)
        self.temp_root = os.path.join("/home/wjh/project/MuseNet-master-test-vit/dataset/qwen")
        os.makedirs(self.temp_root, exist_ok=True)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.img_num = 0  # 每个epoch重置计数

    def __getitem__(self, index):
        path, target = random.choice(self.imgs[index]) if self.select else self.imgs[index]
        img = self.loader(path)

        weather_idx = 0
        if self.iaa_weather_list:
            img = np.array(img)
            if self.current_epoch < 10:  # 前10个epoch，每张图片都顺序遍历所有天气
                weather_idx = (index % len(self.iaa_weather_list))
                if self.iaa_weather_list[weather_idx] is not None:
                    img = self.iaa_weather_list[weather_idx](image=img)
            else:  # 后续epoch，随机天气增强
                if self.shuffle:
                    weather_idx = random.choice(range(len(self.iaa_weather_list)))
                    if weather_idx != 0 and self.iaa_weather_list[weather_idx] is not None:
                        img = self.iaa_weather_list[weather_idx](image=img)
                else:
                    weather_idx = self.img_num // self.batchsize % (len(self.iaa_weather_list) + 1)
                    if weather_idx != 0 and self.iaa_weather_list[weather_idx - 1] is not None:
                        img = self.iaa_weather_list[weather_idx - 1](image=img)
                    self.img_num += 1
                    if self.img_num == len(self):
                        self.img_num = 0

        # 后续其他转换操作
        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)

        # 保存临时图片
        if not isinstance(img, Image.Image):
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            img_to_save = to_pil(img)
        else:
            img_to_save = img

        # 相对路径保存，以便与原路径结构一致
        rel_path = os.path.relpath(path, self.root)
        temp_save_path = os.path.join(self.temp_root, rel_path)
        os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
        img_to_save.save(temp_save_path)

        # 根据 norm 类型决定返回值
        if self.norm == 'ada-ibn' or self.norm == 'spade':
            return temp_save_path, target, weather_idx, index
        else:
            return temp_save_path, target, index

    def __len__(self):
        return len(self.imgs)




### TODO: add the weather condition
class ImageFolder_iaa_multi_weather_single(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, iaa_transform=None, iaa_weather_list=None, batchsize=8, shuffle=False, norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if select:
            imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        else:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))

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
        """
        index (int): 全局索引，对应于self.imgs中的样本顺序
        返回：
            如果 norm 为 'ada-ibn' 或 'spade'：
                (img, target, weather_index, global_index)
            否则：
                (img, target, global_index)
        """
        if self.select:
            path, target = random.choice(self.imgs[index])
        else:
            path, target = self.imgs[index]

        img = self.loader(path)
        img = np.array(img)

        idx = 0
        if self.iaa_weather_list:
            if self.shuffle:
                # 如果 iaa_weather_list 为单个增强器:
                if callable(self.iaa_weather_list):
                    img = self.iaa_weather_list(image=img)
                else:
                    # 如果 iaa_weather_list 为列表:
                    idx = random.randint(0, len(self.iaa_weather_list)-1)
                    img = self.iaa_weather_list[idx](image=img)
            else:
                idx = self.img_num // self.batch % (len(self.iaa_weather_list)+1)
                if idx != 0:
                    img = self.iaa_weather_list[idx-1](image=img)

                self.img_num += 1
                if self.img_num == len(self):
                    self.img_num = 0

        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.norm in ['ada-ibn', 'spade']:
            return img, target, idx+1, index  # 明确返回索引
        else:
            return img, target, index


    def __len__(self):
        return len(self.imgs)
