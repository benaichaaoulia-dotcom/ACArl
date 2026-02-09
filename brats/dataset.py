from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops
import torch.nn.functional as F
import random
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pickle

pic_size = 256


def trim(img: Image.Image, seg: Image.Image):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox), seg.crop(bbox)
    else:
        return img, seg


def aug(img: Image.Image, seg: Image.Image, lab: Image.Image = None):
    img, seg = trim(img, seg)
    rotate_angle = random.randrange(-20, 20)
    img = TF.rotate(img, rotate_angle)
    seg = TF.rotate(seg, rotate_angle)
    if lab:
        lab = lab.resize(img.size)
        lab = TF.rotate(lab, rotate_angle)
    params = T.RandomResizedCrop(pic_size).get_params(
        img, scale=(0.5, 1.0), ratio=(0.7, 1.3)
    )
    img = TF.crop(img, *params)
    seg = TF.crop(seg, *params)
    if lab:
        lab = TF.crop(lab, *params)
    jitter = T.ColorJitter(brightness=0.5, contrast=0.5)
    img = jitter(img)
    if random.random() > 0.5:
        img = TF.hflip(img)
        seg = TF.hflip(seg)
        if lab:
            lab = TF.hflip(lab)
    img = TF.to_tensor(img.resize((pic_size, pic_size)))
    seg = TF.to_tensor(seg.resize((pic_size, pic_size), Image.BILINEAR))
    seg[seg > 0.5] = 1
    if lab:
        lab = TF.to_tensor(lab.resize((pic_size, pic_size), Image.BILINEAR))
        lab[lab > 0.5] = 1
        return img, seg, lab
    return img, seg


def no_aug(img: Image.Image, seg: Image.Image, lab: Image.Image = None):
    img, seg = trim(img, seg)
    img = img.resize((pic_size, pic_size))
    seg = seg.resize((pic_size, pic_size), Image.NEAREST)
    if lab:
        lab = lab.resize((pic_size, pic_size), Image.NEAREST)
    img, seg = TF.to_tensor(img), TF.to_tensor(seg)
    if lab:
        lab = TF.to_tensor(lab)
        return img, seg, lab
    return img, seg


class BraTSDataset(Dataset):
    def __init__(
        self, imgs, segs, train: bool, child_classes: int, cluster_file: str
    ) -> None:
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.train = train
        if child_classes != 0:
            with open(cluster_file, "rb") as f:
                self.clabs = pickle.load(f)
        self.child_classes = child_classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: Any) -> Any:
        # Fast image loading with minimal conversions
        img = Image.open(self.imgs[index]).convert("L")
        seg = Image.open(self.segs[index]).convert("F")

        plab = torch.zeros(1).float()
        # Vectorized check for faster computation
        seg_np = np.array(seg)
        plab[0] = 1.0 if seg_np.sum() != 0 else 0.0

        img, seg = aug(img, seg) if self.train else no_aug(img, seg)
        seg[seg != 0] = 1

        idx = self.imgs[index].split("/")
        idx = idx[-2] + "-" + os.path.splitext(idx[-1])[0]

        if self.child_classes != 0:
            clab = torch.zeros(self.child_classes + 1).float()
            if plab[0] != 0:
                clab[int(self.clabs[idx][0]) + 1] = 1
            else:
                clab[0] = 1

            return {
                "img": img,
                "plab": plab,
                "clab": clab,
                "seg": seg,
                "idx": idx,
                "fname": self.imgs[index],
            }
        else:
            return {
                "img": img,
                "plab": plab,
                "seg": seg,
                "idx": idx,
                "fname": self.imgs[index],
            }


def get_dataset(
    data_path: str, child_classes: int, cluster_file: str
) -> tuple[Dataset, Dataset]:
    # Try data_path/splits first, fallback to brats/splits
    splits_path = os.path.join(data_path, "splits")
    if not os.path.exists(splits_path):
        splits_path = "brats/splits"
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            num_files = len(os.listdir(os.path.join(data_path, sample_name))) // 2
            img_list += [
                os.path.join(data_path, sample_name, f"img-{str(i).zfill(3)}.jpg")
                for i in range(num_files)
            ]
            seg_list += [
                os.path.join(data_path, sample_name, f"seg-{str(i).zfill(3)}.png")
                for i in range(num_files)
            ]
        return img_list, seg_list

    train_dataset = BraTSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        True,
        child_classes,
        cluster_file,
    )
    val_dataset = BraTSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        False,
        0,
        cluster_file,
    )
    test_dataset = BraTSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "test.txt"), "r")
        ),
        False,
        0,
        cluster_file,
    )

    return train_dataset, val_dataset, test_dataset


def get_all_dataset(data_path: str, child_classes: int, cluster_file: str) -> Dataset:
    splits_path = os.path.join(data_path, "splits")
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            num_files = len(os.listdir(os.path.join(data_path, sample_name))) // 2
            img_list += [
                os.path.join(data_path, sample_name, f"img-{str(i).zfill(3)}.jpg")
                for i in range(num_files)
            ]
            seg_list += [
                os.path.join(data_path, sample_name, f"seg-{str(i).zfill(3)}.png")
                for i in range(num_files)
            ]
        return img_list, seg_list

    dataset = BraTSDataset(
        *get_files(
            sample_name.strip()
            for sample_name in (
                list(open(os.path.join(splits_path, "train.txt"), "r"))
                + list(open(os.path.join(splits_path, "val.txt"), "r"))
                + list(open(os.path.join(splits_path, "test.txt"), "r"))
            )
        ),
        False,
        child_classes,
        cluster_file,
    )

    return dataset


class BraTSSegDataset(Dataset):
    def __init__(self, imgs, segs, label_path: str, train: bool) -> None:
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.train = train
        self.label_path = label_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(self.imgs[index]).convert("L")
        seg = Image.open(self.segs[index]).convert("F")
        idx = self.imgs[index].split("/")
        idx = idx[-2] + "-" + os.path.splitext(idx[-1])[0]
        if self.label_path is not None:
            lab = Image.open(os.path.join(self.label_path, f"{idx}.png")).convert("F")
            img, seg, lab = aug(img, seg, lab) if self.train else no_aug(img, seg, lab)
            seg[seg != 0] = 1
            lab[lab != 0] = 1
            lab = (
                F.one_hot(lab.squeeze(0).long(), num_classes=2).permute(2, 0, 1).float()
            )
            return {
                "img": img,
                "lab": lab,
                "seg": seg,
                "idx": idx,
                "fname": self.imgs[index],
            }
        else:
            img, seg = aug(img, seg) if self.train else no_aug(img, seg)
            seg[seg != 0] = 1
            return {
                "img": img,
                "seg": seg,
                "idx": idx,
                "fname": self.imgs[index],
            }


def get_seg_dataset(data_path: str, lab_path: str = None) -> tuple[Dataset, Dataset]:
    splits_path = os.path.join(data_path, "splits")
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            num_files = len(os.listdir(os.path.join(data_path, sample_name))) // 2
            img_list += [
                os.path.join(data_path, sample_name, f"img-{str(i).zfill(3)}.jpg")
                for i in range(num_files)
            ]
            seg_list += [
                os.path.join(data_path, sample_name, f"seg-{str(i).zfill(3)}.png")
                for i in range(num_files)
            ]
        return img_list, seg_list

    train_dataset = BraTSSegDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        lab_path,
        True,
    )
    val_dataset = BraTSSegDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        lab_path,
        False,
    )
    test_dataset = BraTSSegDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "test.txt"), "r")
        ),
        lab_path,
        False,
    )

    return train_dataset, val_dataset, test_dataset


class BraTSSegDatasetWithConf(Dataset):
    """Dataset class that loads images, pseudo-labels, and confidence maps for U-Net training."""
    
    def __init__(self, imgs, segs, label_path: str, conf_path: str, train: bool) -> None:
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.train = train
        self.label_path = label_path
        self.conf_path = conf_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(self.imgs[index]).convert("L")
        seg = Image.open(self.segs[index]).convert("F")
        idx = self.imgs[index].split("/")
        idx = idx[-2] + "-" + os.path.splitext(idx[-1])[0]
        
        # Load pseudo-label
        lab = Image.open(os.path.join(self.label_path, f"{idx}.png")).convert("F")
        
        # Load confidence map (with _conf suffix)
        conf = Image.open(os.path.join(self.conf_path, f"{idx}_conf.png")).convert("F")
        
        # Apply augmentations
        if self.train:
            img, seg, lab = aug(img, seg, lab)
            # Resize confidence to match
            conf = conf.resize((img.shape[1], img.shape[2]) if len(img.shape) > 2 else img.size, Image.BILINEAR)
        else:
            img, seg, lab = no_aug(img, seg, lab)
            conf = conf.resize((img.shape[1], img.shape[2]) if len(img.shape) > 2 else img.size, Image.BILINEAR)
        
        # Process labels
        seg[seg != 0] = 1
        lab[lab != 0] = 1
        conf = torch.from_numpy(np.array(conf)).float() / 255.0
        
        # Don't one-hot encode here - let the training script handle it
        # Just ensure lab is long type for training
        if len(lab.shape) == 3:
            lab = lab.squeeze(0)
        lab = lab.long()
        
        return {
            "img": img,
            "lab": lab,
            "seg": seg,
            "conf": conf.unsqueeze(0) if conf.ndim == 2 else conf,  # Ensure (C, H, W) format
            "idx": idx,
            "fname": self.imgs[index],
        }


def get_seg_dataset_with_conf(data_path: str, lab_path: str, conf_path: str = None) -> tuple[Dataset, Dataset]:
    """Load segmentation dataset with confidence maps from ACArL pseudo-label generation."""
    splits_path = os.path.join(data_path, "splits")
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            num_files = len(os.listdir(os.path.join(data_path, sample_name))) // 2
            img_list += [
                os.path.join(data_path, sample_name, f"img-{str(i).zfill(3)}.jpg")
                for i in range(num_files)
            ]
            seg_list += [
                os.path.join(data_path, sample_name, f"seg-{str(i).zfill(3)}.png")
                for i in range(num_files)
            ]
        return img_list, seg_list

    train_dataset = BraTSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        lab_path,
        conf_path,
        True,
    )
    val_dataset = BraTSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        lab_path,
        conf_path,
        False,
    )
    test_dataset = BraTSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "test.txt"), "r")
        ),
        lab_path,
        conf_path,
        False,
    )

    return train_dataset, val_dataset, test_dataset
