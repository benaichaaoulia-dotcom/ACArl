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


class LITSDataset(Dataset):
    def __init__(
        self, imgs, segs, train: bool, child_classes: int, cluster_file: str = None
    ) -> None:
        super().__init__()
        self.imgs = imgs
        self.segs = segs
        self.train = train
        if child_classes != 0 and cluster_file:
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

        out = {
            "img": img,
            "seg": seg,
            "plab": plab,
            "idx": index,
            "fname": self.imgs[index],
        }

        if self.child_classes != 0:
            clab = torch.zeros(1 + self.child_classes)  # parent (1) + child classes
            clab[0] = 1.0  # parent class (foreground)
            cluster_id = int(self.clabs[index][0])
            clab[1 + cluster_id] = 1.0  # child class
            out["clab"] = clab

        return out


def get_dataset(
    data_path: str, child_classes: int, cluster_file: str
) -> tuple[Dataset, Dataset, Dataset]:
    # Try data_path/splits first
    splits_path = os.path.join(data_path, "splits")
    if not os.path.exists(splits_path):
        splits_path = "lits/splits"
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            # Only process if sample_name is a valid volume directory
            volume_dir = os.path.join(data_path, sample_name)
            if not os.path.isdir(volume_dir):
                continue
            img_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('img-') and f.endswith('.jpg')])
            seg_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('seg-') and f.endswith('.png')])
            img_list += [os.path.join(volume_dir, f) for f in img_files]
            seg_list += [os.path.join(volume_dir, f) for f in seg_files]
        return img_list, seg_list

    train_dataset = LITSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        True,
        child_classes,
        cluster_file,
    )
    val_dataset = LITSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        False,
        0,
        cluster_file,
    )
    test_dataset = LITSDataset(
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
            # Only process if sample_name is a valid volume directory
            volume_dir = os.path.join(data_path, sample_name)
            if not os.path.isdir(volume_dir):
                continue
            img_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('img-') and f.endswith('.jpg')])
            seg_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('seg-') and f.endswith('.png')])
            img_list += [os.path.join(volume_dir, f) for f in img_files]
            seg_list += [os.path.join(volume_dir, f) for f in seg_files]
        return img_list, seg_list

    dataset = LITSDataset(
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


class LITSSegDatasetWithConf(Dataset):
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
        
        # Use the index directly as the pseudo-label filename
        # The pseudo-labels are saved with simple numeric IDs
        lab_path = os.path.join(self.label_path, f"{index}.png")
        
        # Load pseudo-label
        lab = Image.open(lab_path).convert("F")
        
        # Load confidence map if available, otherwise create dummy ones
        if self.conf_path:
            conf_path = os.path.join(self.conf_path, f"{index}_conf.png")
            conf = Image.open(conf_path).convert("F")
        else:
            # Create a dummy confidence map (all 1.0) if not provided
            conf = Image.new("F", lab.size, 1.0)
        
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
        
        # Convert to float tensor for binary cross entropy
        # lab is already a tensor from augmentation, ensure it's float
        # Remove channel dimension if present: (1, H, W) -> (H, W)
        if lab.ndim == 3 and lab.shape[0] == 1:
            lab = lab.squeeze(0)
        lab = lab.float()
        
        idx = self.imgs[index].split("/")
        idx = idx[-2] + "-" + os.path.splitext(idx[-1])[0]
        
        return {
            "img": img,
            "lab": lab,
            "seg": seg,
            "conf": conf.unsqueeze(0) if conf.ndim == 2 else conf,  # Ensure (C, H, W) format
            "idx": idx,
            "fname": self.imgs[index],
        }


def get_seg_dataset_with_conf(data_path: str, lab_path: str, conf_path: str = None) -> tuple[Dataset, Dataset, Dataset]:
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

    train_dataset = LITSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        lab_path,
        conf_path,
        True,
    )
    val_dataset = LITSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        lab_path,
        conf_path,
        False,
    )
    test_dataset = LITSSegDatasetWithConf(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "test.txt"), "r")
        ),
        lab_path,
        conf_path,
        False,
    )

    return train_dataset, val_dataset, test_dataset


def get_seg_dataset(data_path: str, lab_path: str = None) -> tuple[Dataset, Dataset, Dataset]:
    """Load segmentation dataset. With lab_path, loads labels; without, loads only images."""
    if lab_path:
        # Use confidence-aware version if lab_path provided
        return get_seg_dataset_with_conf(data_path, lab_path, conf_path=None)
    
    # Load segmentation dataset for evaluation (without labels/confidence)
    splits_path = os.path.join(data_path, "splits")
    
    def get_files(sample_names):
        img_list = []
        seg_list = []
        for sample_name in sample_names:
            volume_dir = os.path.join(data_path, sample_name)
            if not os.path.isdir(volume_dir):
                continue
            img_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('img-') and f.endswith('.jpg')])
            seg_files = sorted([f for f in os.listdir(volume_dir) if f.startswith('seg-') and f.endswith('.png')])
            img_list += [os.path.join(volume_dir, f) for f in img_files]
            seg_list += [os.path.join(volume_dir, f) for f in seg_files]
        return img_list, seg_list

    train_dataset = LITSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "train.txt"), "r")
        ),
        True,
        0,
        None,
    )
    val_dataset = LITSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "val.txt"), "r")
        ),
        False,
        0,
        None,
    )
    test_dataset = LITSDataset(
        *get_files(
            sample_name.strip() for sample_name in open(os.path.join(splits_path, "test.txt"), "r")
        ),
        False,
        0,
        None,
    )

    return train_dataset, val_dataset, test_dataset
