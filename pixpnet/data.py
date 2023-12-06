# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import namedtuple
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import get_dimensions

import pixpnet
from pixpnet import dataset_labels
from pixpnet.utils import get_logger

logger = get_logger(__name__)

ROOT_DIR = osp.dirname(osp.dirname(osp.realpath(pixpnet.__file__)))
DATA_DIR = osp.join(ROOT_DIR, "data")

DATA_MEAN_STD = {
    "IMAGENET": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "IMAGENETTE": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CUB200": ((0.4862, 0.5005, 0.4334), (0.2321, 0.2277, 0.2665)),
    "CUB200224": ((0.4862, 0.5005, 0.4334), (0.2321, 0.2277, 0.2665)),
    "CUB200BBOX": ((0.4862, 0.5005, 0.4334), (0.2321, 0.2277, 0.2665)),
    "CUB200224BBOX": ((0.4862, 0.5005, 0.4334), (0.2321, 0.2277, 0.2665)),
    "CARS224": ((0.4708, 0.4602, 0.4550), (0.2891, 0.2882, 0.2967)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}

DATA_CHANNELS = {}
# BIST
for _ds, (_mean, _std) in DATA_MEAN_STD.items():
    assert len(_mean) == len(_std)
    DATA_CHANNELS[_ds] = len(_mean)

DATA_NUM_OUTPUTS = {
    "IMAGENET": 1000,
    "IMAGENETTE": 10,
    "CUB200": 200,
    "CUB200224": 200,
    "CUB200BBOX": 200,
    "CUB200224BBOX": 200,
    "CARS224": 196,
    "CIFAR10": 10,
}

DATA_CROP_AND_PAD = {
    "IMAGENET": (224, None),
    "IMAGENETTE": (224, None),
    # https://github.com/chou141253/FGVC-PIM/blob/master/configs/AICUP_SwinT.yaml
    "CUB200": (384, None),
    "CUB200224": (384, None),
    "CUB200BBOX": (None, None),
    "CUB200224BBOX": (224, None),
    "CARS224": (None, None),
    "CIFAR10": (32, 4),
}

DATA_RESIZE = {
    "CUB200": 510,
    "CUB200224": 510,
    "CUB200224BBOX": 224,
    "CARS224": 224,
}

DATA_DOWNSAMPLE = {
    "CUB200224": 224,
    "CARS224": 224,
    "CUB200BBOX": 384,
}

LABEL_NAMES = {
    "IMAGENET": dataset_labels.imagenet_labels,
    "IMAGENETTE": dataset_labels.imagenette_labels,
    "CUB200": dataset_labels.cub200_labels,
    "CUB200224": dataset_labels.cub200_labels,
    "CARS224": dataset_labels.cars_labels,
    "CUB200BBOX": dataset_labels.cub200_labels,
    "CUB200224BBOX": dataset_labels.cub200_labels,
    "CIFAR10": dataset_labels.cifar10_labels,
}

DatasetMeta = namedtuple("DatasetMeta", "output_size,input_channels,input_size,label_names")


def _get_input_size(dataset):
    input_size = DATA_DOWNSAMPLE.get(dataset, DATA_CROP_AND_PAD.get(dataset, (None, None))[0])
    return input_size


# noinspection PyPep8Naming
class CUB200_2011(data.Dataset):
    """Adapted from
    https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/LICENSE"""

    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        loader=default_loader,
        download=True,
        yield_img_id=False,
        yield_orig_shape=False,
        crop_to_bbox=False,
        part_annotations=False,
    ):
        self.root = osp.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train
        self.crop_to_bbox = crop_to_bbox
        self.part_annotations = part_annotations
        self.yield_img_id = yield_img_id
        self.yield_orig_shape = yield_orig_shape

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use " "download=True to download it")

    def _load_metadata(self):
        df_images = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "images.txt"), sep=" ", names=["img_id", "filepath"]
        )
        df_image_class_labels = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "image_class_labels.txt"), sep=" ", names=["img_id", "target"]
        )
        df_train_test_split = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "train_test_split.txt"), sep=" ", names=["img_id", "is_training_img"]
        )

        self.data = df_images.merge(df_image_class_labels, on="img_id")
        self.data = self.data.merge(df_train_test_split, on="img_id")

        if self.crop_to_bbox:
            df_bbox = pd.read_csv(
                osp.join(self.root, "CUB_200_2011", "bounding_boxes.txt"), sep=" ", names=["img_id", "x", "y", "w", "h"]
            )
            df_bbox_int = df_bbox.astype(int)
            assert (df_bbox == df_bbox_int).all().all()
            self.data = self.data.merge(df_bbox_int, on="img_id")

        if self.part_annotations:
            # <image_id> <part_id> <x> <y> <visible>
            df_parts = pd.read_csv(
                osp.join(self.root, "CUB_200_2011", "parts", "part_locs.txt"),
                sep=" ",
                names=["img_id", "part_id", "x", "y", "visible"],
            )
            df_parts_int = df_parts.astype(int)
            assert (df_parts.astype(float) == df_parts_int).all().all()
            df_parts = df_parts_int
            df_parts["part_id_orig"] = df_parts["part_id"]
            df_parts["part_id"] -= 1
            df_parts["part_id"] = df_parts["part_id"].replace(
                {
                    0: 0,  # back
                    1: 1,  # beak
                    2: 2,  # belly
                    3: 3,  # breast
                    4: 4,  # crown
                    5: 5,  # forehead
                    6: 6,  # eye
                    7: 7,  # leg
                    8: 8,  # wing
                    9: 9,  # nape
                    10: 6,
                    11: 7,
                    12: 8,
                    13: 10,  # tail
                    14: 11,  # throat
                }
            )
            self.df_parts = df_parts
        else:
            self.df_parts = None

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:  # noqa
            return False

        for index, row in self.data.iterrows():
            filepath = osp.join(self.root, self.base_folder, row.filepath)
            if not osp.isfile(filepath):
                logger.warning(f"{filepath} is not a file...")
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            logger.info(f"{type(self).__name__} Files already downloaded and " f"verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(osp.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = osp.join(self.root, self.base_folder, sample.filepath)
        # Targets start at 1 by default, so shift to 0
        target = sample.target - 1
        img = self.loader(path)

        if self.yield_orig_shape:
            orig_shape = torch.tensor(get_dimensions(img))

        if self.crop_to_bbox:
            img = img.crop((sample.x, sample.y, sample.x + sample.w, sample.y + sample.h))

        if self.transform is not None:
            img = self.transform(img)

        if self.yield_img_id:
            if self.yield_orig_shape:
                return sample.img_id, orig_shape, img, target
            return sample.img_id, img, target
        elif self.yield_orig_shape:
            return orig_shape, img, target
        return img, target


class RepeatDataset(data.Dataset):
    def __init__(self, dataset, repeat_factor=2):
        self.dataset = dataset
        self.repeat_factor = repeat_factor

    def __len__(self):
        return round(len(self.dataset) * self.repeat_factor)

    def __getitem__(self, idx):
        if idx < 0:
            assert abs(idx) - 1 < len(self)
        else:
            assert idx < len(self)

        idx = idx % len(self.dataset)
        return self.dataset[idx]


class SubsetWithIdx(data.Subset):
    def __getitem__(self, idx):
        if isinstance(idx, list):
            orig_idxs = [self._get_orig_idx(i) for i in idx]
            return [*zip(orig_idxs, *zip(*self.dataset[orig_idxs]))]
        orig_idx = self._get_orig_idx(idx)
        return (orig_idx, *self.dataset[orig_idx])

    def _get_orig_idx(self, idx, dataset=None):
        if dataset is None:
            dataset = self.dataset
        if isinstance(dataset, data.Subset):
            return self._get_orig_idx(self.indices[idx], dataset.dataset)
        else:
            return self.indices[idx]


class ImageNette(datasets.ImageNet):
    # tench, English springer, cassette player, chain saw, church, French horn,
    # garbage truck, gas pump, golf ball, parachute
    class_subset = [
        "n01440764",
        "n02102040",
        "n02979186",
        "n03000684",
        "n03028079",
        "n03394916",
        "n03417042",
        "n03425413",
        "n03445777",
        "n03888257",
    ]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes, class_to_idx = super().find_classes(directory)
        classes = self.class_subset
        assert all(cls in class_to_idx for cls in classes)
        class_to_idx = {k: classes.index(k) for k in classes}
        return classes, class_to_idx


class ConditionalResize(transforms.Resize):
    """
    Resize transform but only if the input is smaller than the resize dims
    """

    @property
    def resize_height(self):
        return self.size if isinstance(self.size, int) else self.size[0]

    @property
    def resize_width(self):
        return self.size if isinstance(self.size, int) else self.size[1]

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        r_h, r_w = self.resize_height, self.resize_width
        if isinstance(img, torch.Tensor):
            h, w = img.shape[1:]
        else:  # PIL Image
            w, h = img.size
        if w < r_w or h < r_h:
            return super().forward(img)
        else:
            return img


def get_image_data_transform(normalize, augment, dataset):
    """
    Augmentations from SOTA (w/o additional data)
    https://github.com/chou141253/FGVC-PIM/blob/master/data/dataset.py
    """
    mean, std = DATA_MEAN_STD[dataset]
    crop_size, padding = DATA_CROP_AND_PAD[dataset]
    resize = DATA_RESIZE.get(dataset)
    downsample = DATA_DOWNSAMPLE.get(dataset)

    # ensure minimum image size is at least crop_size on both dimensions
    transform_list: List = [ConditionalResize(crop_size)] if crop_size and (not resize or resize < crop_size) else []

    if resize:
        if crop_size:
            transform_list.append(transforms.Resize(resize, transforms.InterpolationMode.BILINEAR))
        else:
            transform_list.append(transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR))

    if augment:
        transform_list.extend(
            [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))],
                    p=0.1,
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.RandomApply([transforms.RandomRotation(15)], p=1 / 3),
                transforms.RandomPerspective(p=1 / 3, distortion_scale=0.2),
                transforms.RandomApply([transforms.RandomAffine(0, shear=10)], p=1 / 3),
                transforms.RandomHorizontalFlip(0.5),
            ]
        )
        if crop_size:
            transform_list.append(
                transforms.RandomResizedCrop(crop_size)
                if not padding
                else transforms.RandomCrop(crop_size, padding=padding)
            )
    else:
        if crop_size:
            transform_list.extend([transforms.CenterCrop(crop_size)])

    if downsample:
        transform_list.append(transforms.Resize((downsample, downsample), transforms.InterpolationMode.BILINEAR))

    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def get_transform(dataset, normalize=False, augment=False):
    dataset = dataset.upper()
    if dataset not in DATA_MEAN_STD:
        logger.info(f"Will not apply any dataset transforms to {dataset}")
        transform = transforms.Compose([])
    else:
        transform = get_image_data_transform(normalize, augment, dataset)
    return transform


def get_datasets(config, normalize=True, augment=True, do_transforms=True, **kwargs):
    """Function to get pytorch dataloader objects
    Args:
        config:        Namespace, config
        normalize:     bool, Data normalization switch
        augment:       bool, Data augmentation switch
        do_transforms: bool, Data transforms switch
    Returns:
        ds_train:      Pytorch dataset object with training data
        ds_val:        Pytorch dataset object with validation data
        ds_test:       Pytorch dataset object with testing data
    """
    dataset_name = config.dataset.name.upper().replace("-", "")
    transform_train = get_transform(dataset_name, normalize, augment)
    transform_test = get_transform(dataset_name, normalize, False)

    data_root = config.dataset.root or DATA_DIR
    data_root = osp.join(data_root, dataset_name.lower())

    kwargs_train = {"root": data_root, "download": True}
    kwargs_test = kwargs_train.copy()

    if dataset_name == "IMAGENET":
        ds_cls = datasets.ImageNet
        kwargs_train.pop("download")
        kwargs_test.pop("download")
        kwargs_train["split"] = "train"
        kwargs_test["split"] = "val"
    elif dataset_name == "IMAGENETTE":
        ds_cls = ImageNette
        kwargs_train.pop("download")
        kwargs_test.pop("download")
        kwargs_train["split"] = "train"
        kwargs_test["split"] = "val"
    elif dataset_name == "CARS224":
        ds_cls = datasets.StanfordCars
        kwargs_train["split"] = "train"
        kwargs_test["split"] = "test"
    elif dataset_name.startswith("CUB200"):
        assert dataset_name in DATA_CROP_AND_PAD, dataset_name
        ds_cls = CUB200_2011
        if dataset_name in {"CUB200BBOX", "CUB200224BBOX"}:
            kwargs_train["crop_to_bbox"] = True
            kwargs_test["crop_to_bbox"] = True
        kwargs_train["train"] = True
        kwargs_test["train"] = False
    elif dataset_name == "CIFAR10":
        ds_cls = datasets.CIFAR10
        kwargs_train["train"] = True
        kwargs_test["train"] = False
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented in " f"get_datasets()")

    if do_transforms:
        kwargs_train["transform"] = transform_train
        kwargs_test["transform"] = transform_test

    # hasattr for legacy configs
    if hasattr(config.dataset, "crop_to_bbox") and config.dataset.crop_to_bbox:
        kwargs_train["crop_to_bbox"] = True
        kwargs_test["crop_to_bbox"] = True

    ds_train = ds_cls(**kwargs_train, **kwargs)
    ds_test = ds_cls(**kwargs_test, **kwargs)

    val_len = round(len(ds_train) * config.dataset.val_size)
    if val_len != 0:
        train_len = len(ds_train) - val_len
        ds_train, ds_val = data.random_split(ds_train, [train_len, val_len])
    else:
        ds_train = data.Subset(ds_train, [*range(len(ds_train))])
        ds_val = None

    if config.debug:
        n_debug_batches = 3
        logger.warning(
            "--config.debug enabled. Reducing the size of the " f"dataset to {n_debug_batches} batches worth."
        )
        debug_indices = [*range(n_debug_batches * config.train.batch_size)]
        ds_train = data.Subset(ds_train, debug_indices)
        debug_indices = [*range(n_debug_batches * config.test.batch_size)]
        ds_val = data.Subset(ds_val, debug_indices)
        ds_test = data.Subset(ds_test, debug_indices)

    ds_train_repeat = RepeatDataset(ds_train, repeat_factor=config.dataset.augment_factor)

    if config.dataset.needs_unaugmented:
        if do_transforms:
            kwargs_train["transform"] = transform_test
        # ds_train is Subset, reuse its indices
        ds_train_no_aug = SubsetWithIdx(ds_cls(**kwargs_train, **kwargs), ds_train.indices)
        return ds_train_repeat, ds_train_no_aug, ds_val, ds_test

    return ds_train_repeat, ds_val, ds_test


def get_metadata(config):
    dataset = config.dataset.name.upper().replace("-", "")
    metadata = DatasetMeta(
        output_size=DATA_NUM_OUTPUTS[dataset],
        input_channels=DATA_CHANNELS[dataset],
        input_size=_get_input_size(dataset),
        label_names=LABEL_NAMES.get(dataset),
    )
    return metadata
