import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        feature_files = sorted(file_name for file_name in os.listdir(feature_dir) if file_name.endswith(".npy"))
        label_files = sorted(file_name for file_name in os.listdir(label_dir) if file_name.endswith(".npy"))
        if feature_files and label_files:
            common_files = sorted(set(feature_files).intersection(label_files), key=lambda item: int(item[:-4]))
            self.feature_files = common_files
            self.label_files = common_files
        else:
            self.feature_files = [f"{i}.npy" for i in range(1281167)]
            self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if features.ndim == 3:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        elif self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


class ConsolidatedCodeDataset(Dataset):
    def __init__(self, code_path):
        self.code_path = code_path
        self.codes = np.load(os.path.join(code_path, "codes.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(code_path, "labels.npy"), mmap_mode="r")
        if self.codes.ndim not in (2, 3):
            raise ValueError(f"Expected codes [N,T] or [N,A,T], got {self.codes.shape}")
        if self.labels.ndim != 1 or len(self.codes) != len(self.labels):
            raise ValueError(
                f"Code/label shape mismatch: codes={self.codes.shape}, labels={self.labels.shape}"
            )
        self.augmentations_per_image = int(self.codes.shape[1]) if self.codes.ndim == 3 else 1
        self.flip = self.augmentations_per_image > 1
        self.aug_feature_dir = None

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        features = self.codes[idx]
        if features.ndim == 2:
            aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
            features = features[aug_idx]
        features = np.array(features, dtype=np.int64, copy=True)
        label = np.array([self.labels[idx]], dtype=np.int64)
        return torch.from_numpy(features), torch.from_numpy(label)


class PairedCustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, replay_feature_dir, replay_label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.replay_feature_dir = replay_feature_dir
        self.replay_label_dir = replay_label_dir
        self.flip = False
        self.aug_feature_dir = None

        file_sets = [
            {name for name in os.listdir(path) if name.endswith(".npy")}
            for path in (feature_dir, label_dir, replay_feature_dir, replay_label_dir)
        ]
        common_files = set.intersection(*file_sets)
        if not common_files:
            raise ValueError("No common paired code/label files found")
        if any(files != common_files for files in file_sets):
            missing_counts = [len(files - common_files) for files in file_sets]
            raise ValueError(f"Paired code/label file sets differ: extras={missing_counts}")
        self.feature_files = sorted(common_files, key=lambda item: int(item[:-4]))
        example = np.load(os.path.join(self.feature_dir, self.feature_files[0]), mmap_mode="r")
        self.augmentations_per_image = int(example.shape[1]) if example.ndim == 3 else 1

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        file_name = self.feature_files[idx]
        features = np.load(os.path.join(self.feature_dir, file_name))
        replay_features = np.load(os.path.join(self.replay_feature_dir, file_name))
        if features.shape != replay_features.shape:
            raise ValueError(
                f"Paired feature shape mismatch for {file_name}: "
                f"{features.shape} != {replay_features.shape}"
            )
        if features.ndim == 3:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
            replay_features = replay_features[:, aug_idx]
        elif features.ndim != 2:
            raise ValueError(f"Expected paired codes with 2 or 3 dimensions, got {features.shape}")

        labels = np.load(os.path.join(self.label_dir, file_name))
        replay_labels = np.load(os.path.join(self.replay_label_dir, file_name))
        if not np.array_equal(labels, replay_labels):
            raise ValueError(f"Paired labels differ for {file_name}")
        return (
            torch.from_numpy(features),
            torch.from_numpy(replay_features),
            torch.from_numpy(labels),
        )


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    if os.path.isfile(os.path.join(args.code_path, "codes.npy")):
        return ConsolidatedCodeDataset(args.code_path)
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)


def build_imagenet_paired_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    replay_feature_dir = f"{args.replay_code_path}/imagenet{args.image_size}_codes"
    replay_label_dir = f"{args.replay_code_path}/imagenet{args.image_size}_labels"
    for path in (feature_dir, label_dir, replay_feature_dir, replay_label_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return PairedCustomDataset(feature_dir, label_dir, replay_feature_dir, replay_label_dir)
