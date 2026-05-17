import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class TibetanDataset(Dataset):
    def __init__(self, root_dir, transform=None, train_samples=None):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        
        self.global_samples = []
        self.global_class_samples = {cls_idx: [] for cls_idx in range(self.num_classes)}
        
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(self.image_extensions):
                    img_path = os.path.join(cls_dir, img_name)
                    self.global_samples.append((img_path, cls_idx))
                    self.global_class_samples[cls_idx].append(img_path)
        
        self.train_samples = train_samples if train_samples is not None else self.global_samples
        self.train_class_samples = {cls_idx: [] for cls_idx in range(self.num_classes)}
        for img_path, cls_idx in self.train_samples:
            self.train_class_samples[cls_idx].append(img_path)
        
        self.transform = transform

    def __len__(self):
        return len(self.global_samples)

    def __getitem__(self, idx):
        from utils.preprocess import preprocess_image
        img_path, label = self.global_samples[idx]
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"error")
            return torch.zeros(1, 30, 30), -1, img_path
        image = preprocess_image(image)
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

    def get_triplet_all_neg(self, max_neg_per_batch=5, hard_neg_ratio=0.5):
        valid_train_classes = [cls for cls in self.train_class_samples if len(self.train_class_samples[cls]) >= 2]
        if not valid_train_classes:
            raise ValueError("error")
        
        anchor_cls = random.choice(valid_train_classes)
        anchor_samples = self.train_class_samples[anchor_cls]
        anchor_path = random.choice(anchor_samples)
        positive_path = random.choice([p for p in anchor_samples if p != anchor_path])

        all_neg_cls = [c for c in self.train_class_samples if c != anchor_cls and len(self.train_class_samples[c]) >= 1]
        if not all_neg_cls:
            raise ValueError("error")
        
        def load_img(path):
            from utils.preprocess import preprocess_image
            img = Image.open(path).convert("L")
            img = preprocess_image(img)
            return self.transform(img) if self.transform else img

        anchor = load_img(anchor_path)
        positive = load_img(positive_path)
        negatives = [load_img(random.choice(self.train_class_samples[neg_cls])) for neg_cls in selected_neg_cls]
        return anchor, positive, negatives, selected_neg_cls

    def _get_simple_feat(self, img_path):
        from utils.preprocess import preprocess_image
        img = Image.open(img_path).convert("L")
        img = preprocess_image(img).resize((10, 10))
        return np.array(img).flatten()