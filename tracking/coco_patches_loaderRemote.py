"""
tracking/coco_patches_loader.py
DataLoader for pre-cropped COCO patch pairs produced by prepare_coco_patches.py.
Expects:
    template_dir/  — 128×128 crops, one file per sample
    search_dir/    — 256×256 crops (2.0× context), matching filenames
Augmentation (applied identically to both branches):
    - Random horizontal flip
    - Colour jitter (brightness/contrast/saturation only, no hue)
Both branches receive the same flip; colour jitter is applied independently
per branch (photometric variation is fine, spatial consistency must be kept).
"""
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

DATA_ROOT     = "/workspace/data/coco_patches"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CocoPatchesDataset(Dataset):
    def __init__(self, template_dir: str, search_dir: str, augment: bool = True):
        self.template_dir = template_dir
        self.search_dir   = search_dir
        self.augment      = augment
        all_files = sorted(os.listdir(template_dir))
        self.samples = [f for f in all_files if os.path.isfile(os.path.join(search_dir, f))]
        assert len(self.samples) > 0, (
            f"No matched pairs found.\n"
            f"  template_dir: {template_dir}\n"
            f"  search_dir:   {search_dir}"
        )
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        self.to_tensor    = transforms.ToTensor()
        self.normalize    = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname = self.samples[idx]
        z = Image.open(os.path.join(self.template_dir, fname)).convert("RGB")
        x = Image.open(os.path.join(self.search_dir,   fname)).convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                z = TF.hflip(z)
                x = TF.hflip(x)
            z = self.color_jitter(z)
            x = self.color_jitter(x)
        z = self.normalize(self.to_tensor(z))
        x = self.normalize(self.to_tensor(x))
        return z, x

def build_loader(
    batch_size: int,
    num_workers: int,
    augment: bool = True,
    template_dir: str = os.path.join(DATA_ROOT, "template"),
    search_dir: str = os.path.join(DATA_ROOT, "search"),
) -> DataLoader:
    dataset = CocoPatchesDataset(template_dir, search_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # keeps workers alive between epochs
    )