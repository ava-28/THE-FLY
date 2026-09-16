"""Read the existing episode format, exposing only pixels to the CNN."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    files, seeds, hashes = set(), set(), set()
    for split in ('train', 'val', 'test'):
        records = manifest['splits'][split]
        if not records:
            raise ValueError(f'Empty split: {split}')
        for record in records:
            path = (root / record['file']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Episode path must stay inside the dataset directory.')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record['sha256']:
                raise ValueError(f'Episode changed since generation: {path}')
            if path in files or record['seed'] in seeds or digest in hashes:
                raise ValueError('Repeated episode, seed, or file content across manifest.')
            files.add(path)
            seeds.add(record['seed'])
            hashes.add(digest)
    return manifest


class VisionDataset(Dataset):
    """Small default dataset fits in RAM as uint8; convert each batch to float."""
    def __init__(self, root, split, manifest=None):
        root = Path(root)
        manifest = read_manifest(root) if manifest is None else manifest
        images, headings, ids, steps = [], [], [], []
        self.excluded_dark = self.excluded_blank = self.total = 0
        self.files = []
        for episode_id, record in enumerate(manifest['splits'][split]):
            self.files.append(record['file'])
            with np.load(root / record['file'], allow_pickle=False) as data:
                x, y, dark = data['images'], data['true_headings'], data['darkness']
                if x.dtype != np.uint8 or x.shape != (len(y), 64, 128, 3):
                    raise ValueError('Expected uint8 images with shape (T, 64, 128, 3).')
                if y.shape != (len(x),) or dark.shape != y.shape or dark.dtype != np.bool_:
                    raise ValueError('Invalid headings or darkness array.')
                if not np.isfinite(y).all():
                    raise ValueError('Nonfinite heading label.')
                # FlyCamera uses a uniform background; any spatial change means a landmark.
                visible = np.any(x != x[:, :1, :1, :], axis=(1, 2, 3))
                keep = ~dark & visible
                self.total += len(y)
                self.excluded_dark += int(dark.sum())
                self.excluded_blank += int((~dark & ~visible).sum())
                images.append(x[keep])
                headings.append(y[keep].astype(np.float32))
                ids.extend([episode_id] * int(keep.sum()))
                steps.extend(np.flatnonzero(keep).tolist())
        self.images = np.concatenate(images)
        self.headings = np.concatenate(headings)
        self.episode_ids = np.array(ids)
        self.steps = np.array(steps)
        if not len(self.images):
            raise ValueError(f'No illuminated landmark frames in {split}.')
        self.targets = np.stack([np.cos(self.headings), np.sin(self.headings)], axis=1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(self.targets[index].copy())

    def summary(self):
        return {'included': len(self), 'total': self.total,
                'excluded_dark': self.excluded_dark, 'excluded_blank': self.excluded_blank}
