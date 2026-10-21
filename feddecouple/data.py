import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

def read_labels(path, width, height):
    objects, ignored = [], []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        vals = [float(x.strip()) for x in line.rstrip(',').split(',')]
        if len(vals) < 8:
            raise ValueError(f'Invalid VisDrone annotation: {path}: {line}')
        x, y, w, h, score, category = vals[:6]
        x1, y1 = max(0.,x), max(0.,y)
        x2, y2 = min(float(width), x+w), min(float(height),y+h)
        if x2 <= x1 or y2 <= y1:
            continue
        box = [x1,y1,x2-x1,y2-y1]
        if score <= 0 or int(category) not in range(1,11):
            ignored.append(box)
        else:
            objects.append({'bbox': box, 'category_id': int(category), 'occlusion': int(vals[7])})
    return objects, ignored

class VisDrone(Dataset):
    def __init__(self, records, size=640, train=False):
        self.records, self.size, self.train = records, size, train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record['image']).convert('RGB')
        width, height = image.size
        boxes = torch.tensor([o['bbox'] for o in record['objects']], dtype=torch.float32).reshape(-1,4)
        labels = torch.tensor([o['category_id']-1 for o in record['objects']], dtype=torch.long)
        # Declared unified preprocessing: mask excluded annotation regions in train.
        if self.train:
            draw = ImageDraw.Draw(image)
            for x,y,w,h in record['ignored']:
                draw.rectangle((x,y,x+w,y+h), fill=(114,114,114))
        if len(boxes):
            boxes[:, :2] += boxes[:, 2:]/2
            boxes /= torch.tensor([width,height,width,height])
        image = image.resize((self.size,self.size), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32)/255.
        # One augmentation per batch item; shared by teacher and all student forwards.
        if self.train and torch.rand(()).item() < .5:
            array = array[:, ::-1].copy()
            boxes[:,0] = 1-boxes[:,0]
        tensor = torch.from_numpy(array.copy()).permute(2,0,1)
        target = {'boxes': boxes, 'labels': labels,
                  'image_id': torch.tensor(record['id']), 'orig_size': torch.tensor([height,width])}
        return tensor, target

def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

def load_manifest(path, clients):
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if len(manifest['clients']) != clients:
        raise ValueError('Manifest client count differs from config')
    partitions = list(manifest['clients'])+[manifest['dev'],manifest['test']]
    seen_images, seen_groups = set(), set()
    for part in partitions:
        if not part:
            raise ValueError('Empty client or evaluation partition')
        images = {r['image'] for r in part}
        groups = {r['group'] for r in part}
        if len(images) != len(part) or seen_images & images or seen_groups & groups:
            raise ValueError('Image/sequence leakage between partitions')
        seen_images |= images; seen_groups |= groups
    if 'test_domains' in manifest:
        allowed = {r['image']: r for r in manifest['test']}
        domain_seen = set()
        for domain, records in manifest['test_domains'].items():
            for r in records:
                if r['image'] not in allowed or r != allowed[r['image']]:
                    raise ValueError('Domain evaluation records must be unchanged members of the final set')
                if r['image'] in domain_seen:
                    raise ValueError('Domain evaluation subsets must be disjoint')
                domain_seen.add(r['image'])
    return manifest
