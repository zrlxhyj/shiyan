"""Import existing local YOLO clients; no verified sequence-disjoint claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/partition.json'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    roots = [args.root / f'client{i}' / 'VisDrone2019-DET-train' for i in range(1, 6)]
    roots.append(args.root / 'VisDrone2019-DET-val')
    files = [sorted(p for p in (r / 'images').iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')) for r in roots]
    parts = [[] for _ in roots]
    seen = {}
    duplicates = []
    record_id = 0
    degenerate_boxes = 0
    # Retain duplicates in smaller clients first to preserve scarce client data.
    order = sorted(range(5), key=lambda i: (len(files[i]), i)) + [5]
    for i in order:
        for image in files[i]:
            destination = i
            label = roots[i] / 'labels' / (image.stem + '.txt')
            text = label.read_text(encoding='utf-8-sig')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != i:
                owner, old_texts = seen[digest]
                if i == 5 or owner == 5:
                    raise ValueError(f'Train/validation image overlap: {image}')
                if text in old_texts:
                    duplicates.append({'image': str(image), 'kept_client': owner + 1,
                                       'removed_client': i + 1})
                    continue
                # Identical frames with different annotations remain separate records,
                # but must stay together in the same client to prevent content leakage.
                destination = owner
            seen.setdefault(digest, (destination, set()))[1].add(text)
            with Image.open(image) as im:
                width, height = im.size
            objects = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5:
                    raise ValueError(f'Expected YOLO detection label: {label}: {line}')
                category, x, y, w, h = values
                if category != int(category) or not 0 <= category < 10:
                    raise ValueError(f'Invalid class: {label}: {line}')
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError(f'Invalid normalized box: {label}: {line}')
                if w == 0 or h == 0:
                    degenerate_boxes += 1
                    continue
                x1, y1 = max(0., (x-w/2)*width), max(0., (y-h/2)*height)
                x2, y2 = min(width, (x+w/2)*width), min(height, (y+h/2)*height)
                if x2 > x1 and y2 > y1:
                    objects.append({'bbox': [x1, y1, x2-x1, y2-y1],
                                    'category_id': int(category)+1})
            record_id += 1
            parts[destination].append({'id': record_id, 'image': str(image.resolve()),
                             'width': width, 'height': height,
                             'group': 'image-sha256:' + digest,
                             'objects': objects, 'ignored': []})
    eval_groups = {}
    for record in parts[5]:
        eval_groups.setdefault(record['group'], []).append(record)
    group_keys = sorted(eval_groups)
    random.Random(args.seed).shuffle(group_keys)
    midpoint = len(group_keys) // 2
    if any(not p for p in parts[:5]) or midpoint == 0:
        raise ValueError('Empty client or evaluation split')
    hist = []
    for records in parts[:5]:
        counts = Counter(o['category_id'] for r in records for o in r['objects'])
        hist.append([counts[c] for c in range(1, 11)])
    manifest = {
        'protocol': 'Existing local YOLO clients; content-deduplicated; image-random validation halves',
        'limitations': ['Sequence IDs unverified; group represents image content, not sequence',
                        'YOLO conversion does not recover ignored regions or occlusion metadata',
                        'Existing client assignment retained except duplicate removal; no Dirichlet alpha inferred'],
        'seed': args.seed, 'source_root': str(args.root.resolve()),
        'clients': parts[:5],
        'dev': [r for k in group_keys[:midpoint] for r in eval_groups[k]],
        'test': [r for k in group_keys[midpoint:] for r in eval_groups[k]],
        'statistics': {'source_images': [len(p) for p in files],
                       'images': [len(p) for p in parts[:5]], 'class_histograms': hist,
                       'duplicates_removed': len(duplicates),
                       'degenerate_boxes_skipped': degenerate_boxes,
                       'dev_images': sum(len(eval_groups[k]) for k in group_keys[:midpoint]),
                       'test_images': sum(len(eval_groups[k]) for k in group_keys[midpoint:])},
        'duplicate_removals': duplicates,
    }
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing manifest: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['statistics'], indent=2))


if __name__ == '__main__':
    main()
