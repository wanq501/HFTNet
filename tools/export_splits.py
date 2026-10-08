"""Write the image lists of every split in a dataset yaml to text files, so that the exact splits can be shared.

Usage:
    python tools/export_splits.py --data path/to/data.yaml --out splits
"""
import argparse, os
import yaml

IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def images(entry, root):
    paths = entry if isinstance(entry, list) else [entry]
    out = []
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.isdir(p):
            for dp, _, fs in os.walk(p):
                out += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1].lower() in IMG]
        elif p.endswith('.txt') and os.path.isfile(p):
            out += [l.strip() for l in open(p) if l.strip()]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='splits')
    a = ap.parse_args()
    d = yaml.safe_load(open(a.data))
    root = d.get('path') or os.path.dirname(os.path.abspath(a.data))
    os.makedirs(a.out, exist_ok=True)
    for split in ('train', 'val', 'test'):
        if not d.get(split):
            continue
        files = images(d[split], root)
        if not files:
            print(f'{split}: no images found, skipped')
            continue
        rel = [os.path.relpath(f, root) if os.path.isabs(f) else f for f in files]
        with open(os.path.join(a.out, f'{split}.txt'), 'w') as fh:
            fh.write('\n'.join(rel) + '\n')
        print(f'{split}: {len(rel)} images -> {os.path.join(a.out, split + ".txt")}')


if __name__ == '__main__':
    main()
