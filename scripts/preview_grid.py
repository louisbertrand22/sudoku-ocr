# scripts/preview_grid.py
import os, glob, random, cv2
import numpy as np

ROOT = "data/assets"  # un seul dossier avec 0_*.jpg, 1_*.jpg, ...

def load_by_class(root):
    files = sorted(glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png")))
    by_cls = {i: [] for i in range(10)}
    for p in files:
        base = os.path.basename(p)
        try:
            lab = int(base.split("_", 1)[0])
        except:
            continue
        if lab in by_cls: by_cls[lab].append(p)
    return by_cls

def make_grid(paths, n=25, size=28):
    paths = paths if len(paths) <= n else random.sample(paths, n)
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        imgs.append(im)
    if not imgs: return None
    k = int(np.ceil(np.sqrt(len(imgs))))
    H, W = k*size, k*size
    grid = np.zeros((H, W), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, k)
        grid[r*size:(r+1)*size, c*size:(c+1)*size] = im
    return grid

def main():
    by_cls = load_by_class(ROOT)
    os.makedirs("data/outputs/debug", exist_ok=True)
    for cls, paths in by_cls.items():
        grid = make_grid(paths)
        if grid is None: continue
        out = f"data/outputs/debug/preview_class_{cls}.png"
        cv2.imwrite(out, grid)
        print("saved", out)

if __name__ == "__main__":
    main()
