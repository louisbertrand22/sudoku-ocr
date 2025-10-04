# scripts/auto_filter.py
import os, glob, shutil, cv2, numpy as np

SRC = "data/assets"
DST = "data/assets_trash"
THRESH_LAP = 10.0   # <10 = très flou
THRESH_STD = 0.04   # <0.04 = trop peu de contraste (sur [0..1])

os.makedirs(DST, exist_ok=True)

def bad(img):
    im = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    var_lap = cv2.Laplacian(im, cv2.CV_64F).var()
    std = im.std() / 255.0
    return (var_lap < THRESH_LAP) or (std < THRESH_STD)

def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.jpg")) + glob.glob(os.path.join(SRC, "*.png")))
    moved = 0
    for p in files:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if bad(im):
            shutil.move(p, os.path.join(DST, os.path.basename(p)))
            moved += 1
    print(f"Déplacé {moved} fichiers bruyés/illisibles → {DST}")

if __name__ == "__main__":
    main()
