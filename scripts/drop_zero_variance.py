# scripts/drop_zero_variance.py
import os, glob, shutil, cv2
SRC = "data/assets"
DST = "data/assets_trash_zero"
os.makedirs(DST, exist_ok=True)

moved = 0
for p in sorted(glob.glob(os.path.join(SRC, "*.jpg")) + glob.glob(os.path.join(SRC, "*.png"))):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        continue
    im28 = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    var_lap = cv2.Laplacian(im28, cv2.CV_64F).var()
    std = im28.std()
    if var_lap == 0.0 or std == 0.0:  # image uniforme / totalement noire
        shutil.move(p, os.path.join(DST, os.path.basename(p)))
        moved += 1
print(f"Déplacé {moved} fichiers → {DST}")
