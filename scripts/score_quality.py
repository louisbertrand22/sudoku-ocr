# scripts/score_quality.py
import os, glob, cv2, numpy as np

ROOT = "data/assets"

def scores(img):
    # img en niveaux de gris [0..255]
    var_lap = cv2.Laplacian(img, cv2.CV_64F).var()  # flou (bas = flou)
    mean = float(img.mean())                        # luminosité
    std = float(img.std())                          # contraste (bas = plat)
    return var_lap, mean, std

def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*.jpg")) + glob.glob(os.path.join(ROOT, "*.png")))
    vals = []
    for p in files:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
        var_lap, mean, std = scores(im)
        vals.append((p, var_lap, mean, std))
    # stats rapides
    vl = np.array([v[1] for v in vals]); st = np.array([v[3] for v in vals])
    print("Flou (var Laplacian)  min/med/max:", float(vl.min()), float(np.median(vl)), float(vl.max()))
    print("Contraste (std)       min/med/max:", float(st.min()), float(np.median(st)), float(st.max()))
    # top 10 pires par flou et contraste
    vals_sorted = sorted(vals, key=lambda t: (t[1], t[3]))[:20]
    for p, var_lap, mean, std in vals_sorted[:10]:
        print("BLURRY/LOW-C:", os.path.basename(p), f"lap={var_lap:.2f}", f"std={std:.3f}")

if __name__ == "__main__":
    main()
