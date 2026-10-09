"""Ревью (calib_plan.md, Р2, Р3): чувствительность F_l к оператору съёмки — та же структура модели, снятая по-разному
(пиксель, утолщение травлением, минимальная длина); и гидриды через стык поля по толщине (периодичность).
python check_operator.py папка_колец"""
import sys, glob, os, json
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rhc_model as RM

def fl_mask(m, um, Lmin=5.0):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    out = {"all": [0, 0], "out": [0, 0], "mid": [0, 0], "in": [0, 0]}
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        ys, xs = np.nonzero(lab[sl] == k)
        if len(ys) < 3: continue
        P = np.stack([ys * um, xs * um], 1).astype(float); P -= P.mean(0)
        w, v = np.linalg.eigh(P.T @ P); ax = v[:, -1]; pr = P @ ax
        L = pr.max() - pr.min() + um
        if L < Lmin: continue
        rad = abs(ax[0]) >= np.cos(np.radians(45))
        yc = (ys.mean() + sl[0].start) / m.shape[0]
        third = ("out", "mid", "in")[min(2, int(yc * 3))]
        for key in ("all", third):
            out[key][0] += L * rad; out[key][1] += L
    return {k: round(a / b, 2) if b else None for k, (a, b) in out.items()}

for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    if not os.path.exists(f[:-5] + ".npz"): continue
    meta = json.load(open(f)); z = np.load(f[:-5] + ".npz")
    if "hyd" not in z.files: continue
    hyd = np.asarray(z["hyd"], float); dx = float(z["dx"])
    print(os.path.basename(f)[:60], "текущий F_l_5 по третям:", [round(meta[f"F_l_5_{s}"], 2) for s in ("out", "mid", "in")])
    for etch in (0.0, 1.0, 2.0, 3.0):
        m = RM.optical(hyd, dx, etch)
        r5 = fl_mask(m, RM.UM, 5.0); r10 = fl_mask(m, RM.UM, 10.0)
        print(f"   0.5 мкм/пикс, травление {etch} мкм: площадь {m.mean():.3f}; L≥5: {r5}; L≥10: {r10}")
    for um_px in (1.0, 3.5):
        g = RM.as_micrograph(hyd, dx, 3.0, um_px)
        m = g < 0.5
        print(f"   {um_px} мкм/пикс, травление 3: L≥5 {fl_mask(m, um_px, 5.0)}; L≥10 {fl_mask(m, um_px, 10.0)}")
    hb = hyd > 0.2
    t = np.concatenate([hb[-50:], hb[:50]], 0)              # стык внутренней (низ) и наружной (верх) поверхностей
    lab, n = ndi.label(t, structure=np.ones((3, 3)))
    cross = [k for k, s in enumerate(ndi.find_objects(lab), 1) if s[0].start < 50 and s[0].stop > 50]
    print("   гидридов через стык поля по толщине:", len(cross))
