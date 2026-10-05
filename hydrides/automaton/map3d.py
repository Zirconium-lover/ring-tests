"""Карта притяжения в 3D: где поле пластинки-диска выгодно для следующей пластинки.
Для сравнения — та же пластинка в 2D (бесконечная вдоль оси трубы, плоская деформация).
python map3d.py; python map3d.py shapes; python map3d.py needles  → числа в stdout и figs/map3d_numbers.json; рисунок — fig_map3d.py"""
import os
import json
import numpy as np
from ca3d import Elastic3, disc_fraction, eps_star, _contract, EPS_N

HERE = os.path.dirname(os.path.abspath(__file__))
N, DX = 128, 0.25                      # куб 32 мкм, шаг 0.25 мкм
R, H = 1.5, 0.6                        # радиус (полудлина в 2D) и толщина пластинки, мкм
E_MOD, NU = 90e3, 0.34
BETA, CAP = 0.12, 90.0                 # как в автомате
C0 = np.array([16.0, 16.0, 16.0]) + DX / 2


def normal(psi_deg, gam_deg=0.0):
    """Нормаль пластинки, след которой в сечении r–θ идёт под ψ к TD; γ — наклон нормали к оси трубы."""
    a, g = np.radians(psi_deg), np.radians(gam_deg)
    return np.array([np.sin(a) * np.cos(g), np.cos(a) * np.cos(g), np.sin(g)])


def strip_fraction(nv):
    """Пластинка 2D: полоса ширины 2R в плоскости r–θ, бесконечная вдоль z (доля в клетках, 4×4 подточки)."""
    sub = 4
    off = (np.arange(sub) + 0.5) / sub
    x = (np.arange(N)[:, None, None, None] + off[None, None, :, None]) * DX - C0[0]
    y = (np.arange(N)[None, :, None, None] + off[None, None, None, :]) * DX - C0[1]
    t = np.array([nv[1], -nv[0]]) / np.hypot(nv[0], nv[1])   # след в плоскости
    q = x * nv[0] + y * nv[1]
    s = x * t[0] + y * t[1]
    f2 = ((np.abs(q) <= H / 2) & (np.abs(s) <= R)).mean(axis=(2, 3))
    return np.repeat(f2[:, :, None], N, axis=2)


def disc_field(nv):
    ax, fr = disc_fraction(N, DX, C0, nv, R, H, sub=2)
    f = np.zeros((N,) * 3)
    f[np.ix_(*ax)] = fr
    return f


def ellipse_fraction(nv, a_t, a_z, sub=2):
    """Пластинка-эллипс в своей плоскости: полуось a_t вдоль следа в r–θ, a_z вдоль оси трубы."""
    t = np.array([nv[1], -nv[0], 0.0]) / np.hypot(nv[0], nv[1])
    zt = np.cross(nv, t)                                       # вторая ось в плоскости (≈ ось трубы)
    r = max(a_t, a_z) + H
    lo = np.floor((C0 - r) / DX).astype(int) - 1
    hi = np.floor((C0 + r) / DX).astype(int) + 2
    ax = [np.arange(lo[i], hi[i]) for i in range(3)]
    off = (np.arange(sub) + 0.5) / sub
    X = (ax[0][:, None, None, None, None, None] + off[None, None, None, :, None, None]) * DX - C0[0]
    Y = (ax[1][None, :, None, None, None, None] + off[None, None, None, None, :, None]) * DX - C0[1]
    Z = (ax[2][None, None, :, None, None, None] + off[None, None, None, None, None, :]) * DX - C0[2]
    q = X * nv[0] + Y * nv[1] + Z * nv[2]
    s = X * t[0] + Y * t[1] + Z * t[2]
    u = X * zt[0] + Y * zt[1] + Z * zt[2]
    fr = ((np.abs(q) <= H / 2) & ((s / a_t) ** 2 + (u / a_z) ** 2 <= 1)).mean(axis=(3, 4, 5))
    f = np.zeros((N,) * 3)
    f[np.ix_(*[a % N for a in ax])] = fr
    return f


EL = Elastic3(N, DX, E_MOD, NU)
G = np.indices((N,) * 3).astype(float)
RX, RY, RZ = [(G[i] + 0.5) * DX - C0[i] for i in range(3)]
DIST = np.sqrt(RX ** 2 + RY ** 2 + RZ ** 2)
del G


def gain(f, n1, n2):
    """g = σ(пластинки n1):ε*(n2)/ε_n по клеткам, МПа."""
    S = EL.stress_of(np.fft.rfftn(f), eps_star(n1))
    return _contract(list(eps_star(n2)), S)


def free_mask(f):
    """Клетки, где можно зародиться: вне пластинки и зазора 0.5 мкм вокруг неё."""
    from scipy.ndimage import binary_dilation
    return ~binary_dilation(f > 0.05, iterations=int(round(0.5 / DX)))


def stats(g, f, plane_only=False):
    """Избыточная привлекательность w = exp(β·min(g, cap)) − 1 (> 0) в шаре 3R вокруг центра:
    доли по направлению смещения (где больше |Δ|: TD, ND или ось трубы) и доля в слое |Δz| < 0.5 мкм."""
    m = free_mask(f) & (DIST <= 3 * R)
    if plane_only:
        m &= np.abs(RZ) < DX
    w = np.where(m, np.clip(np.exp(BETA * np.clip(g, -CAP, CAP)) - 1, 0, None), 0.0)
    a = np.stack([np.abs(RX), np.abs(RY), np.abs(RZ)])
    k = np.argmax(a, axis=0)
    tot = w.sum()
    out = {nm: float(w[k == i].sum() / tot) for i, nm in enumerate(("TD", "ND", "L"))}
    out["slab"] = float(w[np.abs(RZ) < 0.5].sum() / tot)
    # направление цепочки в плоскости r–θ: главная ось тензора Σ w r̂r̂ по клеткам у среза z ≈ 0
    ms = m & (np.abs(RZ) < 0.5)
    rr = np.stack([RX[ms], RY[ms]]); ww = w[ms]
    rr = rr / np.maximum(np.linalg.norm(rr, axis=0), 1e-9)
    T = (ww * rr) @ rr.T
    ev, vec = np.linalg.eigh(T)
    v = vec[:, -1]
    out["chain_deg"] = float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))   # 0° — вдоль TD, 90° — вдоль радиуса
    out["chain_anis"] = float(ev[-1] / max(ev.sum(), 1e-12))
    return out


def needle_field(nv, u, A, B):
    """Игла (эллипс A × B в базисной плоскости, длинная ось u) с центром C0 — доля по клеткам куба."""
    from ca3d import ellipse_fraction
    ax, fr = ellipse_fraction(N, DX, C0, nv, u, A, B, H)
    f = np.zeros((N,) * 3)
    f[np.ix_(*ax)] = fr
    return f


def needle_axis(nv, phi_deg):
    """Ось ⟨11-20⟩ в базисной плоскости под φ к проекции оси трубы."""
    ref = np.array([0, 0, 1.0]) - nv[2] * nv
    ref /= np.linalg.norm(ref)
    w = np.cross(nv, ref)
    f = np.radians(phi_deg)
    return np.cos(f) * ref + np.sin(f) * w


def run_needles(A=2.5, B=0.5):
    """Иглы 5 × 1 мкм под φ = 0° (вдоль оси трубы), 30° (CWSR) и 90° (в плоскости r–θ) к оси трубы."""
    res = {}
    for psi in (20, 48, 70):
        n1 = normal(psi)
        for phi in (0, 30, 90):
            f = needle_field(n1, needle_axis(n1, phi), A, B)
            for lab, n2 in (("parallel", n1), ("mirror", normal(-psi))):
                g = gain(f, n1, n2)
                st = stats(g, f)
                st["g_max"] = float(np.max(np.where(free_mask(f), g, -1e9)))
                res[f"needle_psi{psi}_phi{phi}_{lab}"] = st
                print("игла", psi, phi, lab, {k: round(v, 2) for k, v in st.items()}, flush=True)
    return res


def run_shapes():
    """Вытянутые вдоль оси трубы пластинки (лента): полуось вдоль следа 1.25 мкм, вдоль оси 1.25…5 мкм."""
    res = {}
    for psi in (20, 48, 70):
        n1 = normal(psi)
        for az in (1.25, 2.5, 5.0, 10.0):
            f = ellipse_fraction(n1, 1.25, az)
            for lab, n2 in (("parallel", n1), ("mirror", normal(-psi))):
                g = gain(f, n1, n2)
                st = stats(g, f)
                st["g_max"] = float(np.max(np.where(free_mask(f), g, -1e9)))
                res[f"ell_psi{psi}_az{az}_{lab}"] = st
                print("эллипс", psi, az, lab, {k: round(v, 2) for k, v in st.items()}, flush=True)
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("shapes", "needles"):
        fn = os.path.join(HERE, "figs", "map3d_numbers.json")
        res = json.load(open(fn)) if os.path.exists(fn) else {}
        res.update(run_shapes() if sys.argv[1] == "shapes" else run_needles())
        json.dump(res, open(fn, "w"), indent=1)
        sys.exit()
    res = {}
    for psi in (20, 48, 70):
        n1 = normal(psi)
        fd, fs = disc_field(n1), strip_fraction(n1)
        for kind, f in (("3D", fd), ("2D", fs)):
            for lab, n2 in (("parallel", n1), ("mirror", normal(-psi))):
                g = gain(f, n1, n2)
                st = stats(g, f, plane_only=(kind == "2D"))
                st["g_max"] = float(np.max(np.where(free_mask(f), g, -1e9)))
                res[f"{kind}_psi{psi}_{lab}"] = st
                print(kind, psi, lab, {k: round(v, 2) for k, v in st.items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "figs", "map3d_numbers.json"), "w"), indent=1)
