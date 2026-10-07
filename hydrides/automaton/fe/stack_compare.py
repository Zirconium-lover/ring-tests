"""Сверка Фурье (stack_runs.py --fut-el --fields) с МКЭ (stack_fe.py) для стопок пластинок: работа новой и старых
пластинок, коллективная прибавка, разность радиальная − окружная, поля в матрице, пластическая зона.
Поля сравниваются в осях пластинки (t — вдоль, n — по нормали): МКЭ (центры элементов матрицы) интерполируется
в центры клеток Фурье, полоса 0.15 мкм у границ гидрида исключается (там скачок и полклетки сдвига схемы).
g_next = (σ_tt ε_t + σ_nn ε_n + σ_zz ε_t)/ε_n — выгода следующей пластинки той же ориентации, как в автомате.
python stack_compare.py папка_фурье папка_мкэ [папка_фурье_основная]"""
import os
import sys
import json
import glob
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EPS_N, EPS_T, R2 = 0.0720, 0.0458, np.sqrt(2.0)
L = 30.0
BLUE, ORANGE, GREEN, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#2f9e6e", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"


def fft_tn(z, orient):
    """Поля Фурье в осях пластинки: массивы [t, n], компоненты tt, nn, zz, tn; p; маска гидрида."""
    s = z["sig"].astype(np.float64)
    hyd = z["new"] | z["old"]
    if orient == "rad":                       # нормаль по x: [x, y] = [n, t]
        tt, nn, zz, tn = s[1].T, s[0].T, s[2].T, s[5].T / R2
        return dict(tt=tt, nn=nn, zz=zz, tn=tn, p=z["p"].T, hyd=hyd.T)
    return dict(tt=s[0], nn=s[1], zz=s[2], tn=s[5] / R2, p=z["p"], hyd=hyd)


def derived(d):
    d["h"] = (d["tt"] + d["nn"] + d["zz"]) / 3
    d["g"] = (d["tt"] * EPS_T + d["nn"] * EPS_N + d["zz"] * EPS_T) / EPS_N
    return d


def ccx_on_grid(z, n):
    """МКЭ → центры клеток Фурье (только элементы матрицы)."""
    h = L / n
    x = (np.arange(n) + 0.5) * h - L / 2
    T, N = np.meshgrid(x, x, indexing="ij")
    m = z["own"] < 0
    pts = z["xy"][m].astype(np.float64)
    S = z["S"][m].astype(np.float64)
    out = {}
    for k, i in (("tt", 0), ("nn", 1), ("zz", 2), ("tn", 3)):
        out[k] = LinearNDInterpolator(pts, S[:, i])(T, N)
    return derived(out), T, N


def compare(fz, cz, orient):
    f = derived(fft_tn(fz, orient))
    n = f["tt"].shape[0]
    c, T, N = ccx_on_grid(cz, n)
    dist = ndi.distance_transform_edt(~f["hyd"]) * (L / n)            # расстояние клетки до гидрида
    band = (dist > 0.15) & np.isfinite(c["h"])
    near = band & (dist < 1.0)
    st = {}
    STEP = {"deck": (2.5, 1.2), "chain": (5.8, 0.6)}
    for conf, (a, b) in STEP.items():                      # g, средняя по месту следующей пластинки колоды/цепочки
        site = (np.abs(T - a) < 2.5) & (np.abs(N - b) < 0.3) & ~f["hyd"]
        st["site_" + conf] = (float(np.mean(f["g"][site])), float(np.nanmean(c["g"][site])))
    for k in ("h", "g", "nn"):
        dlt = f[k] - c[k]
        st[k] = dict(rms_near=float(np.sqrt(np.mean(dlt[near] ** 2))), max_near=float(np.abs(dlt[near]).max()),
                     scale_near=float(np.sqrt(np.mean(c[k][near] ** 2))),
                     rms_all=float(np.sqrt(np.mean(dlt[band] ** 2))))
    return f, c, st, T, N


def line(f, c, T, N, conf):
    """Профили через перемычку: колода — по n при t = −1.25 (середина перекрытия), цепочка — по t при n = −0.25."""
    n = T.shape[0]; h = L / n
    if conf == "deck":
        i = int(round((-1.25 + L / 2) / h - 0.5))
        sel = (N[i] > -3.2) & (N[i] < 1.6)
        m = sel & ~f["hyd"][i]
        return N[i][m], f["g"][i][m], c["g"][i][m], f["h"][i][m], c["h"][i][m], "n, мкм (t = −1.25), гидрид исключён"
    j = int(round((-0.25 + L / 2) / h - 0.5))
    sel = (T[:, j] > -6.0) & (T[:, j] < 1.0)
    m = sel & ~f["hyd"][:, j]
    return T[:, j][m], f["g"][:, j][m], c["g"][:, j][m], f["h"][:, j][m], c["h"][:, j][m], "t, мкм (n = −0.25), гидрид исключён"


if __name__ == "__main__":
    DF, DC = sys.argv[1], sys.argv[2]
    DP = sys.argv[3] if len(sys.argv) > 3 else None
    rows = {}
    for fn in sorted(glob.glob(os.path.join(DC, "*.json"))):
        cm = json.load(open(fn))
        if "conf" not in cm:
            continue
        key = f"{cm['conf']}_{cm['orient']}_{cm['load']}"
        if cm.get("Eh", 1.0) != 1.0 or not cm.get("plastic", True):
            continue
        ff = glob.glob(os.path.join(DF, key + "_h0.1_fe.json"))
        fm = json.load(open(ff[0])) if ff else None
        if fm is None and cm["conf"] == "single" and DP:                 # одиночная: --fut-el не нужен
            p = os.path.join(DP, f"{cm['orient']}_single_{cm['load']}.json")
            fm = json.load(open(p)) if os.path.exists(p) else None
        rows[key] = dict(ccx=cm, fft=fm)
    print(f"{'случай':18s} | g: Фурье   МКЭ    Δ  | W_старых: Фурье  МКЭ | пласт. мкм²: Фурье  МКЭ")
    for k, r in rows.items():
        cm, fm = r["ccx"], r["fft"]
        if fm is None:
            print(f"{k:18s} | МКЭ {cm['g']:8.1f} (Фурье ещё нет)")
            continue
        wo = (f"{fm['W_old'] / EPS_N:8.1f} {cm['W_old'] / EPS_N:8.1f}" if cm["W_old"] is not None else " " * 17)
        print(f"{k:18s} | {fm['g']:8.1f} {cm['g']:8.1f} {fm['g'] - cm['g']:5.1f} | {wo} | {fm['pl_area']:6.1f} {cm['pl_area']:6.1f}")
    # производные величины: коллективная прибавка и радиальная − окружная
    der = {}
    single = lambda o, lk: rows.get(f"single_{o}_{lk}") or (rows.get("single_circ_0") if lk == "0" else None)
    ok = lambda r: r is not None and r["fft"] is not None
    for conf in ("single", "deck", "chain"):
        for lk in ("0", "U110"):
            for o in ("rad", "circ"):
                a, b = rows.get(f"{conf}_{o}_{lk}"), single(o, lk)
                if conf != "single" and ok(a) and ok(b):
                    der[f"Δg колл. {conf} {o} {lk}"] = (a["fft"]["g"] - b["fft"]["g"], a["ccx"]["g"] - b["ccx"]["g"])
            ra, ca = rows.get(f"{conf}_rad_{lk}"), rows.get(f"{conf}_circ_{lk}")
            if ok(ra) and ok(ca):
                der[f"рад − окр {conf} {lk}"] = (ra["fft"]["g"] - ca["fft"]["g"], ra["ccx"]["g"] - ca["ccx"]["g"])
    print("\nпроизводные, МПа:          Фурье     МКЭ")
    for k, (a, b) in der.items():
        print(f"  {k:26s} {a:8.1f} {b:8.1f}")
    # поля
    fields = {}
    for k, r in rows.items():
        fz = os.path.join(DF, k + "_h0.1_fe.npz"); cz = os.path.join(DC, k + ".npz")
        if r["fft"] is None or not (os.path.exists(fz) and os.path.exists(cz)):
            continue
        f, c, st, T, N = compare(np.load(fz), np.load(cz), r["ccx"]["orient"])
        fields[k] = (f, c, T, N)
        r["field"] = st
        print(f"{k:18s} поля в 0.15–1 мкм от гидрида: σ_h Δrms {st['h']['rms_near']:5.1f} (масштаб {st['h']['scale_near']:5.0f}),"
              f" g Δrms {st['g']['rms_near']:5.1f} (масштаб {st['g']['scale_near']:5.0f}), макс |Δg| {st['g']['max_near']:5.0f};"
              f" дальше: g Δrms {st['g']['rms_all']:5.1f}; g на месте следующей: колода {st['site_deck'][0]:.0f} / {st['site_deck'][1]:.0f},"
              f" цепочка {st['site_chain'][0]:.0f} / {st['site_chain'][1]:.0f}")
    json.dump(dict(rows=rows, derived={k: list(v) for k, v in der.items()}),
              open(os.path.join(DC, "compare.json"), "w"), indent=1, default=float)
    # рисунок: карты g_next Фурье / МКЭ / разность для колоды и цепочки (радиальные, U110) и профили
    show = [k for k in ("deck_rad_U110", "chain_rad_U110", "deck_circ_U110", "chain_circ_U110") if k in fields][:2]
    if not show:
        sys.exit()
    from matplotlib.colors import TwoSlopeNorm
    fig, axs = plt.subplots(len(show), 4, figsize=(22, 3.9 * len(show)), facecolor=BG, squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.25, 1.0]))
    for row, k in zip(axs, show):
        f, c, T, N = fields[k]
        conf = k.split("_")[0]
        win = (T > -9) & (T < 4) & (N > -3.5) & (N < 2) if conf == "deck" else (T > -10) & (T < 4) & (N > -2.5) & (N < 2)
        ti = np.nonzero(win.any(1))[0]; ni = np.nonzero(win.any(0))[0]
        ext = [T[ti[0], 0], T[ti[-1], 0], N[0, ni[0]], N[0, ni[-1]]]
        sub = lambda a: a[ti[0]:ti[-1] + 1, ni[0]:ni[-1] + 1].T
        fg = np.where(f["hyd"], np.nan, f["g"])
        for ax, a, ttl in ((row[0], fg, "Фурье, шаг 0.1 мкм"), (row[1], c["g"], "МКЭ (CalculiX)")):
            im = ax.imshow(sub(np.where(f["hyd"], np.nan, a)), origin="lower", extent=ext, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0.0, -1500.0, 500.0), aspect="equal")
            ax.contour(sub(f["hyd"].astype(float)), [0.5], colors=INK, linewidths=0.8, extent=ext)
            ax.set_title(f"{k}: g следующей, {ttl}", loc="left", fontsize=10, color=INK)
        plt.colorbar(im, ax=row[1], fraction=0.03, label="МПа")
        d = np.where(f["hyd"], np.nan, f["g"] - c["g"])
        im = row[2].imshow(sub(d), origin="lower", extent=ext, cmap="PuOr", vmin=-100, vmax=100, aspect="equal")
        row[2].contour(sub(f["hyd"].astype(float)), [0.5], colors=INK, linewidths=0.8, extent=ext)
        row[2].set_title("Фурье − МКЭ", loc="left", fontsize=10, color=INK)
        plt.colorbar(im, ax=row[2], fraction=0.03, label="МПа")
        x, gf, gc, hf_, hc, lab = line(f, c, T, N, conf)
        row[3].plot(x, gc, color=INK, lw=1.6, label="МКЭ: g")
        row[3].plot(x, gf, "o", color=BLUE, ms=3, label="Фурье: g")
        row[3].plot(x, hc, color=MUTED, lw=1.2, ls="--", label="МКЭ: σ_h")
        row[3].plot(x, hf_, "s", color=ORANGE, ms=3, label="Фурье: σ_h")
        row[3].set_xlabel(lab); row[3].set_ylabel("МПа"); row[3].grid(color=GRID); row[3].legend(frameon=False, fontsize=8)
        row[3].set_title("профиль через перемычку", loc="left", fontsize=10, color=INK)
        for ax in row:
            ax.set_facecolor(BG)
    fig.suptitle("Фурье против МКЭ для стопок пластинок: поле выгоды следующей пластинки (нагрузка 110 МПа)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(os.path.join(HERE, "figs", "fig_fe7_stack_check.png"), dpi=95, facecolor=BG, bbox_inches="tight")
    print("рисунок сохранён")
