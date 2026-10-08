"""Пояснительный рисунок: как работает спектральный анализ гидридов (hydride_spec.py) на окне снимка.

Панели: окно снимка; сигнал гидридов («чёрный цилиндр»); двумерный спектр мощности и полосы масштабов;
распределение энергии спектра по углу; линии на мелком и крупном масштабе (гаусс + вторые производные в Фурье);
суррогат с перемешанными фазами (тот же спектр, без линий) и порог шума; доля радиальных по масштабу.

python spec_explain.py снимок.jpg --um-per-px 0.862 --win 300 --at 0.5,0.5 --out рисунок.png
(--at — центр окна в долях кадра: x, y; окно должно лежать внутри стенки, окружное направление — по горизонтали)
"""
import argparse

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402

import hydride_spec as hs  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img"); ap.add_argument("--um-per-px", type=float, required=True)
    ap.add_argument("--win", type=float, default=300.0); ap.add_argument("--at", default="0.5,0.5")
    ap.add_argument("--out", default="spec_explain.png")
    ap.add_argument("--scales", default="1.5,6", help="мелкий и крупный масштаб σ для панелей, мкм")
    a = ap.parse_args()
    um = a.um_per_px
    g = hs.read_gray(a.img)
    fx, fy = (float(v) for v in a.at.split(","))
    n = int(round(a.win / um))
    y0 = int(fy * g.shape[0] - n / 2); x0 = int(fx * g.shape[1] - n / 2)
    G = g[y0:y0 + n, x0:x0 + n]
    h = hs.hydride_signal(G, um)
    s1, s2 = (float(v) for v in a.scales.split(","))

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 4, wspace=0.32, hspace=0.42)
    ext = (0, n * um, n * um, 0)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(G, cmap="gray", extent=ext)
    ax.set_title("а) окно снимка\nгоризонталь — дуга, вертикаль — радиус", loc="left", fontsize=10)
    ax.set_xlabel("мкм", color=MUTED)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(h, cmap="gray_r", extent=ext)
    ax.set_title("б) сигнал гидридов: насколько точка\nтемнее светлого окружения", loc="left", fontsize=10)

    # спектр мощности
    f = (h - h.mean()) * np.outer(hs.tukey(n, 0.25), hs.tukey(n, 0.25))
    P = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
    k = np.fft.fftshift(np.fft.fftfreq(n, um))                       # 1/мкм
    ax = fig.add_subplot(gs[0, 2])
    kmax = min(0.12, 0.5 / um)
    ax.imshow(np.log10(P + P.max() * 1e-7), cmap="magma", extent=(k[0], k[-1], k[-1], k[0]))
    ax.set_xlim(-kmax, kmax); ax.set_ylim(kmax, -kmax)
    for sig, col in ((s1, "#7fd3ff"), (s2, "#ffe08a")):
        r = 1 / (2 * np.pi * sig)                                    # гаусс σ пропускает частоты до ~1/(2πσ)
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=col, lw=1.3, ls="--"))
        ax.text(r * 0.75, -r * 0.8, f"σ = {sig:g}", color=col, fontsize=9)
    ax.set_xlabel("k вдоль дуги, 1/мкм", color=MUTED); ax.set_ylabel("k по радиусу, 1/мкм", color=MUTED)
    ax.set_title("в) спектр |F(k)|² (лог.): окружные гидриды —\nвертикальная ось, радиальные — горизонтальная;\nкруги — частоты, которые пропускает гаусс σ", loc="left", fontsize=10)

    # энергия по углу волнового вектора
    KX, KY = np.meshgrid(k, k)
    kk = np.hypot(KX, KY)
    band = (kk > 1 / 30.0) & (kk < 1 / 2.0)
    ang = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))             # 0 — волна вдоль дуги (структура радиальная)
    edges = np.linspace(0, 90, 19); mid = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([P[band & (ang >= lo) & (ang < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    prof = np.clip(prof - prof.min(), 0, None); prof /= prof.sum()
    ax = fig.add_subplot(gs[0, 3])
    ax.bar(90 - mid, prof, width=4.5, color=[ORANGE if m < 45 else BLUE for m in mid])
    fn_spec = float(prof[mid < 45].sum())
    ax.set_xlabel("угол гидрида к дуге, °  (0 — окружной, 90 — радиальный)", color=MUTED, fontsize=9)
    ax.set_ylabel("доля энергии спектра", color=MUTED)
    ax.set_title(f"г) энергия спектра по углу\n(волны 2–30 мкм): радиальных {fn_spec:.2f}", loc="left", fontsize=10)
    ax.grid(color=GRID, lw=0.6)

    # линии на масштабах
    surr = hs.phase_surrogates(h, n=2)
    for col_i, sig in ((0, s1), (1, s2)):
        s_px = sig / um
        st, an = hs.ridges(h, s_px)
        thr = hs.null_threshold(surr, s_px)
        sk = hs.ridge_skeleton(st, min_len_px=max(3, int(2 * s_px)), thr=thr)
        dev = np.minimum(an, 180 - an)
        rgb = np.ones((n, n, 3))
        rgb[sk & (dev > 45)] = matplotlib.colors.to_rgb(ORANGE)
        rgb[sk & (dev <= 45)] = matplotlib.colors.to_rgb(BLUE)
        ax = fig.add_subplot(gs[1, col_i])
        ax.imshow(G, cmap="gray", extent=ext, alpha=0.35)
        ax.imshow(rgb, extent=ext, alpha=0.9)
        fn, simon = hs.fn_from(an[sk])
        ax.set_title(f"{'де'[col_i]}) масштаб σ = {sig:g} мкм, линии сильнее шума\nоранж. — радиальные, синий — окружные\nRHF = {simon:.2f}",
                     loc="left", fontsize=10)

    # суррогат
    ax = fig.add_subplot(gs[1, 2])
    s_px = s1 / um
    ax.imshow(surr[0], cmap="gray_r", extent=ext)
    st_r = hs.ridges(h, s_px)[0].ravel(); st_s = np.concatenate([hs.ridges(x, s_px)[0].ravel() for x in surr])
    thr = np.percentile(st_s, 99.0)
    ax.set_title("ж) суррогат: тот же спектр, фазы\nперемешаны — волны те же, линий нет", loc="left", fontsize=10)
    ins = ax.inset_axes([0.55, 0.05, 0.42, 0.35])
    bins = np.linspace(0, np.percentile(st_r, 99.9), 60)
    ins.hist(st_s, bins, color="#9a9893", alpha=0.8, density=True, label="суррогат")
    ins.hist(st_r, bins, color=INK, histtype="step", density=True, label="снимок")
    ins.axvline(thr, color=ORANGE, lw=1); ins.set_yscale("log"); ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("сила линий; порог — 99 % суррогата", fontsize=7)

    # RHF по масштабу
    ax = fig.add_subplot(gs[1, 3])
    sigs = tuple(s1 * f for f in (1, 2, 4, 8))
    rh = []
    for sig in sigs:
        s_px = sig / um
        st, an = hs.ridges(h, s_px)
        sk = hs.ridge_skeleton(st, min_len_px=max(3, int(2 * s_px)), thr=hs.null_threshold(surr, s_px))
        rh.append(hs.fn_from(an[sk])[1] if sk.any() else np.nan)
    ok = np.isfinite(rh)
    ax.plot(np.array(sigs)[ok], np.array(rh)[ok], "-o", color=BLUE, lw=2)
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(sigs); ax.set_xticklabels([f"{s:g}" for s in sigs])
    ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("масштаб σ, мкм", color=MUTED); ax.set_ylabel("RHF (веса Simon et al.)", color=MUTED)
    ax.set_title("з) доля радиальных по масштабу:\nмелкий — отдельные гидриды,\nкрупный — как они выстроены", loc="left", fontsize=10)
    for axx in fig.axes:
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
    fig.savefig(a.out, dpi=110, facecolor="white", bbox_inches="tight")
    print("→", a.out, "спектральная доля", round(fn_spec, 3), "RHF по масштабам", [round(v, 2) for v in rh])


if __name__ == "__main__":
    main()
