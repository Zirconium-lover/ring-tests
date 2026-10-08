"""Фурье на пальцах: идеализированные «снимки» гидридов и их спектры.

Окно 200 × 200 мкм, пиксель 0.5 мкм. По горизонтали — дуга (окружное
направление), по вертикали — радиус (через стенку). Гидриды — тёмные
пластинки толщиной ~1 мкм.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage.draw import line_aa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs") + os.sep
import os; os.makedirs(OUT, exist_ok=True)
PX = 0.5
rng = np.random.default_rng(7)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
SPEC = LinearSegmentedColormap.from_list("spec", [BG, "#9cc3ef", BLUE, "#0b2a55"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


# ---------------- рисование ----------------
def render(plates, W=200, H=200):
    """plates: список (x, y, угол_град, длина) в мкм; угол 0 — вдоль дуги, 90 — по радиусу."""
    img = np.zeros((int(H / PX), int(W / PX)))
    for x, y, a, L in plates:
        dx, dy = 0.5 * L * np.cos(np.radians(a)), 0.5 * L * np.sin(np.radians(a))
        r0, c0, r1, c1 = [int(round(v / PX)) for v in (y - dy, x - dx, y + dy, x + dx)]
        rr, cc, val = line_aa(r0, c0, r1, c1)
        ok = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
        img[rr[ok], cc[ok]] = np.maximum(img[rr[ok], cc[ok]], val[ok])
    img = ndi.grey_dilation(img, size=(2, 2))
    return np.clip(ndi.gaussian_filter(img, 0.5) * 1.3, 0, 1)


def random_plates(n, angle, L, W=200, H=200, jitter=8):
    out = []
    for _ in range(n):
        a = angle + rng.normal(0, jitter)
        out.append((rng.uniform(0, W), rng.uniform(0, H), a, L))
    return out


# ---------------- спектр ----------------
def hann2(shape):
    """Окно с плавными краями (Тьюки): в середине вес 1, к краям плавно до 0."""
    from scipy.signal.windows import tukey
    return np.outer(tukey(shape[0], 0.3), tukey(shape[1], 0.3))


def spectrum(img, window=True):
    f = img - img.mean()
    if window:
        f = f * hann2(f.shape)
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(img.shape[0], PX))
    kx = np.fft.fftshift(np.fft.fftfreq(img.shape[1], PX))
    return P, kx, ky


def radial_share(img, pmin=2, pmax=10, window=True):
    """Доля энергии спектра от «радиальных» структур (±45° от радиуса).
    Радиальная пластинка даёт полосу в спектре вдоль оси частот дуги (поперёк себя)."""
    P, kx, ky = spectrum(img, window)
    KX, KY = np.meshgrid(kx, ky)
    k = np.hypot(KX, KY)
    band = (k > 1 / pmax) & (k < 1 / pmin)
    ang_k = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))   # 0 — волна вдоль дуги
    rad = band & (ang_k < 45)
    return P[rad].sum() / P[band].sum()


KMAX = 0.25


def show_img(ax, img, title, W=200, H=200, xlabel=True, ylabel=True):
    ax.imshow(1 - img, cmap="gray", vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="antialiased")
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    if xlabel: ax.set_xlabel("вдоль дуги, мкм")
    if ylabel: ax.set_ylabel("по радиусу, мкм")
    ax.set_xticks([0, W // 2, W]); ax.set_yticks([0, H // 2, H])


def show_spec(ax, img, title="", kmax=KMAX, window=True, rings=(50, 10, 5), smooth=1.2, ylabel=True, vmin=-3.0):
    P, kx, ky = spectrum(img, window)
    if smooth:
        P = ndi.gaussian_filter(P, smooth)
    L = np.log10(P / P.max() + 1e-12)
    ax.imshow(L, cmap=SPEC, vmin=vmin, vmax=0, origin="lower",
              extent=(kx[0], kx[-1], ky[0], ky[-1]), interpolation="antialiased")
    ax.set_xlim(-kmax, kmax); ax.set_ylim(-kmax, kmax)
    for p in rings:
        if 1 / p < kmax:
            ax.add_patch(plt.Circle((0, 0), 1 / p, fill=False, ls=(0, (2, 3)), lw=0.8, color=MUTED))
            ax.text(0.71 / p, 0.71 / p, f"{p} мкм", fontsize=7.5, color=MUTED)
    ax.axhline(0, color=GRID, lw=0.5); ax.axvline(0, color=GRID, lw=0.5)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.set_xlabel("частота вдоль дуги, 1/мкм")
    if ylabel: ax.set_ylabel("частота по радиусу, 1/мкм")
    ax.set_xticks([-0.2, 0, 0.2]); ax.set_yticks([-0.2, 0, 0.2])


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(OUT + name, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("saved", name)


N = 400
yy, xx = np.mgrid[0:N, 0:N] * PX

# ===== Рис. 1. Азбука: волна → точка =====
waves = [
    ("а) Полосы вдоль дуги,\nпериод 20 мкм", 0.5 + 0.5 * np.cos(2 * np.pi * yy / 20)),
    ("б) Те же полосы,\nповёрнутые на 30°", 0.5 + 0.5 * np.cos(2 * np.pi * (yy * np.cos(np.radians(30)) - xx * np.sin(np.radians(30))) / 20)),
    ("в) Полосы мельче,\nпериод 8 мкм", 0.5 + 0.5 * np.cos(2 * np.pi * yy / 8)),
    ("г) Резкие тонкие линии,\nпериод 20 мкм", (np.abs(((yy + 10) % 20) - 10) < 0.8).astype(float)),
]
fig, axs = plt.subplots(2, 4, figsize=(14, 7.2), facecolor=BG,
                        gridspec_kw=dict(hspace=0.45, wspace=0.35))
for k, (t, img) in enumerate(waves):
    show_img(axs[0, k], img, t, ylabel=(k == 0))
    show_spec(axs[1, k], img, "спектр", ylabel=(k == 0), smooth=0, rings=(20, 8), vmin=-4)
save(fig, "fig1_azbuka.png")

# ===== Рис. 2. Ориентация гидридов =====
cases = [
    ("а) Все окружные", random_plates(120, 0, 20)),
    ("б) Все радиальные", random_plates(120, 90, 20)),
    ("в) 70 % окружных,\n30 % радиальных", random_plates(84, 0, 20) + random_plates(36, 90, 20)),
]
imgs2 = [render(p) for _, p in cases]
fig = plt.figure(figsize=(14, 7.4), facecolor=BG)
gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35, width_ratios=[1, 1, 1, 1.15])
shares = []
for k, ((t, p), img) in enumerate(zip(cases, imgs2)):
    show_img(fig.add_subplot(gs[0, k]), img, t, ylabel=(k == 0))
    show_spec(fig.add_subplot(gs[1, k]), img, "спектр", ylabel=(k == 0))
    true = sum(1 for q in p if abs(q[2] - 90) < 45) / len(p)
    shares.append((true, radial_share(img)))
ax = fig.add_subplot(gs[:, 3])
lab = ["а", "б", "в"]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2, [s[0] * 100 for s in shares], w * 0.92, color=MUTED, label="на самом деле\n(подсчёт пластинок)")
b2 = ax.bar(x + w / 2, [s[1] * 100 for s in shares], w * 0.92, color=BLUE, label="по спектру")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, lab); ax.set_ylim(0, 115); ax.set_ylabel("доля радиальных, %")
ax.set_title("г) Доля радиальных:\nподсчёт и спектр", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left"); tidy(ax)
save(fig, "fig2_orientation.png")
print("доля радиальных (истина, спектр):", [(round(a, 2), round(b, 2)) for a, b in shares])

# ===== Рис. 3. Длина пластинок и шаг между ними =====
NREP = 8
shorts = [render(random_plates(240, 0, 10, jitter=0)) for _ in range(NREP)]
longs = [render(random_plates(60, 0, 40, jitter=0)) for _ in range(NREP)]


def stacked(jit):
    """Окружные пластинки рядами через 25 мкм по радиусу; jit — разброс рядов, мкм."""
    out = []
    for row in np.arange(12.5, 200, 25):
        for _ in range(12):
            out.append((rng.uniform(0, 200), row + rng.normal(0, jit), rng.normal(0, 3), 20))
    return out


regs = [render(stacked(0.0)) for _ in range(NREP)]
jits = [render(stacked(4.0)) for _ in range(NREP)]
rnds = [render(random_plates(96, 0, 20, jitter=3)) for _ in range(NREP)]


def mean_spec(imgs):
    Ps = [spectrum(im)[0] for im in imgs]
    _, kx, ky = spectrum(imgs[0])
    return np.mean(Ps, axis=0), kx, ky


def profile_across(imgs):
    """Поперёк полосы: по частоте дуги, усреднено по частотам радиуса 0.03–0.3."""
    P, kx, ky = mean_spec(imgs)
    sel = (np.abs(ky) > 0.03) & (np.abs(ky) < 0.3)
    pr = P[sel, :].mean(axis=0)
    m = kx >= 0
    return kx[m], pr[m] / pr[m].max()


def profile_along(imgs):
    """Вдоль полосы: по частоте радиуса, частоты дуги |kx| < 0.02."""
    P, kx, ky = mean_spec(imgs)
    pr = P[:, np.abs(kx) < 0.02].mean(axis=1)
    m = ky > 0.005
    return ky[m], pr[m]


fig = plt.figure(figsize=(14, 11), facecolor=BG)
gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.38)
show_img(fig.add_subplot(gs[0, 0]), shorts[0], "а) Короткие, 10 мкм")
show_spec(fig.add_subplot(gs[0, 1]), shorts[0], "спектр а)", ylabel=False)
show_img(fig.add_subplot(gs[0, 2]), longs[0], "б) Длинные, 40 мкм", ylabel=False)
show_spec(fig.add_subplot(gs[0, 3]), longs[0], "спектр б)", ylabel=False)
ax = fig.add_subplot(gs[1, :2])
for imgs, col, L in ((shorts, ORANGE, 10), (longs, BLUE, 40)):
    kk, pr = profile_across(imgs)
    ax.plot(kk, pr, color=col, lw=2, label=f"длина {L} мкм")
    ax.axvline(1 / L, color=col, lw=1, ls=(0, (3, 3)))
    ax.text(1 / L + 0.003, 0.9, f"1/{L}", color=col, fontsize=9)
ax.set_xlim(0, 0.2); ax.set_ylim(0, 1.05)
ax.set_xlabel("частота вдоль дуги, 1/мкм (поперёк полосы в спектре)")
ax.set_ylabel("сила, отн.")
ax.set_title("в) Полоса в спектре кончается около 1 / длина пластинки", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False); tidy(ax)
show_img(fig.add_subplot(gs[1, 2]), regs[0], "г) Ряды через 25 мкм", ylabel=False)
show_spec(fig.add_subplot(gs[1, 3]), regs[0], "спектр г)", ylabel=False)
show_img(fig.add_subplot(gs[2, 0]), jits[0], "д) Ряды с разбросом ±4 мкм")
show_img(fig.add_subplot(gs[2, 1]), rnds[0], "е) Без всякого порядка", ylabel=False)
ax = fig.add_subplot(gs[2, 2:])
kk, p_rnd = profile_along(rnds)
for imgs, col, lab_, ls in ((regs, BLUE, "г) ровные ряды", "-"), (jits, AQUA, "д) ряды с разбросом", (0, (5, 2))),
                            (rnds, ORANGE, "е) без порядка", (0, (1.5, 1.5)))):
    _, pr = profile_along(imgs)
    ax.plot(kk, ndi.gaussian_filter1d(pr / p_rnd, 0.7), color=col, lw=2, ls=ls, label=lab_)
for m in (1, 2, 3):
    ax.axvline(m / 25, color=GRID, lw=1, zorder=0)
ax.text(1 / 25 + 0.002, 0.3, "1/25", color=MUTED, fontsize=9)
ax.text(2 / 25 + 0.002, 0.3, "2/25", color=MUTED, fontsize=9)
ax.set_xlim(0.01, 0.13); ax.set_ylim(0, None)
ax.set_xlabel("частота по радиусу, 1/мкм (вдоль полосы в спектре)")
ax.set_ylabel("во сколько раз сильнее,\nчем у беспорядка")
ax.set_title("ж) Пик на 1/шаг есть, только если есть порядок", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper right"); tidy(ax)
save(fig, "fig3_length_spacing.png")


# ===== Рис. 4. Чего спектр не помнит: «где» =====
def chains_img():
    out = []
    for x0 in (30, 85, 140, 175):
        y, x = 0.0, x0
        while y < 200:
            a = 90 + rng.normal(0, 12)
            L = 18
            out.append((x, y + L / 2, a, L))
            y += L + 1.5
            x += rng.normal(0, 2.5)
    out += random_plates(50, 0, 18)
    return render(out)


ch = chains_img()
F = np.fft.fft2(ch - ch.mean())
phase = np.angle(np.fft.fft2(rng.normal(size=ch.shape)))
scr = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * phase)))
scr = (scr - scr.min()) / (scr.max() - scr.min())
scr = np.clip((scr - np.median(scr)) * 4, 0, 1)        # контраст для показа
fig, axs = plt.subplots(2, 2, figsize=(8.6, 8.4), facecolor=BG,
                        gridspec_kw=dict(hspace=0.45, wspace=0.3))
show_img(axs[0, 0], ch, "а) Радиальные цепочки\nчерез всё окно")
show_img(axs[0, 1], scr, "б) Тот же спектр,\nно «где» перемешано", ylabel=False)
show_spec(axs[1, 0], ch, "спектр а)")
P2 = np.abs(F) ** 2
show_spec(axs[1, 1], ch, "спектр б) — тот же самый", ylabel=False)
save(fig, "fig4_no_where.png")


# ===== Рис. 5. Окна: спектр по кусочкам → «где» =====
W5, H5 = 400, 150
pl = []
for _ in range(260):
    x = rng.uniform(0, W5)
    p_rad = np.clip((x - 80) / 240, 0, 1)                 # слева окружные, справа радиальные
    a = 90 if rng.random() < p_rad else 0
    pl.append((x, rng.uniform(0, H5), a + rng.normal(0, 8), 18))
img5 = render(pl, W=W5, H=H5)
win = int(50 / PX); step = int(10 / PX)
xc, rs, tr = [], [], []
for c0 in range(0, img5.shape[1] - win + 1, step):
    sub = img5[:, c0:c0 + win]
    xc.append((c0 + win / 2) * PX)
    rs.append(radial_share(sub) * 100)
    tr.append(np.clip(((c0 + win / 2) * PX - 80) / 240, 0, 1) * 100)
fig = plt.figure(figsize=(14, 7.6), facecolor=BG)
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.3, width_ratios=[2.6, 1, 0.05])
ax = fig.add_subplot(gs[0, 0])
show_img(ax, img5, "а) Слева окружные, справа радиальные; рамка — скользящее окно 50 мкм", W=W5, H=H5)
ax.add_patch(plt.Rectangle((150, 1), 50, H5 - 2, fill=False, lw=1.5, ec=ORANGE))
ax.set_xticks([0, 100, 200, 300, 400])
show_spec(fig.add_subplot(gs[0, 1]), img5, "б) спектр всего окна")
ax = fig.add_subplot(gs[1, 0])
ax.plot(xc, tr, color=MUTED, lw=2, ls=(0, (4, 3)), label="задано при генерации")
ax.plot(xc, rs, color=BLUE, lw=2, label="по спектру окна 50 мкм")
ax.set_xlim(0, W5); ax.set_ylim(0, 105)
ax.set_xlabel("положение окна вдоль дуги, мкм"); ax.set_ylabel("доля радиальных, %")
ax.set_title("в) Спектр по кусочкам: доля радиальных вдоль дуги", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, loc="upper left"); tidy(ax)
save(fig, "fig5_windows.png")


# ===== Рис. 6. Масштабы: иглы и цепочки =====
def needles_in_chains():
    out = []
    for x0 in (25, 70, 120, 160):                       # радиальные цепочки
        y, x = 0.0, x0
        while y < 200:
            a = 0 if rng.random() < 0.85 else rng.uniform(0, 180)   # иглы в основном окружные
            out.append((x + rng.normal(0, 2.5), y, a + rng.normal(0, 10), 8))
            y += rng.uniform(2.5, 7)
            x += rng.normal(0, 1.0)
    return render(out)


nc = needles_in_chains()


def bandpass(img, p_lo, p_hi):
    f = img - img.mean()
    ky = np.fft.fftfreq(img.shape[0], PX)[:, None]
    kx = np.fft.fftfreq(img.shape[1], PX)[None, :]
    k = np.hypot(kx, ky)
    g = lambda p: np.exp(-(k * p / 2.0) ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(f) * (g(p_lo) - g(p_hi))))


low = bandpass(nc, 15, 120)
high = bandpass(nc, 2, 6)
sh_low = radial_share(nc, pmin=15, pmax=120)
sh_high = radial_share(nc, pmin=2, pmax=6)
fig = plt.figure(figsize=(13, 9), facecolor=BG)
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
show_img(fig.add_subplot(gs[0, 0]), nc, "а) Цепочки из коротких игл")
ax = fig.add_subplot(gs[0, 1])
v = np.percentile(np.abs(low), 99.5)
ax.imshow(-low, cmap="gray", vmin=-v, vmax=v, extent=(0, 200, 200, 0))
ax.set_title("б) Оставили только крупное\n(периоды 15–120 мкм)", fontsize=10.5, loc="left", color=INK)
ax.set_xlabel("вдоль дуги, мкм"); ax.set_xticks([0, 100, 200]); ax.set_yticks([0, 100, 200])
ax = fig.add_subplot(gs[0, 2])
v = np.percentile(np.abs(high), 99.5)
ax.imshow(-high, cmap="gray", vmin=-v, vmax=v, extent=(0, 200, 200, 0))
ax.set_title("в) Оставили только мелкое\n(периоды 2–6 мкм)", fontsize=10.5, loc="left", color=INK)
ax.set_xlabel("вдоль дуги, мкм"); ax.set_xticks([0, 100, 200]); ax.set_yticks([0, 100, 200])
ax = fig.add_subplot(gs[1, 0])
show_spec(ax, nc, "г) спектр: внутри — цепочки,\nснаружи — иглы", kmax=0.5, rings=(15, 6, 2))
ax.set_xticks([-0.4, 0, 0.4]); ax.set_yticks([-0.4, 0, 0.4])
ax = fig.add_subplot(gs[1, 1])
zoom = nc[200:300, 20:120]
ax.imshow(1 - zoom, cmap="gray", vmin=0, vmax=1, extent=(10, 60, 150, 100), interpolation="nearest")
ax.set_title("д) Увеличено: иглы лежат вдоль дуги,\nа цепочка идёт по радиусу", fontsize=10.5, loc="left", color=INK)
ax.set_xlabel("вдоль дуги, мкм"); ax.set_ylabel("по радиусу, мкм")
ax = fig.add_subplot(gs[1, 2])
bars = ax.bar(["иглы\n(мелкое)", "цепочки\n(крупное)"], [sh_high * 100, sh_low * 100],
              color=[ORANGE, BLUE], width=0.55)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f} %", ha="center", color=INK)
ax.set_ylim(0, 110); ax.set_ylabel("доля радиальных по спектру, %")
ax.set_title("е) Один снимок — два ответа", fontsize=10.5, loc="left", color=INK); tidy(ax)
save(fig, "fig6_scales.png")
print(f"рис. 6: радиальная доля иглы {sh_high:.2f}, цепочки {sh_low:.2f}")
