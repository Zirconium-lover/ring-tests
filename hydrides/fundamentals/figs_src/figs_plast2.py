"""Рисунки к дефектам и пластичности ГПУ: системы скольжения и карты факторов Шмида; факторы Шмида в текстуре трубы;
рекристаллизация (JMAK) и свойства после отжига; переход ГПУ → ГЦК частичными дислокациями Шокли."""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import BoundaryNorm

from style import *
import texture_model as tm

CA = tm.CA
A1 = np.array([1.0, 0, 0]); A2 = np.array([-0.5, np.sqrt(3) / 2, 0]); A3 = -(A1 + A2); C = np.array([0, 0, CA])


def plane_normal(h, k, l):
    return tm.hkil_normal(h, k, l)


def slip_systems():
    """Системы скольжения ГПУ: призматическое ⟨a⟩, базисное ⟨a⟩, пирамидальное ⟨c+a⟩ {10-11}⟨11-23⟩."""
    avecs = [A1, A2, A3]
    out = {"prism": [], "basal": [], "pyr": []}
    for ang in np.radians([30, 90, 150]):
        n = np.array([np.cos(ang), np.sin(ang), 0])
        d = np.array([np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2), 0])
        out["prism"].append((n, d))
    for a in avecs:
        out["basal"].append((np.array([0, 0, 1.0]), a / np.linalg.norm(a)))
    pyr_n = []
    for ang in np.radians(np.arange(30, 360, 60)):
        n0 = np.array([np.cos(ang), np.sin(ang), 0]) * (2 / np.sqrt(3)) + np.array([0, 0, 1 / CA])   # нормаль (10-11)-типа
        pyr_n.append(n0 / np.linalg.norm(n0))
    for a in [A1, A2, A3, -A1, -A2, -A3]:
        d = a + C; d /= np.linalg.norm(d)
        for n in pyr_n + [-p for p in pyr_n]:
            if abs(n @ d) < 1e-9 and n[2] > 0:
                out["pyr"].append((n, d))
    return out


def schmid_max(t, systems):
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    return np.max(np.stack([np.abs((t @ n) * (t @ d)) for n, d in systems], -1), -1)


def f_schmid():
    """Наибольший фактор Шмида при растяжении вдоль направления из стандартного треугольника: а) призматическое,
    б) базисное, в) пирамидальное ⟨c+a⟩ скольжение; г) распределение призматического фактора в трубе для
    растяжения по окружности T и по радиусу R (схематичная текстура)."""
    sy = slip_systems()
    print({k: len(v) for k, v in sy.items()})
    fig = plt.figure(figsize=(W1, 2.35))
    al = np.radians(np.linspace(0, 90, 91)); ph = np.radians(np.linspace(0, 30, 31))
    AL, PH = np.meshgrid(al, ph, indexing="ij")
    T = np.stack([np.sin(AL) * np.cos(PH), np.sin(AL) * np.sin(PH), np.cos(AL)], -1)
    rho = np.sqrt(2) * np.sin(AL / 2)
    X, Y = rho * np.cos(PH), rho * np.sin(PH)
    levels = np.linspace(0, 0.5, 11)
    norm = BoundaryNorm(levels, 256)
    cs = None
    for i, (key, nm) in enumerate((("prism", r"призм. $\langle a\rangle$"), ("basal", r"базисное $\langle a\rangle$"),
                                   ("pyr", r"пирам. $\langle c{+}a\rangle$"))):
        ax = fig.add_axes([0.0 + i * 0.205, 0.30, 0.2, 0.62])
        m = schmid_max(T, sy[key])
        cs = ax.contourf(X, Y, m, levels=levels, cmap="Purples", norm=norm)
        ax.add_patch(Wedge((0, 0), 1, 0, 30, fill=False, ec=SUB, lw=0.8))
        ax.text(0.0, 0.05, r"$[0001]$", ha="center", va="bottom", fontsize=6.6)
        ax.text(1.0, -0.03, r"$\langle 2\bar1\bar10\rangle$", ha="center", va="top", fontsize=6.6)
        ax.text(0.87, 0.53, r"$\langle 10\bar10\rangle$", ha="center", va="bottom", fontsize=6.6)
        ax.text(0.5, -0.22, nm, ha="center", va="top", fontsize=7.8)
        ax.set_xlim(-0.2, 1.1); ax.set_ylim(-0.4, 0.68); ax.set_aspect("equal"); ax.axis("off")
        label(ax, "абв"[i], 0.1, 0.92)
    cax = fig.add_axes([0.08, 0.13, 0.45, 0.04])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5]); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("наибольший фактор Шмида $m$", fontsize=7.4)
    g = fig.add_axes([0.70, 0.2, 0.29, 0.72])
    G = tm.sample(60000)
    for d, c, nm in (((0, 1, 0), CIRC, "по окружности T"), ((1, 0, 0), RAD, "по радиусу R"), ((0, 0, 1), DIM, "по оси A")):
        u = np.einsum("nji,j->ni", G, np.asarray(d, float))
        m = schmid_max(u, sy["prism"])
        g.hist(m, bins=np.linspace(0, 0.5, 26), density=True, histtype="step", color=c, lw=1.3, label=nm)
    g.set_xlabel(r"призматический $m$", fontsize=8); g.set_ylabel("плотность", fontsize=8); g.legend(fontsize=6.6, loc="upper left")
    label(g, "г")
    save(fig, "p1_schmid")


def f_recryst():
    """а) доля рекристаллизованного объёма X(t) = 1 − exp(−k t^n), k по Аррениусу, при трёх температурах;
    б) схема свойств после часового отжига холоднодеформированного сплава: возврат, рекристаллизация, рост зерна."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    Q, n = 300e3, 2.0
    Tref, tref = 580 + 273.15, 1.0                       # при 580 °C половина — за 1 ч
    k_ref = np.log(2) / tref ** n
    t = np.logspace(-2, 3, 400)
    for Tc, c in ((540, CIRC), (580, LOAD), (620, RAD)):
        k = k_ref * np.exp(-Q / 8.314 * (1 / (Tc + 273.15) - 1 / Tref)) ** n
        a.plot(t, 1 - np.exp(-k * t ** n), color=c, label=f"{Tc}" + r"\,\textdegree C")
    a.set_xscale("log"); a.set_xlabel("время отжига, ч"); a.set_ylabel(r"доля рекристаллизации $X$"); a.legend(fontsize=7.6, loc="upper left")
    label(a, "а")
    T = np.linspace(300, 750, 400)
    t1 = 1.0
    k = k_ref * np.exp(-Q / 8.314 * (1 / (T + 273.15) - 1 / Tref)) ** n
    X = 1 - np.exp(-k * t1 ** n)
    rec = 1 / (1 + np.exp(-(T - 440) / 25))                   # возврат: снятие части упрочнения до рекристаллизации
    s_cw, s_rx = 1.0, 0.55
    sy = s_rx + (s_cw - s_rx) * (1 - X) * (1 - 0.25 * rec)
    d = np.where(X > 0.98, 1 + 0.004 * np.clip(T - 600, 0, None) ** 1.3, 1.0)
    sy = sy - 0.06 * (1 - 1 / np.sqrt(d)) * X
    el = 0.15 + 0.75 * X + 0.08 * rec * (1 - X)
    b.plot(T, sy, color=INK, label="предел текучести")
    b.plot(T, el, color=CIRC, label="пластичность")
    b.plot(T, X, color=RAD, ls="--", lw=1.0, label=r"доля рекристаллизации")
    for i, (x0, x1, nm) in enumerate(((300, 500, "возврат"), (500, 640, "рекристал-\nлизация"), (640, 750, "рост\nзерна"))):
        b.axvspan(x0, x1, color=[CIRC, LOAD, RAD][i], alpha=0.06)
        b.text((x0 + x1) / 2, 1.06, nm, ha="center", va="bottom", fontsize=7.0)
    b.axvline(480, color=SUB, lw=0.6, ls=":"); b.text(482, 0.02, "СРО", fontsize=7, color=SUB)
    b.axvline(580, color=SUB, lw=0.6, ls=":"); b.text(584, 0.12, "РО", fontsize=7, color=SUB)
    b.set_ylim(0, 1.25); b.set_xlabel(r"температура часового отжига, \textdegree C"); b.set_ylabel("отн. величина")
    b.legend(fontsize=6.8, loc="center left", bbox_to_anchor=(0.0, 0.45))
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "p3_recryst")


def f_shockley():
    """Переход ГПУ → ГЦК: а) укладка ABAB (вид вдоль ⟨11-20⟩, сдвиги слоёв показаны по горизонтали);
    б) частичная дислокация Шокли a/3⟨10-10⟩ прошла между 2-м и 3-м слоями — дефект упаковки (местная ГЦК);
    в) частичные на каждом втором слое дают ABCABC — решётку ГЦК (подрешётка Zr в δ-гидриде)."""
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.35))
    pos = {"A": 0.0, "B": 1 / 3, "C": 2 / 3}
    col = {"A": CIRC, "B": RAD, "C": HYDR}
    seqs = (list("ABABABAB"), list("ABCACACA"), list("ABCABCAB"))
    titles = ("ГПУ: ABAB", "дефект упаковки", "ГЦК: ABCABC")
    for ax, seq, tt in zip(axs, seqs, titles):
        for j, s in enumerate(seq):
            for i in range(4):
                ax.plot(i + pos[s], j * 0.55, "o", ms=8.5, color=col[s], mec=INK, mew=0.4)
            ax.text(-0.55, j * 0.55, s, color=col[s], fontsize=8.5, va="center", ha="center")
        ax.text(1.8, len(seq) * 0.55 + 0.05, tt, ha="center", fontsize=8.5)
        ax.set_xlim(-0.8, 4.0); ax.set_ylim(-0.4, len(seq) * 0.55 + 0.35); ax.axis("off")
    # стрелки сдвига
    axs[1].annotate("", xy=(3.75 - 1 / 3, 1.1 + 0.27), xytext=(3.75, 1.1 + 0.27), arrowprops=dict(arrowstyle="->", color=PLAST, lw=1.2))
    axs[1].plot([-0.3, 3.95], [0.55 * 1.5] * 2, color=PLAST, lw=0.9, ls="--")
    axs[1].text(4.0, 0.55 * 1.5 + 0.1, r"$\mathbf b_p$", color=PLAST, fontsize=8.5)
    for j in (1.5, 3.5, 5.5):
        axs[2].plot([-0.3, 3.95], [0.55 * j] * 2, color=PLAST, lw=0.7, ls="--")
    for ax, s in zip(axs, "абв"):
        label(ax, s, 0.08, 0.98)
    fig.tight_layout(w_pad=0.3)
    save(fig, "p4_shockley")


if __name__ == "__main__":
    f_schmid(); f_recryst(); f_shockley()
