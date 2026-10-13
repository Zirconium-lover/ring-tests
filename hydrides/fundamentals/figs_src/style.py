"""Оформление рисунков «Фундамента»: та же палитра, что в учебнике по модели (../textbook), шрифт — Computer Modern
через LaTeX (T2A), чтобы рисунки и текст выглядели одним документом. Цвет несёт смысл: окружные гидриды и
«основное» — синий, радиальные — красный, нагрузка — золотой, пластичность — фиолетовый, водород — зелёный."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB, DIM, FAINT = "#1d1d1f", "#5f6368", "#9aa0a6", "#eef0f2"
CIRC, RAD, LOAD, PLAST, HYDR = "#2a6fdb", "#d93a3a", "#d18b00", "#7b4ea3", "#1e9e6a"
METAL, METAL_E, HYD_F = "#f3f1ec", "#c9c4b8", "#9ec3f5"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
W1, W2 = 6.3, 3.05                     # ширина рисунка во всю строку и в половину, дюймы

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T2A]{fontenc}\usepackage[utf8]{inputenc}\usepackage[russian]{babel}"
                           r"\usepackage{amsmath,amssymb,bm}",
    "font.family": "serif", "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": SUB, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "lines.linewidth": 1.6, "legend.frameon": False, "axes.titlelocation": "left",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("рисунок", name)


def label(ax, s, x=-0.02, y=1.04):
    """Буква панели (а), (б), …"""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, ha="right", va="bottom")
