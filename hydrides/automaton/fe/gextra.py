"""Поправка автомата на напряжения несовместности: МКЭ на той же раскладке зёрен, что у прогона.
g_extra = (ΔT/−100 К)·g_остывания + (σ_app/100 МПа)·(g_нагрузки − формула автомата − среднее)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ca_hydride import Params, make_grains  # noqa: E402
from incompat import solve  # noqa: E402
from gmaps import gmap, EPS_N, EPS_T  # noqa: E402


def fields(p: Params, work):
    """g_остывания (−100 К) и разброс g_нагрузки (100 МПа) для каждой клетки автомата; кэш в work."""
    tag = f"gx_{p.size_um[0]:g}x{p.size_um[1]:g}_dx{p.dx:g}_chi{p.chi0:g}_s{p.chi_s:g}_prof{'-'.join(map(str, p.chi0_profile))}_seed{p.seed}"
    fn = os.path.join(work, tag + ".npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["g_th"], z["g_mech"]
    gr, gpsi = make_grains(p, np.random.default_rng(p.seed))
    psi_deg = np.degrees(gpsi[gr])
    out = {}
    for case in ("thermal", "mech"):
        S = solve(os.path.join(work, "ccx"), tag + "_" + case, gr, gpsi, p.dx, case=case)
        g = np.zeros(gr.shape)
        for gid in np.unique(gr):
            m = gr == gid
            g[m] = gmap(S["sxx"][m], S["syy"][m], S["sxy"][m], S["szz"][m], np.degrees(gpsi[gid]))
        out[case] = g
    psi = gpsi[gr]
    e11 = EPS_N * np.sin(psi) ** 2 + EPS_T * np.cos(psi) ** 2
    formula = 100.0 * (e11 - 0.5 * (EPS_N + EPS_T)) / EPS_N
    g_th = out["thermal"] - out["thermal"].mean()
    g_mech = out["mech"] - formula
    g_mech -= g_mech.mean()
    np.savez_compressed(fn, g_th=g_th.astype(np.float32), g_mech=g_mech.astype(np.float32))
    return g_th, g_mech


def g_extra(p: Params, work, dT=-85.0):
    g_th, g_mech = fields(p, work)
    return (dT / -100.0) * g_th + (p.sigma_app / 100.0) * g_mech
