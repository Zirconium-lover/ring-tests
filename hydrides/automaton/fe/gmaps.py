import numpy as np
EPS_N, EPS_T = 0.0720, 0.0458


def gmap(sxx, syy, sxy, szz, psi_deg):
    """Выгода новой пластинки с углом ψ (против часовой от x), МПа: g = σ:ε*/ε_n."""
    p = np.radians(psi_deg)
    tx, ty = np.cos(p), np.sin(p)
    nx, ny = -np.sin(p), np.cos(p)
    e11 = EPS_N * nx ** 2 + EPS_T * tx ** 2
    e22 = EPS_N * ny ** 2 + EPS_T * ty ** 2
    e12 = EPS_N * nx * ny + EPS_T * tx * ty
    return (e11 * sxx + e22 * syy + 2 * e12 * sxy + EPS_T * szz) / EPS_N


def case(z, name):
    if name.startswith("fft"):
        return {k: z[f"{name}_{k}"] for k in ("sxx", "syy", "sxy", "szz")}
    return {k: z[f"{name}_{k}"] for k in ("sxx", "syy", "sxy", "szz")}
