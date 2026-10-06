"""Сводка hill_runs.py: выгода g = W/ε_n для окружной и радиальной пластинки, разность Δg = g_рад − g_окр
(в упругости 0.364·σθ), добавка анизотропии Δ_Хилл = Δg(Хилл) − Δg(Мизес). python hill_report.py папка"""
import sys
import json
import glob
from collections import defaultdict

EPS_N, EPS_T = 0.0720, 0.0458


def load(d):
    r = defaultdict(dict)
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        r[(m["hill"], m["alpha"], m["load"])][m["orient"]] = m
    for k, v in list(r.items()):                 # Мизес без нагрузки: радиальная = окружной по симметрии
        if k[0] == "iso" and k[2] == "0" and "rad" not in v and "circ" in v:
            v["rad"] = v["circ"]
    return r


def table(d):
    r = load(d)
    rows = []
    for k, v in r.items():
        if "circ" in v and "rad" in v:
            c, q = v["circ"], v["rad"]
            rows.append(dict(hill=k[0], alpha=k[1], load=k[2], s_theta=c["s_theta"], s_z=c["s_z"],
                             g_circ=c["g"], g_rad=q["g"], dg=q["g"] - c["g"],
                             dg_el=(EPS_N - EPS_T) / EPS_N * c["s_theta"],
                             pl_circ=c["pl_area"], pl_rad=q["pl_area"]))
    by = {(x["hill"], x["alpha"], x["load"]): x for x in rows}
    for x in rows:
        iso = by.get(("iso", 1.0, x["load"]))
        x["d_hill"] = x["dg"] - iso["dg"] if iso else float("nan")
    rows.sort(key=lambda x: (x["hill"], x["alpha"], x["s_z"] > 0, x["s_theta"]))
    return rows


if __name__ == "__main__":
    for x in table(sys.argv[1]):
        print(f"{x['hill']:7s} α {x['alpha']:<4g} {x['load']:5s} σθ {x['s_theta']:5.0f} σz {x['s_z']:5.1f} | "
              f"g окр {x['g_circ']:8.1f} рад {x['g_rad']:8.1f} | Δg {x['dg']:7.1f} (упр. {x['dg_el']:5.1f}) "
              f"Δ_Хилл {x['d_hill']:6.1f} | пласт. зона {x['pl_circ']:5.1f} / {x['pl_rad']:5.1f} мкм²")
