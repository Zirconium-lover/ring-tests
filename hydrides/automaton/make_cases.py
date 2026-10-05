"""Список случаев для опытов на откалиброванном автомате (то, по чему построены рис. 4–6).
Запуск: python make_cases.py cases.json && python run_cases.py папка cases.json 4"""
import json, sys
base = dict(beta=0.12, sigma_cap=90.0, capture_um=35.0)
prof = [38.3, 38.7, 32.7]                       # χ₀ по слоям канала Э635 (наружный, средний, внутренний)
L = []
for k in (1, 2):
    L.append(dict(base, name=f"wall_free_seed{k}", size_um=[800, 240], chi0_profile=prof, seed=k))
    L.append(dict(base, name=f"wall_unif200_seed{k}", size_um=[800, 240], chi0_profile=prof, sigma_app=200, seed=k))
    L.append(dict(base, name=f"wall_bend200_seed{k}", size_um=[800, 240], chi0_profile=prof, sigma_top=200, sigma_bottom=-200, seed=k))
    L.append(dict(base, name=f"wall_bend100_seed{k}", size_um=[800, 240], chi0_profile=prof, sigma_top=100, sigma_bottom=-100, seed=k))
    # выступ ячейки решётки Zr-1Nb: стенка 250 мкм, +135 снаружи / −150 внутри, ≈120 wppm
    L.append(dict(base, name=f"dimple_free_seed{k}", size_um=[250, 240], chi0=24.0, frac=0.0073, seed=k))
    L.append(dict(base, name=f"dimple_bend_seed{k}", size_um=[250, 240], chi0=24.0, frac=0.0073, sigma_top=135, sigma_bottom=-150, seed=k))
L.append(dict(base, name="big_s0_seed1", size_um=[480, 480], seed=1))          # шаг пакетов
L.append(dict(base, name="big_s250_seed1", size_um=[480, 480], sigma_app=250, seed=1))
L += [dict(base, name=f"tex_chi{c}_s{s}_seed{k}", chi0=c, sigma_app=s, seed=k)   # текстура
      for c in (22, 30, 33, 38.5, 41.5) for s in (0, 100, 150, 200, 250) for k in (1, 2, 3)]
L += [dict(base, name=f"bist_s{s}_seed{k}", sigma_app=s, seed=10 + k) for s in (100, 150, 200) for k in range(1, 9)]
json.dump(L, open(sys.argv[1] if len(sys.argv) > 1 else "cases.json", "w"))
print(len(L))
