"""Списки случаев для опытов на откалиброванном автомате."""
import json, sys
b, cap = float(sys.argv[1]), float(sys.argv[2])
base = dict(beta=b, sigma_cap=cap)
D = "/tmp/claude-0/-home-user/16339a70-3da3-5a74-97cf-f17fbd8cb16f/scratchpad/ca_runs/"
# A. большое поле: шаг пакетов (Lepine: 57 мкм по ND при 0 МПа, ~70 мкм по TD при 200–250 МПа)
A = [dict(base, name=f"big_s{s}_seed{k}", size_um=[480, 480], sigma_app=s, seed=k) for s in (0, 250) for k in (1, 2)]
# B. текстура: χ0 по Кернсу (Zr-1Nb труба ≈ 22–26, Zry-4 лист ≈ 30, Э635 внутр. ≈ 33, средн./нар. ≈ 38.5, др. образец ≈ 41.5)
B = [dict(base, name=f"tex_chi{c}_s{s}_seed{k}", chi0=c, sigma_app=s, seed=k)
     for c in (22, 30, 33, 38.5, 41.5) for s in (0, 100, 150, 200, 250) for k in (1, 2, 3)]
# C. стенка Э635 0.8 мм с текстурой по слоям: без нагрузки, равномерно +200, изгиб ±200 МПа
prof = [38.3, 38.7, 32.7]
C = []
for k in (1, 2):
    C.append(dict(base, name=f"wall_free_seed{k}", size_um=[800, 240], chi0_profile=prof, seed=k))
    C.append(dict(base, name=f"wall_unif200_seed{k}", size_um=[800, 240], chi0_profile=prof, sigma_app=200, seed=k))
    C.append(dict(base, name=f"wall_bend200_seed{k}", size_um=[800, 240], chi0_profile=prof, sigma_top=200, sigma_bottom=-200, seed=k))
# D. без взаимодействия пластинок (sigma_cap = 0): откуда пакеты
Dc = [dict(base, name=f"noint_s{s}_seed{k}", sigma_cap=0.0, sigma_app=s, seed=k) for s in (0, 250) for k in (1, 2)]
for nm, L in (("A", A), ("B", B), ("C", C), ("D", Dc)):
    json.dump(L, open(D + f"cases_{nm}.json", "w"))
    print(nm, len(L))
