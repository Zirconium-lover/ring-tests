"""Прогоны автомата по списку случаев (JSON), параллельно. Каждый случай — словарь Params
плюс поле name; результат — name.json (метрики) и name.npz (поле гидрида, пластинки).
Поле sigma_top/sigma_bottom (МПа) задаёт линейный профиль напряжения по толщине
(верх — наружная поверхность)."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run
from ca_analysis import packet_metrics, image_rhf, interdistance

OUT = sys.argv[1]


def job(case):
    case = dict(case)
    name = case.pop("name")
    fn = os.path.join(OUT, name + ".json")
    if os.path.exists(fn):
        return
    if "sigma_top" in case:
        top, bot = case.pop("sigma_top"), case.pop("sigma_bottom")
        case["sigma_app"] = 0.5 * (top + bot)
        case["sigma_app_grad"] = top - bot
    for k in ("size_um", "grain_um", "chi0_profile"):
        if k in case:
            case[k] = tuple(case[k])
    t0 = time.time()
    r = run(Params(**case))
    m, C, T = packet_metrics(r)
    m["RHF_image"] = image_rhf(r) if r["hyd"].size <= 1.5e6 else None
    m["dist_ND"], m["n_ND"] = interdistance(r, r["params"].dx, 0)
    m["dist_TD"], m["n_TD"] = interdistance(r, r["params"].dx, 1)
    m.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in case.items()})
    m["name"] = name; m["time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, name + ".npz"), hyd=r["hyd"].astype(np.float32),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)
    print(name, f"{m['time_s']:.0f} с", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cases = json.load(open(sys.argv[2]))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for _ in pool.imap_unordered(job, cases):
            pass
    print("готово", len(cases))
