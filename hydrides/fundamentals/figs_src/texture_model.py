"""Схематичная текстура трубы и операции с ней: выборка ориентаций зёрен, полюсные фигуры, обратные полюсные фигуры,
плотность полюсов для рентгеновской съёмки с поверхности трубы и факторы Кернса.

Системы координат. Образец (труба): R — радиус, T — окружность, A — ось трубы (R × T = A). Кристалл ГПУ: декартов
базис e1 ∥ a1 = [2-1-10], e2 ∥ [01-10], e3 ∥ c. Ориентация зерна — матрица g, переводящая векторы кристалла в
координаты образца: v_sample = g @ v_crystal.

Модель текстуры холоднодеформированной (и отожжённой для снятия напряжений) трубы: ось c лежит около плоскости R–T и
наклонена от R к T на ±χ (χ ~ N(35°, 12°)), к оси трубы — на η ~ N(0, 15°); направление ⟨10-10⟩ (e2) — около оси
трубы с разбросом поворота вокруг c ψ ~ N(0, 15°). Для рекристаллизованной трубы вокруг c добавлен поворот на 30°:
вдоль оси трубы тогда ⟨11-20⟩."""
import numpy as np

CA = 5.147 / 3.232


def rot(axis, ang):
    """Матрица поворота вокруг оси 0/1/2 на углы ang (массив, рад): форма (n, 3, 3)."""
    c, s = np.cos(ang), np.sin(ang)
    n = len(np.atleast_1d(ang))
    M = np.zeros((n, 3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    M[:, axis, axis] = 1
    M[:, i, i] = c; M[:, j, j] = c; M[:, i, j] = -s; M[:, j, i] = s
    return M


def sample(n=200000, kind="cwsr", seed=3, chi0=35.0, dchi=12.0, deta=15.0, dpsi=15.0):
    """Ориентации n зёрен, форма (n, 3, 3). kind: 'cwsr', 'rxa' или 'random'."""
    rng = np.random.default_rng(seed)
    if kind == "random":
        q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
        w, x, y, z = q.T
        return np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    M0 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)      # e1→T, e2→A, e3→R (столбцы — образы e1, e2, e3)
    chi = np.radians(rng.choice([-1, 1], n) * rng.normal(chi0, dchi, n))
    eta = np.radians(rng.normal(0, deta, n))
    psi = np.radians(rng.normal(0, dpsi, n) + (30.0 if kind == "rxa" else 0.0))
    return rot(1, -eta) @ rot(2, chi) @ M0[None] @ rot(2, psi)


def hkil_normal(h, k, l):
    """Нормаль к плоскости (hk.l) ГПУ в декартовом базисе кристалла (единичный вектор)."""
    a1 = np.array([1.0, 0, 0]); a2 = np.array([-0.5, np.sqrt(3) / 2, 0])
    # обратная решётка: b1 ⟂ a2, c; b2 ⟂ a1, c (масштаб a = 1)
    b1 = np.array([1.0, 1 / np.sqrt(3), 0]); b2 = np.array([0.0, 2 / np.sqrt(3), 0]); b3 = np.array([0, 0, 1 / CA])
    v = h * b1 + k * b2 + l * b3
    return v / np.linalg.norm(v)


def family(h, k, l):
    """Все различные нормали семейства {hk.l} с учётом симметрии 6/mmm (± считаются отдельно)."""
    out = []
    for ang in np.radians(np.arange(0, 360, 60)):
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        for sgn in (1, -1):
            for mir in (False, True):
                n = hkil_normal(h, k, l)
                if mir:
                    n = n * np.array([1, -1, 1])
                n = R @ n * np.array([1, 1, sgn])
                if not any(np.allclose(n, m, atol=1e-6) for m in out):
                    out.append(n)
    return np.array(out)


def poles(G, h, k, l):
    """Полюсы семейства во всех зёрнах в координатах образца (верхняя полусфера относительно R)."""
    fam = family(h, k, l)
    v = np.einsum("nij,mj->nmi", G, fam).reshape(-1, 3)
    v[v[:, 0] < 0] *= -1
    return v


def kearns(G):
    """Факторы Кернса (f_R, f_T, f_A) — средние cos² угла между осью c и направлениями образца."""
    c = G[:, :, 2]
    return (c ** 2).mean(0)


def to_fundamental(u):
    """Направление в кристалле → (угол от c, азимут от a1 в секторе 0…30°) по симметрии 6/mmm."""
    u = u / np.linalg.norm(u, axis=1)[:, None]
    alpha = np.degrees(np.arccos(np.abs(u[:, 2])))
    phi = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 60
    phi = np.where(phi > 30, 60 - phi, phi)
    return alpha, phi


def pole_density(G, n_cryst, d_sample=(1, 0, 0), delta=5.0):
    """Плотность полюсов (в долях случайной) семейства n_cryst вдоль направления образца d_sample: доля зёрен,
    у которых какой-то полюс семейства лежит в конусе delta вокруг d_sample, делённая на ту же долю для случайной
    текстуры. Именно это меряет симметричная съёмка θ–2θ с поверхности, нормаль которой — d_sample."""
    fam = n_cryst
    u = np.einsum("nji,j->ni", G, np.asarray(d_sample, float))             # d_sample в координатах кристалла
    cosmax = np.max(np.abs(u @ fam.T), axis=1)
    hit = np.mean(cosmax > np.cos(np.radians(delta)))
    m = len(fam)                                                           # число различных ±нормалей
    rand = m * (1 - np.cos(np.radians(delta))) / 2
    return hit / rand
