#!/usr/bin/env python3
"""Сравнение серии этапа 1 (после tools/post_stage1.py по каждому расчёту).

    python3 tools/compare_stage1.py [--runs runs/stage1] [--figs notes/figs]

Графики:
  stage1_force.png    — сила на конусе на единицу высоты F_z/H от номинальной
                        деформации u_r/R_i, панели по μ, линии по H;
  stage1_profiles.png — ε_θ(φ) на внутренней поверхности в середине высоты
                        (сплошные) и у торца (пунктир) при ε_ном = 0.10 и 0.20, μ = 0.05;
  stage1_hotspot.png  — опасная точка (max PEEQ в середине высоты): η, σ_zz/σ_θθ
                        и концентрация ε_max/ε_ср от ε_ном, μ = 0.05;
  (три рисунка выше — по серии C3D8I; C3D4 с 6 тетраэдрами запирается);
  stage1_localization.png — наибольшая ε_θ по всей внутренней поверхности
                        от ε_ном: панели по μ, линии по H;
  stage1_height.png   — max по дуге ε_θ в каждом ряду z внутренней поверхности
                        (post/surfmap.json, tools/surfmap_stage1.py) по высоте;
  stage1_elements.png — H = 5, μ = 0.05 на C3D4, C3D8I и C3D20R (запирание);
  stage1_tetmesh.png  — то же для сеток C3D4 (6 и 24 тетраэдра на шестигранник);
Таблица runs/stage1/summary.md — значения при ε_ном = 0.10 и 0.20 и
отклонения от следующей по величине высоты (критерий 3 %).
"""
import argparse
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H_COL = {3.0: '#2a78d6', 5.0: '#eb6834', 8.0: '#1baf7a', 12.0: '#eda100', 16.0: '#e87ba4',
         20.0: '#008300'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
TARGETS = (0.10, 0.20)


def load(rdir):
    runs = []
    for s in sorted(glob.glob(os.path.join(rdir, '*', 'post', 'summary.json'))):
        d = os.path.dirname(os.path.dirname(s))
        summ = json.load(open(s))
        ts = [{k: float(v) for k, v in r.items() if v != ''}
              for r in csv.DictReader(open(os.path.join(d, 'post', 'timeseries.csv')))]
        prof = json.load(open(os.path.join(d, 'post', 'profiles.json')))
        sp = os.path.join(d, 'post', 'surfmap.json')
        surf = json.load(open(sp)) if os.path.exists(sp) else None
        runs.append(dict(name=summ['run'], H=summ['H'], mu=summ['mu'], elem=summ['elem'],
                         summ=summ, ts=ts, prof=prof, surf=surf))
    return runs


def interp(ts, key, x, xkey='eps_nom'):
    for a, b in zip(ts, ts[1:]):
        if a[xkey] <= x <= b[xkey] and key in a and key in b:
            w = (x - a[xkey]) / (b[xkey] - a[xkey]) if b[xkey] > a[xkey] else 0.0
            return a[key] + w * (b[key] - a[key])
    return float('nan')


def nearest_profile(run, x):
    best = min(run['ts'], key=lambda r: abs(r['eps_nom'] - x))
    return run['prof'].get('%.6f' % best['time']), best['eps_nom']


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def fig_force(runs, out, elem='C3D8I'):
    mus = sorted({r['mu'] for r in runs if r['elem'] == elem})
    fig, axs = plt.subplots(1, len(mus), figsize=(5.2 * len(mus), 4.4), sharey=True)
    axs = axs if len(mus) > 1 else [axs]
    for ax, mu in zip(axs, mus):
        for r in sorted((r for r in runs if r['mu'] == mu and r['elem'] == elem),
                        key=lambda r: r['H']):
            x = [p['eps_nom'] for p in r['ts']]
            y = [p['Fz'] / r['H'] / 1000.0 for p in r['ts']]
            ax.plot(x, y, color=H_COL.get(r['H'], INK), lw=2, label='H = %g мм' % r['H'])
        ax.set_title('μ = %g' % mu, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
        style(ax)
    axs[0].set_ylabel('сила на конусе / высота кольца, кН/мм', color=INK)
    axs[-1].legend(frameon=False, fontsize=9, loc='lower right')
    fig.suptitle('Этап 1, %s: сила на конусе F_z/H (без трения конус–сегмент)' % elem, fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def fig_profiles(runs, out, mu=0.05, elem='C3D4'):
    sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == elem), key=lambda r: r['H'])
    fig, axs = plt.subplots(1, len(TARGETS), figsize=(6.2 * len(TARGETS), 4.6), sharey=False)
    for ax, x in zip(axs, TARGETS):
        for r in sel:
            p, xe = nearest_profile(r, x)
            if not p:
                continue
            c = H_COL.get(r['H'], INK)
            ax.plot(p['IN_Z0']['phi'], p['IN_Z0']['eps'], color=c, lw=2,
                    label='H = %g: середина' % r['H'])
            ax.plot(p['IN_ZTOP']['phi'], p['IN_ZTOP']['eps'], color=c, lw=1.4, ls='--',
                    label='H = %g: торец' % r['H'])
        ax.axvline(21.458, color=MUTED, lw=0.8, ls=':')
        ax.text(21.3, ax.get_ylim()[0], 'край\nсегмента', fontsize=8, color=MUTED, ha='right',
                va='bottom')
        ax.set_title('ε_ном ≈ %.2f' % x, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('φ, град (0 — середина сегмента, 22.5 — середина зазора)', color=INK)
        ax.set_ylabel('окружная деформация ε_θ (лог.), внутр. поверхность', color=INK)
        style(ax)
    axs[0].legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    fig.suptitle('Этап 1, μ = %g, %s: окружная деформация по дуге — середина высоты и торец'
                 % (mu, elem),
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def fig_hotspot(runs, out, mu=0.05, elem='C3D4'):
    sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == elem), key=lambda r: r['H'])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for r in sel:
        c = H_COL.get(r['H'], INK)
        x = [p['eps_nom'] for p in r['ts']]
        axs[0].plot(x, [p.get('hs_z0_eta_avg', float('nan')) for p in r['ts']], color=c, lw=2,
                    label='H = %g мм' % r['H'])
        axs[1].plot(x, [p.get('hs_z0_szz_stt_avg', float('nan')) for p in r['ts']], color=c, lw=2)
        axs[2].plot(x, [p['epsmax_IN_Z0'] / p['epsm_IN_Z0'] if p['epsm_IN_Z0'] > 1e-4
                        else float('nan') for p in r['ts']], color=c, lw=2)
    titles = ('η = σ_m/σ_экв (среднее в r ≤ 0.15 мм)', 'σ_zz/σ_θθ (то же)', 'концентрация ε_θ,max/ε_θ,ср (внутр.)')
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
        style(ax)
    axs[0].axhline(1 / 3, color=MUTED, lw=0.8, ls=':')
    axs[0].axhline(1 / math.sqrt(3), color=MUTED, lw=0.8, ls=':')
    axs[0].text(0.005, 1 / 3 + 0.005, 'одноосн. 1/3', fontsize=8, color=MUTED)
    axs[0].text(0.005, 1 / math.sqrt(3) + 0.005, 'плоск. деф. 0.577', fontsize=8, color=MUTED)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Этап 1, μ = %g, %s: опасная точка (max PEEQ в середине высоты)' % (mu, elem),
                 fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


E_COL = {'C3D4': '#2a78d6', 'C3D8I': '#eb6834', 'C3D20R': INK,
         'C3D4-x24': '#1baf7a', 'C3D4-h0.05': '#e87ba4'}


def fig_elements(runs, out, elems=('C3D4', 'C3D8I', 'C3D20R'), H=5.0, mu=0.05,
                 title='тип элемента (C3D20R — эталон)'):
    """Один и тот же расчёт (H, μ) на разных сетках: запирание C3D4."""
    sel = [r for e in elems for r in runs if r['H'] == H and r['mu'] == mu and r['elem'] == e]
    if len(sel) < 2:
        return
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax = axs[0][0]
    for r in sel:
        p, xe = nearest_profile(r, 0.20)
        if not p:
            continue
        c = E_COL[r['elem']]
        ax.plot(p['IN_Z0']['phi'], p['IN_Z0']['eps'], color=c, lw=2, label='%s: середина' % r['elem'])
        ax.plot(p['IN_ZTOP']['phi'], p['IN_ZTOP']['eps'], color=c, lw=1.4, ls='--',
                label='%s: торец' % r['elem'])
    ax.axvline(21.458, color=MUTED, lw=0.8, ls=':')
    ax.set_title('ε_θ(φ) на внутренней поверхности, ε_ном ≈ 0.20', fontsize=11, color=INK,
                 loc='left')
    ax.set_xlabel('φ, град (0 — середина сегмента, 22.5 — середина зазора)', color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc='upper left')
    panels = ((axs[0][1], 'epsmax_IN_Z0', 'epsmax_IN_ZTOP',
               'ε_θ,max на внутр. поверхности: середина (—), торец (- -)'),
              (axs[1][0], 'hs_z0_eta_avg', None, 'η в опасной точке (z = 0), среднее в r ≤ 0.15 мм'),
              (axs[1][1], 'hs_z0_szz_stt_avg', None, 'σ_zz/σ_θθ там же, среднее в r ≤ 0.15 мм'))
    for ax, k1, k2, t in panels:
        for r in sel:
            x = [p['eps_nom'] for p in r['ts']]
            ax.plot(x, [p.get(k1, float('nan')) for p in r['ts']], color=E_COL[r['elem']], lw=2,
                    label=r['elem'])
            if k2:
                ax.plot(x, [p.get(k2, float('nan')) for p in r['ts']], color=E_COL[r['elem']],
                        lw=1.4, ls='--')
        ax.set_title(t, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
    axs[1][0].axhline(1 / 3, color=MUTED, lw=0.8, ls=':')
    axs[1][0].text(0.005, 1 / 3 + 0.005, 'одноосн. 1/3', fontsize=8, color=MUTED)
    axs[1][0].legend(frameon=False, fontsize=9)
    for row in axs:
        for ax in row:
            style(ax)
    fig.suptitle('Этап 1, H = %g мм, μ = %g: %s' % (H, mu, title),
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def fig_height(runs, out, elem='C3D8I'):
    """ε_θ,max по дуге в каждом ряду z (внутр. поверхность, surfmap.json) по высоте.

    Строки — μ, столбцы — ε_ном; ось x — расстояние от свободного торца,
    кривая кончается в середине высоты (точка). Чёрный пунктир — C3D20R."""
    have = [r for r in runs if r.get('surf')]
    mus = sorted({r['mu'] for r in have if r['elem'] == elem})
    if not mus:
        return
    xs = (0.10, 0.15, 0.20)
    fig, axs = plt.subplots(len(mus), len(xs), figsize=(5.4 * len(xs), 3.9 * len(mus)),
                            squeeze=False)
    for i, mu in enumerate(mus):
        sel = sorted((r for r in have if r['mu'] == mu and r['elem'] == elem), key=lambda r: r['H'])
        ref = [r for r in have if r['mu'] == mu and r['elem'] == 'C3D20R']
        for j, x in enumerate(xs):
            ax = axs[i][j]
            for r in sel + ref:
                f = min(r['surf']['frames'], key=lambda f: abs(f['eps_nom'] - x))
                d = [0.5 * r['H'] - z for z in f['z']]
                isref = r['elem'] == 'C3D20R'
                c = INK if isref else H_COL.get(r['H'], INK)
                ax.plot(d, f['emax'], color=c, lw=1.4 if isref else 2, ls='--' if isref else '-',
                        label='H = %g мм%s' % (r['H'], ', C3D20R' if isref else ''))
                ax.plot(d[0], f['emax'][0], 'o', color=c, ms=5)
            ax.set_title('μ = %g, ε_ном ≈ %.2f' % (mu, x), fontsize=10.5, color=INK, loc='left')
            if i == len(mus) - 1:
                ax.set_xlabel('расстояние от свободного торца, мм (точка — середина высоты)',
                              color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel('max по дуге ε_θ, внутр. пов.', color=INK)
            style(ax)
        axs[i][0].legend(frameon=False, fontsize=8)
    fig.suptitle('Этап 1, %s: наибольшая окружная деформация внутренней поверхности по высоте'
                 % elem, fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')

def fig_localization(runs, out, elem='C3D8I'):
    """Наибольшая ε_θ по всей внутренней поверхности от ε_ном: панели по μ, линии по H.

    Пунктир — номинальная деформация ln(1 + ε_ном); вертикаль — максимум силы."""
    have = [r for r in runs if r.get('surf') and r['elem'] == elem]
    mus = sorted({r['mu'] for r in have})
    if not mus:
        return
    fig, axs = plt.subplots(1, len(mus), figsize=(5.4 * len(mus), 4.6), sharey=True, squeeze=False)
    for ax, mu in zip(axs[0], mus):
        sel = sorted((r for r in have if r['mu'] == mu), key=lambda r: r['H'])
        for r in sel:
            fr = r['surf']['frames']
            ax.plot([f['eps_nom'] for f in fr], [max(f['emax']) for f in fr],
                    color=H_COL.get(r['H'], INK), lw=2, label='H = %g мм' % r['H'])
        xx = [0.0, 0.364]
        ax.plot(xx, [math.log(1 + x) for x in xx], color=MUTED, lw=1, ls='--')
        if sel:
            xf = sel[-1]['summ']['eps_nom_at_Fmax']
            ax.axvline(xf, color=MUTED, lw=0.8, ls=':')
            ax.text(xf + 0.004, 0.97, 'максимум силы', fontsize=8, color=MUTED, va='top',
                    transform=ax.get_xaxis_transform())
        ax.set_title('μ = %g' % mu, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc='center left')
    axs[0][0].set_ylabel('max ε_θ по внутренней поверхности', color=INK)
    fig.suptitle('Этап 1, %s: локализация — наибольшая окружная деформация '
                 '(пунктир — равномерная раздача)' % elem, fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def table(runs, out):
    keys = [('epsmax_IN_Z0', 'ε_θ,max сер.'), ('epsmax_IN_ZTOP', 'ε_θ,max торец'),
            ('hs_z0_eta_avg', 'η'), ('hs_z0_szz_stt_avg', 'σzz/σθθ'), ('tmin_Z0', 't_min/t0'),
            ('Fz', 'F_z, Н')]
    L = ['# Этап 1: сводка серии', '',
         'ε_ном = u_r/R_i. Величины «сер.» — внутренняя поверхность в середине высоты,',
         '«торец» — у свободного торца; η, σzz/σθθ — опасная точка (max PEEQ, z = 0),',
         'средние по элементам в радиусе 0.15 мм.',
         'Δ — относительное отличие от следующей по величине высоты при том же μ и типе элемента.',
         '']
    L.append('| расчёт | ε_ном при F_max | F_z,max, Н | итог |')
    L.append('|---|---|---|---|')
    for r in sorted(runs, key=lambda r: (r['elem'] != 'C3D4', r['elem'], r['mu'], r['H'])):
        s = r['summ']
        L.append('| %s | %.3f | %.0f | rc=%s, %s с, до ε_ном = %.3f |' % (
            r['name'], s['eps_nom_at_Fmax'], s['Fz_max'], s.get('return_code'),
            s.get('wall_seconds'), s['last_eps_nom']))
    for x in TARGETS:
        L += ['', '## ε_ном = %.2f' % x, '']
        L.append('| расчёт | ' + ' | '.join(k[1] for k in keys) + ' | Δ ε_max сер. | Δ η | Δ F_z/H |')
        L.append('|---' * (len(keys) + 4) + '|')
        groups = sorted({(r['elem'] != 'C3D4', r['elem'], r['mu']) for r in runs})
        for _, elem, mu in groups:
            sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == elem),
                         key=lambda r: r['H'])
            for i, r in enumerate(sel):
                vals = [interp(r['ts'], k[0], x) for k in keys]
                dd = ['', '', '']
                if i + 1 < len(sel):
                    n = sel[i + 1]
                    a = interp(r['ts'], 'epsmax_IN_Z0', x); b = interp(n['ts'], 'epsmax_IN_Z0', x)
                    c = interp(r['ts'], 'hs_z0_eta_avg', x); d = interp(n['ts'], 'hs_z0_eta_avg', x)
                    e = interp(r['ts'], 'Fz', x) / r['H']; f = interp(n['ts'], 'Fz', x) / n['H']
                    dd = ['%+.1f %%' % (100 * (a / b - 1)), '%+.1f %%' % (100 * (c / d - 1)),
                          '%+.1f %%' % (100 * (e / f - 1))]
                L.append('| %s | ' % r['name'] + ' | '.join(
                    ('%.0f' % v if k[0] == 'Fz' else '%.4f' % v) for v, k in zip(vals, keys))
                    + ' | ' + ' | '.join(dd) + ' |')
    open(out, 'w').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs/stage1')
    ap.add_argument('--figs', default='notes/figs')
    ap.add_argument('--table', default=None, help='путь таблицы (по умолчанию RUNS/summary.md)')
    a = ap.parse_args()
    runs = load(a.runs)
    print('runs:', ', '.join(r['name'] for r in runs))
    os.makedirs(a.figs, exist_ok=True)
    fig_force(runs, os.path.join(a.figs, 'stage1_force.png'))
    fig_profiles(runs, os.path.join(a.figs, 'stage1_profiles.png'), elem='C3D8I')
    fig_hotspot(runs, os.path.join(a.figs, 'stage1_hotspot.png'), elem='C3D8I')
    fig_localization(runs, os.path.join(a.figs, 'stage1_localization.png'))
    fig_height(runs, os.path.join(a.figs, 'stage1_height.png'))
    fig_elements(runs, os.path.join(a.figs, 'stage1_elements.png'))
    fig_elements(runs, os.path.join(a.figs, 'stage1_tetmesh.png'),
                 elems=('C3D4', 'C3D4-x24', 'C3D20R'),
                 title='сетки C3D4 против эталона C3D20R')
    tpath = a.table or os.path.join(a.runs, 'summary.md')
    table(runs, tpath)
    print('written figures and', tpath)


if __name__ == '__main__':
    main()
