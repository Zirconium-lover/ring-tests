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

H_COL = {3.0: '#2a78d6', 5.0: '#eb6834', 8.0: '#1baf7a', 12.0: '#eda100'}
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
        runs.append(dict(name=summ['run'], H=summ['H'], mu=summ['mu'], elem=summ['elem'],
                         summ=summ, ts=ts, prof=prof))
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


def fig_force(runs, out):
    mus = sorted({r['mu'] for r in runs if r['elem'] == 'C3D4'})
    fig, axs = plt.subplots(1, len(mus), figsize=(5.2 * len(mus), 4.4), sharey=True)
    axs = axs if len(mus) > 1 else [axs]
    for ax, mu in zip(axs, mus):
        for r in sorted((r for r in runs if r['mu'] == mu and r['elem'] == 'C3D4'),
                        key=lambda r: r['H']):
            x = [p['eps_nom'] for p in r['ts']]
            y = [p['Fz'] / r['H'] / 1000.0 for p in r['ts']]
            ax.plot(x, y, color=H_COL.get(r['H'], INK), lw=2, label='H = %g мм' % r['H'])
        for r in runs:
            if r['mu'] == mu and r['elem'] != 'C3D4':
                ax.plot([p['eps_nom'] for p in r['ts']],
                        [p['Fz'] / r['H'] / 1000.0 for p in r['ts']],
                        color=INK, lw=1.2, ls='--', label='H = %g мм, %s' % (r['H'], r['elem']))
        ax.set_title('μ = %g' % mu, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
        style(ax)
    axs[0].set_ylabel('сила на конусе / высота кольца, кН/мм', color=INK)
    axs[-1].legend(frameon=False, fontsize=9, loc='lower right')
    fig.suptitle('Этап 1: сила на конусе F_z/H (без трения конус–сегмент)', fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def fig_profiles(runs, out, mu=0.05):
    sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == 'C3D4'), key=lambda r: r['H'])
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
    fig.suptitle('Этап 1, μ = %g: окружная деформация по дуге — середина высоты и торец' % mu,
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def fig_hotspot(runs, out, mu=0.05):
    sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == 'C3D4'), key=lambda r: r['H'])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for r in sel:
        c = H_COL.get(r['H'], INK)
        x = [p['eps_nom'] for p in r['ts']]
        axs[0].plot(x, [p.get('hs_z0_eta', float('nan')) for p in r['ts']], color=c, lw=2,
                    label='H = %g мм' % r['H'])
        axs[1].plot(x, [p.get('hs_z0_szz_stt', float('nan')) for p in r['ts']], color=c, lw=2)
        axs[2].plot(x, [p['epsmax_IN_Z0'] / p['epsm_IN_Z0'] if p['epsm_IN_Z0'] > 1e-4
                        else float('nan') for p in r['ts']], color=c, lw=2)
    for r in runs:
        if r['mu'] == mu and r['elem'] != 'C3D4':
            x = [p['eps_nom'] for p in r['ts']]
            axs[0].plot(x, [p.get('hs_z0_eta', float('nan')) for p in r['ts']], color=INK,
                        lw=1.2, ls='--', label='H = %g, %s' % (r['H'], r['elem']))
            axs[1].plot(x, [p.get('hs_z0_szz_stt', float('nan')) for p in r['ts']], color=INK,
                        lw=1.2, ls='--')
            axs[2].plot(x, [p['epsmax_IN_Z0'] / p['epsm_IN_Z0'] if p['epsm_IN_Z0'] > 1e-4
                            else float('nan') for p in r['ts']], color=INK, lw=1.2, ls='--')
    titles = ('трёхосность η = σ_m/σ_экв', 'σ_zz/σ_θθ', 'концентрация ε_θ,max/ε_θ,ср (внутр.)')
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
        style(ax)
    axs[0].axhline(1 / 3, color=MUTED, lw=0.8, ls=':')
    axs[0].axhline(1 / math.sqrt(3), color=MUTED, lw=0.8, ls=':')
    axs[0].text(0.005, 1 / 3 + 0.005, 'одноосн. 1/3', fontsize=8, color=MUTED)
    axs[0].text(0.005, 1 / math.sqrt(3) + 0.005, 'плоск. деф. 0.577', fontsize=8, color=MUTED)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Этап 1, μ = %g: опасная точка (max PEEQ в середине высоты)' % mu, fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def table(runs, out):
    keys = [('epsmax_IN_Z0', 'ε_θ,max сер.'), ('epsmax_IN_ZTOP', 'ε_θ,max торец'),
            ('hs_z0_eta', 'η'), ('hs_z0_szz_stt', 'σzz/σθθ'), ('tmin_Z0', 't_min/t0'),
            ('Fz', 'F_z, Н')]
    L = ['# Этап 1: сводка серии', '',
         'ε_ном = u_r/R_i. Величины «сер.» — внутренняя поверхность в середине высоты,',
         '«торец» — у свободного торца; η, σzz/σθθ — опасная точка (max PEEQ, z = 0).',
         'Δ — относительное отличие от следующей по величине высоты при том же μ.', '']
    L.append('| расчёт | ε_ном при F_max | F_z,max, Н | итог |')
    L.append('|---|---|---|---|')
    for r in sorted(runs, key=lambda r: (r['elem'], r['mu'], r['H'])):
        s = r['summ']
        L.append('| %s | %.3f | %.0f | rc=%s, %s с, до ε_ном = %.3f |' % (
            r['name'], s['eps_nom_at_Fmax'], s['Fz_max'], s.get('return_code'),
            s.get('wall_seconds'), s['last_eps_nom']))
    for x in TARGETS:
        L += ['', '## ε_ном = %.2f' % x, '']
        L.append('| расчёт | ' + ' | '.join(k[1] for k in keys) + ' | Δ ε_max сер. | Δ η | Δ F_z/H |')
        L.append('|---' * (len(keys) + 4) + '|')
        for mu in sorted({r['mu'] for r in runs}):
            sel = sorted((r for r in runs if r['mu'] == mu and r['elem'] == 'C3D4'),
                         key=lambda r: r['H'])
            for i, r in enumerate(sel):
                vals = [interp(r['ts'], k[0], x) for k in keys]
                dd = ['', '', '']
                if i + 1 < len(sel):
                    n = sel[i + 1]
                    a = interp(r['ts'], 'epsmax_IN_Z0', x); b = interp(n['ts'], 'epsmax_IN_Z0', x)
                    c = interp(r['ts'], 'hs_z0_eta', x); d = interp(n['ts'], 'hs_z0_eta', x)
                    e = interp(r['ts'], 'Fz', x) / r['H']; f = interp(n['ts'], 'Fz', x) / n['H']
                    dd = ['%+.1f %%' % (100 * (a / b - 1)), '%+.1f %%' % (100 * (c / d - 1)),
                          '%+.1f %%' % (100 * (e / f - 1))]
                L.append('| %s | ' % r['name'] + ' | '.join(
                    ('%.0f' % v if k[0] == 'Fz' else '%.4f' % v) for v, k in zip(vals, keys))
                    + ' | ' + ' | '.join(dd) + ' |')
        for r in runs:
            if r['elem'] != 'C3D4':
                vals = [interp(r['ts'], k[0], x) for k in keys]
                L.append('| %s | ' % r['name'] + ' | '.join(
                    ('%.0f' % v if k[0] == 'Fz' else '%.4f' % v) for v, k in zip(vals, keys))
                    + ' | | | |')
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
    fig_profiles(runs, os.path.join(a.figs, 'stage1_profiles.png'))
    fig_hotspot(runs, os.path.join(a.figs, 'stage1_hotspot.png'))
    tpath = a.table or os.path.join(a.runs, 'summary.md')
    table(runs, tpath)
    print('written figures and', tpath)


if __name__ == '__main__':
    main()
