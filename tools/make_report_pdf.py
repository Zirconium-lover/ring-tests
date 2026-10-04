#!/usr/bin/env python3
"""PDF отчёта из Markdown-выгрузки документа и рисунков notes/figs.

    python3 tools/make_report_pdf.py notes/report_stage0_1.md -o notes/report_stage0_1.pdf

Markdown — выгрузка документа «Кольца Э635 на сегментной оправке — этапы 0 и 1»
(картинки в ней заменены строками «[image: подпись]»). Скрипт подставляет
вместо них рисунки из notes/figs по подписи, собирает HTML (A4, подписи под
рисунками, оглавление, номера страниц) и печатает его в PDF безголовым
Chromium. Путь к Chromium — переменная CHROME (по умолчанию тот, что
поставлен для Playwright). Нужен пакет markdown (pip install markdown).
"""
import argparse
import glob
import html
import os
import re
import subprocess
import sys

import markdown

FIGS = {
    'Схема испытания': 'setup_scheme.png',
    'Кривые упрочнения A и B': 'hardening_AB.png',
    'Геометрия и сетка, H = 5 мм': 'stage1_mesh_H5.png',
    'Сравнение C3D4, C3D8I и C3D20R': 'stage1_elements.png',
    'Сетки C3D4 у кромки сегмента': 'stage1_tetsplit.png',
    'Сетки C3D4 против эталона': 'stage1_tetmesh.png',
    'ε_θ по дуге, μ = 0.05': 'stage1_profiles.png',
    'Карты ε_θ на внутренней поверхности': 'stage1_maps.png',
    'Деформированное сечение z = 0': 'stage1_sections.png',
    'Наибольшая ε_θ от ε_ном': 'stage1_localization.png',
    'Наибольшая ε_θ от высоты кольца': 'stage1_hconv.png',
    'Наибольшая ε_θ по высоте кольца': 'stage1_height.png',
    'Сила на конусе на единицу высоты': 'stage1_force.png',
    'Опасная точка': 'stage1_hotspot.png',
}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm;
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt 'Liberation Sans', sans-serif; color: #6b6a63; }
  @bottom-left { content: "Кольца Э635 на сегментной оправке — этапы 0 и 1"; font: 8pt 'Liberation Sans', sans-serif; color: #6b6a63; } }
@page :first { @bottom-left { content: none; } }
html { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 9.6pt; color: #1f1f1e; line-height: 1.42; }
body { margin: 0; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; color: #1f1f1e; }
.sub { color: #6b6a63; font-size: 10pt; margin-bottom: 14pt; }
h2 { font-size: 14pt; margin: 14pt 0 6pt; padding-top: 4pt; border-top: 2px solid #2a78d6; color: #1f1f1e;
     break-after: avoid; }
h2.newpage { break-before: page; margin-top: 0; }
p { margin: 0 0 6pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 7pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
strong { font-weight: 700; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #f1f0eb; padding: 0 2pt; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.3pt; line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #eef3fb; text-align: left; font-weight: 700; }
th, td { border: 0.6pt solid #c9c8c0; padding: 2.5pt 4pt; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafaf7; }
figure { margin: 6pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 205mm; }
figcaption { font-size: 8.4pt; color: #4a4943; text-align: left; margin-top: 3pt; font-style: italic; }
.toc { margin: 8pt 0 0; padding: 8pt 10pt; background: #f6f6f2; border-left: 3px solid #2a78d6; }
.toc ol { margin: 0; padding-left: 16pt; columns: 2; column-gap: 18pt; }
.toc li { margin: 1pt 0; }
.toc a { color: #1f1f1e; text-decoration: none; }
.check { list-style: none; padding-left: 4pt; }
.check li::before { content: "☐  "; }
.lead { font-size: 10.5pt; }
table.dense { font-size: 7.4pt; line-height: 1.2; }
table.dense th, table.dense td { padding: 1.2pt 3pt; }
"""

MONTHS = 'января февраля марта апреля мая июня июля августа сентября октября ноября декабря'.split()


def fig_file(alt):
    for key, f in FIGS.items():
        if alt.startswith(key):
            return f
    sys.exit('нет рисунка для подписи: %r' % alt)


def build_html(md_text, figs_dir):
    lines = md_text.splitlines()
    title = lines[0].lstrip('# ').strip()
    body = '\n'.join(lines[1:])
    # строка «Oct 4, 2026 · @…» — дата по-русски, без упоминания
    m = re.search(r'^(\w{3}) (\d+), (\d{4}) · @\S+\s*$', body, flags=re.M)
    sub = ''
    if m:
        mon = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split().index(m.group(1))
        sub = '%s %s %s г.' % (m.group(2), MONTHS[mon], m.group(3))
        body = body.replace(m.group(0), '')
    # картинки: «&#91;image: подпись\\]» → ![подпись](файл)
    def img(mm):
        alt = mm.group(1).replace('\\', '')
        return '![%s](%s)' % (alt, 'file://' + os.path.abspath(os.path.join(figs_dir, fig_file(alt))))
    body = re.sub(r'&#91;image: (.*?)\\\]', img, body)
    body = re.sub(r'^- \[ \] ', '- ', body, flags=re.M)
    h = markdown.markdown(body, extensions=['tables', 'sane_lists'])
    # рисунок + следующий курсивный абзац → <figure> с подписью
    h = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>\s*<p><em>(.*?)</em></p>',
               lambda m: '<figure><img alt="%s" src="%s"><figcaption>%s</figcaption></figure>'
               % (m.group(1), m.group(2), m.group(3)), h, flags=re.S)
    # длинные таблицы (приложение) — плотнее
    h = re.sub(r'<table>(.*?)</table>', lambda m: ('<table class="dense">' if m.group(1).count('<tr>') > 20
                                                    else '<table>') + m.group(1) + '</table>', h, flags=re.S)
    # список решений с галочками (последний список раздела «Этап 2»)
    h = h.replace('<p>Решения за вами:</p>\n<ul>', '<p>Решения за вами:</p>\n<ul class="check">')
    # якоря разделов и оглавление
    toc = []

    def anchor(m):
        n = len(toc) + 1
        toc.append((n, m.group(1)))
        # с новой страницы — крупные части: этап 0, этап 1, рекомендации, приложение
        newpage = any(m.group(1).startswith(k) for k in ('Этап 0', 'Этап 1', 'Рекомендации', 'Приложение'))
        cls = ' class="newpage"' if newpage else ''
        return '<h2 id="s%d"%s>%d. %s</h2>' % (n, cls, n, m.group(1))
    h = re.sub(r'<h2>(.*?)</h2>', anchor, h)
    toc_html = '<div class="toc"><b>Содержание</b><ol>%s</ol></div>' % ''.join(
        '<li><a href="#s%d">%s</a></li>' % (n, t) for n, t in toc)
    head = '<h1>%s</h1><div class="sub">%s · расчёты CalculiX (ccx-arch2 @ c75ad9b), репозиторий ring-tests</div>%s' % (
        html.escape(title), sub, toc_html)
    return ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>%s</title><style>%s</style>'
            '</head><body>%s%s</body></html>' % (html.escape(title), CSS, head, h))


def find_chrome():
    c = os.environ.get('CHROME')
    if c:
        return c
    hits = sorted(glob.glob('/opt/pw-browsers/chromium-*/chrome-linux/chrome'))
    if not hits:
        sys.exit('не найден Chromium: задайте CHROME')
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('md')
    ap.add_argument('-o', required=True)
    ap.add_argument('--figs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notes', 'figs'))
    ap.add_argument('--html', default=None, help='куда сохранить промежуточный HTML')
    a = ap.parse_args()
    page = build_html(open(a.md, encoding='utf-8').read(), a.figs)
    hpath = a.html or os.path.splitext(a.o)[0] + '.html'
    open(hpath, 'w', encoding='utf-8').write(page)
    cmd = [find_chrome(), '--headless', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
           '--no-pdf-header-footer', '--print-to-pdf=' + os.path.abspath(a.o), 'file://' + os.path.abspath(hpath)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not os.path.exists(a.o):
        sys.exit(r.stderr[-2000:])
    print('wrote', a.o, os.path.getsize(a.o), 'bytes')


if __name__ == '__main__':
    main()
