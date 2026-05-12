# -*- coding: utf-8 -*-
"""
labview_diagrams.py
===================

Генерирует схематические «псевдо-скриншоты» того, как должны выглядеть:
  1) фронт-панель (Front Panel) VI;
  2) блок-диаграмма (Block Diagram) VI.

Картинки используются в пояснительной записке как наглядные иллюстрации,
показывающие — куда и как подключить контролы, MathScript Node и XY Graph.

Это НЕ настоящие скриншоты LabVIEW (среды у нас нет в headless-окружении),
а схематические рисунки в стиле LabVIEW. Размещение, подписи и цвета
максимально приближены к тому, что вы увидите в реальной программе.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Палитра, близкая к LabVIEW
COL_PANEL_BG  = "#f3f3f3"
COL_BLOCK_BG  = "#ffffff"
COL_CTRL      = "#fff7c8"           # жёлтый — controls
COL_IND       = "#cfe1ff"           # голубой — indicators
COL_NODE      = "#e6e6e6"           # серый — узел/функция
COL_GRAPH     = "#ffffff"
COL_WIRE      = "#1a1a1a"
COL_WIRE_NUM  = "#1a4a8a"           # синий — численные провода
COL_WIRE_ARR  = "#7e3a9a"           # фиолетовый — массивы


def _box(ax, x, y, w, h, fc, ec="black", lw=1.0, label=None,
         label_color="black", fontsize=10, rounded=False):
    if rounded:
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc, edgecolor=ec, linewidth=lw)
    else:
        box = patches.Rectangle(
            (x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=label_color)


def _arrow(ax, x1, y1, x2, y2, color=COL_WIRE_NUM, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, shrinkA=2, shrinkB=2))


# -----------------------------------------------------------------------------
# 1. Фронт-панель
# -----------------------------------------------------------------------------
def front_panel():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Общий фон панели
    _box(ax, 0.1, 0.1, 10.3, 5.8, COL_PANEL_BG, ec="#888")

    # Заголовок
    ax.text(5.25, 5.6, "VI Front Panel — Complex Euler Experiment",
            ha="center", fontsize=12, weight="bold")

    # === Левая колонка: Controls ===
    ax.text(0.4, 5.2, "Controls (управляющие элементы)", fontsize=10, weight="bold")

    # Numeric control eq_idx (1..5)
    _box(ax, 0.4, 4.50, 1.8, 0.5, COL_CTRL, rounded=True,
         label="eq_idx", fontsize=10)
    ax.text(2.3, 4.75, "← номер уравнения 1..5", fontsize=9, va="center")

    # Slider n_macro
    _box(ax, 0.4, 3.80, 3.5, 0.5, COL_CTRL, rounded=True,
         label="n_macro  ▮▮▮▮▮▯▯▯▯  =  20", fontsize=10)
    ax.text(0.4, 3.65, "(Slider, 2..200)", fontsize=8, color="#555")

    # === Индикаторы погрешности ===
    ax.text(0.4, 3.20, "Indicators (выходные значения)", fontsize=10, weight="bold")
    _box(ax, 0.4, 2.55, 2.2, 0.55, COL_IND, rounded=True,
         label="err_real = 0.231", fontsize=10)
    _box(ax, 0.4, 1.85, 2.2, 0.55, COL_IND, rounded=True,
         label="err_cmpl = 0.027", fontsize=10)
    _box(ax, 0.4, 1.15, 2.2, 0.55, COL_IND, rounded=True,
         label="dt = 0.15", fontsize=10)

    # === Правая часть: XY Graph ===
    _box(ax, 4.30, 0.50, 5.95, 4.60, COL_GRAPH, ec="#444")
    ax.text(7.27, 4.85, "XY Graph — численные решения и точное",
            ha="center", fontsize=10, weight="bold")

    # Сетка
    for i in range(1, 7):
        ax.plot([4.30 + i * 0.85, 4.30 + i * 0.85], [0.55, 4.45],
                color="#dddddd", linewidth=0.5)
    for j in range(1, 5):
        ax.plot([4.35, 10.20], [0.55 + j * 0.78, 0.55 + j * 0.78],
                color="#dddddd", linewidth=0.5)

    # Кривые-«заглушки» для XY Graph
    t = np.linspace(0, 1, 200)
    # точное (зелёное)
    ax.plot(4.45 + t * 5.6, 0.7 + np.exp(2.2 * t) * 0.07 * 4,
            color="#2ca02c", linewidth=2.0)
    # классический (синий ломаный, отстающий)
    tk = np.linspace(0, 1, 6)
    ax.plot(4.45 + tk * 5.6, 0.7 + np.exp(1.9 * tk) * 0.06 * 4,
            "o-", color="#1f77b4", linewidth=1.5, markersize=5)
    # комплексный (чёрный)
    tk2 = np.linspace(0, 1, 4)
    ax.plot(4.45 + tk2 * 5.6, 0.7 + np.exp(2.15 * tk2) * 0.068 * 4,
            "s-", color="#000000", linewidth=1.5, markersize=5)

    # Мини-легенда
    ax.plot([4.45, 4.75], [4.35, 4.35], color="#2ca02c", linewidth=2)
    ax.text(4.80, 4.35, "exact", fontsize=8, va="center")
    ax.plot([5.55, 5.85], [4.35, 4.35], "o-", color="#1f77b4", markersize=4)
    ax.text(5.90, 4.35, "classic Euler", fontsize=8, va="center")
    ax.plot([6.95, 7.25], [4.35, 4.35], "s-", color="#000000", markersize=4)
    ax.text(7.30, 4.35, "complex 2-step", fontsize=8, va="center")

    # Подписи осей
    ax.text(7.27, 0.30, "t", ha="center", fontsize=9)
    ax.text(4.20, 2.80, "y", ha="center", fontsize=9, rotation=90)

    out = os.path.join(PLOTS_DIR, "labview_front_panel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# 2. Блок-диаграмма
# -----------------------------------------------------------------------------
def block_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Фон диаграммы
    _box(ax, 0.1, 0.1, 11.3, 6.3, COL_BLOCK_BG, ec="#888")

    # Заголовок
    ax.text(5.75, 6.1, "VI Block Diagram — соединения MathScript Node",
            ha="center", fontsize=12, weight="bold")

    # === Контролы (входы) — слева ===
    _box(ax, 0.30, 4.60, 1.20, 0.50, COL_CTRL, rounded=True,
         label="eq_idx", fontsize=10)
    _box(ax, 0.30, 3.90, 1.20, 0.50, COL_CTRL, rounded=True,
         label="n_macro", fontsize=10)

    # === Центральный узел: MathScript Node ===
    node_x, node_y, node_w, node_h = 3.20, 1.60, 5.20, 4.20
    _box(ax, node_x, node_y, node_w, node_h, COL_NODE, ec="black", lw=2)
    ax.text(node_x + node_w / 2, node_y + node_h + 0.10,
            "MathScript Node", ha="center", fontsize=11, weight="bold")

    # «Окно» внутри узла с текстом скрипта
    _box(ax, node_x + 0.20, node_y + 0.20, node_w - 0.40, node_h - 0.40,
         "#fefefe", ec="#aaa")
    script_lines = [
        "% Внутри MathScript Node:",
        "switch eq_idx",
        "  case 1; f_rhs=@(t,y) y;       y0=1;  t0=0; tN=3;",
        "  case 2; f_rhs=@(t,y) y.^2;    y0=1;  t0=0; tN=0.5;",
        "  ...",
        "end",
        "",
        "w1 = 0.5 + 0.5i;",
        "w2 = 0.5 - 0.5i;",
        "dt = (tN - t0) / n_macro;",
        "",
        "% --- классический Эйлер ---",
        "for k = 1:N_r",
        "   y_real(k+1) = y_real(k) + dt_r * f_rhs(t_real(k), y_real(k));",
        "end",
        "",
        "% --- комплексный 2-шаговый ---",
        "for k = 1:n_macro",
        "   y_star = yc + w1*dt*f_rhs(tk, yc);",
        "   yc     = real(y_star + w2*dt*f_rhs(tk+w1*dt, y_star));",
        "end",
    ]
    for i, line in enumerate(script_lines):
        ax.text(node_x + 0.30, node_y + node_h - 0.45 - i * 0.18, line,
                fontsize=8, family="monospace")

    # Терминалы входов (на левой кромке узла) — маленькие треугольники-стрелки
    ax.add_patch(patches.Polygon(
        [(node_x, 4.85), (node_x - 0.10, 4.78), (node_x - 0.10, 4.92)],
        closed=True, color="black"))
    ax.text(node_x + 0.05, 4.85, "eq_idx", fontsize=8, va="center", color="#444")
    ax.add_patch(patches.Polygon(
        [(node_x, 4.15), (node_x - 0.10, 4.08), (node_x - 0.10, 4.22)],
        closed=True, color="black"))
    ax.text(node_x + 0.05, 4.15, "n_macro", fontsize=8, va="center", color="#444")

    # === Индикаторы (выходы) — справа ===
    indicators_y = [5.10, 4.55, 4.00, 3.45, 2.90, 2.35, 1.80]
    indicator_names = [
        "t_real (array)",
        "y_real (array)",
        "t_cmpl (array)",
        "y_cmpl (array)",
        "t_exact (array)",
        "y_exact (array)",
        "err_real | err_cmpl (DBL)",
    ]
    colors_arr = [COL_WIRE_ARR] * 6 + [COL_WIRE_NUM]
    for name, y, c in zip(indicator_names, indicators_y, colors_arr):
        _box(ax, 9.20, y - 0.20, 1.95, 0.40, COL_IND, rounded=True,
             label=name, fontsize=8)
        # Терминал на правой кромке MathScript Node
        ax.add_patch(patches.Polygon(
            [(node_x + node_w, y), (node_x + node_w + 0.10, y - 0.07),
             (node_x + node_w + 0.10, y + 0.07)],
            closed=True, color="black"))
        # Провод от терминала к индикатору
        _arrow(ax, node_x + node_w + 0.12, y, 9.18, y, color=c, lw=1.5)

    # Провода контролы → узел
    _arrow(ax, 1.52, 4.85, node_x - 0.12, 4.85, color=COL_WIRE_NUM)
    _arrow(ax, 1.52, 4.15, node_x - 0.12, 4.15, color=COL_WIRE_NUM)

    # Внизу — XY Graph (показано как «принимающий» массивы блок)
    _box(ax, 4.20, 0.40, 3.20, 1.00, COL_IND, ec="black", rounded=True)
    ax.text(5.80, 0.90, "XY Graph", fontsize=10, ha="center", weight="bold")
    ax.text(5.80, 0.60, "(массивы (t_real,y_real),(t_cmpl,y_cmpl),(t_exact,y_exact))",
            fontsize=7, ha="center", color="#555")

    # Стрелки от индикаторов-массивов к XY Graph (символически — через Build Array)
    # Чтобы не перегружать диаграмму, проводов несколько обобщённых
    for y in (4.00, 3.45, 2.90, 2.35):
        ax.annotate("", xy=(5.80, 1.42), xytext=(10.15, y),
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.7))

    # Легенда цветов проводов
    _box(ax, 0.30, 0.40, 3.20, 1.00, "#ffffff", ec="#888")
    ax.text(0.40, 1.20, "Цвета проводов:", fontsize=9, weight="bold")
    ax.plot([0.45, 0.95], [0.95, 0.95], color=COL_WIRE_NUM, linewidth=2)
    ax.text(1.05, 0.95, "числовой (DBL/INT)", fontsize=8, va="center")
    ax.plot([0.45, 0.95], [0.65, 0.65], color=COL_WIRE_ARR, linewidth=2.5)
    ax.text(1.05, 0.65, "массив (1D-array)", fontsize=8, va="center")

    out = os.path.join(PLOTS_DIR, "labview_block_diagram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# 3. Скриншот: настройка терминалов MathScript Node (важно для инструкции)
# -----------------------------------------------------------------------------
def mathscript_terminals():
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    _box(ax, 0.10, 0.10, 7.80, 4.80, "#fafafa", ec="#888")
    ax.text(4.0, 4.6, "Настройка терминалов MathScript Node",
            ha="center", fontsize=12, weight="bold")
    ax.text(4.0, 4.30,
            "ПКМ по границе узла → Add Input / Add Output → задать имя и тип",
            ha="center", fontsize=9, color="#444")

    # Таблица: имя | направление | тип | назначение
    headers = ["Имя",       "Тип",          "Направление", "Назначение"]
    rows = [
        ("eq_idx",        "int32 (I32)",  "Input",  "номер уравнения 1..5"),
        ("n_macro",       "int32 (I32)",  "Input",  "число макрошагов компл. метода"),
        ("t_real",        "1D-Array DBL", "Output", "сетка времени классич. Эйлера"),
        ("y_real",        "1D-Array DBL", "Output", "значения классич. Эйлера"),
        ("t_cmpl",        "1D-Array DBL", "Output", "сетка времени компл. метода"),
        ("y_cmpl",        "1D-Array DBL", "Output", "значения компл. метода"),
        ("t_exact",       "1D-Array DBL", "Output", "сетка точного решения"),
        ("y_exact",       "1D-Array DBL", "Output", "точное решение"),
        ("err_real",      "DBL",          "Output", "‖err‖∞ классич. Эйлера"),
        ("err_cmpl",      "DBL",          "Output", "‖err‖∞ компл. метода"),
    ]

    # Шапка
    cols_x = [0.40, 2.20, 3.80, 5.20]
    y_head = 3.90
    for x, h in zip(cols_x, headers):
        ax.text(x, y_head, h, fontsize=9, weight="bold")
    ax.plot([0.30, 7.70], [y_head - 0.10, y_head - 0.10], color="#666")

    # Строки
    for i, row in enumerate(rows):
        y = y_head - 0.35 - i * 0.28
        for x, val in zip(cols_x, row):
            ax.text(x, y, val, fontsize=8.5)

    out = os.path.join(PLOTS_DIR, "labview_terminals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def main():
    front_panel()
    block_diagram()
    mathscript_terminals()


if __name__ == "__main__":
    main()
