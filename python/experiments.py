# -*- coding: utf-8 -*-
"""
experiments.py
==============

Повторение экспериментов из раздела 2 статьи George–Jung–Mangan
(arXiv:2110.04402): сравнение классического Эйлера и 2-шагового комплексного
Эйлера на наборе ОДУ из Fig.1 и Fig.2.

Запуск:
    python experiments.py

Создаёт PNG-картинки в каталоге ../plots/ (относительно расположения файла).
Эти изображения затем вставляются в пояснительные записки.

Главная цель — наглядно показать, что при ОДИНАКОВОМ количестве вычислений
правой части f(t,y) комплексный 2-шаговый Эйлер имеет 2-й порядок и заметно
ближе к точному решению.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")          # отключаем интерактивный backend для batch-режима
import matplotlib.pyplot as plt
import numpy as np

from python.complex_euler import classic_euler, complex_two_step_euler
from python.equations import EQUATIONS


# Каталог для PNG-выходов
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Общие настройки matplotlib (читаемые подписи)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "legend.framealpha": 0.9,
})


# -----------------------------------------------------------------------------
# Утилита: запустить ОБА метода с одинаковым числом вычислений f(t,y).
#
# Комплексный 2-шаговый метод за один «макрошаг» Δt делает 2 обращения к f.
# Чтобы сравнение было «честным» по числу вычислений правой части,
# классическому Эйлеру даём в 2 раза больше шагов размера Δt/2.
# -----------------------------------------------------------------------------
def run_pair(eq, n_macro: int):
    """Возвращает три набора (t, y): classic, complex, exact."""
    # Комплексный метод: n_macro «макрошагов», по 2 обращения к f каждый =
    # = 2·n_macro вычислений f.
    t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n_macro)
    # Классический Эйлер с 2·n_macro шагами для честного сравнения.
    t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n_macro)
    # Точное решение на густой сетке (для линии «exact»).
    t_e = np.linspace(eq.t0, eq.tN, 400)
    y_e = eq.exact(t_e) if eq.exact is not None else None
    return (t_r, y_r), (t_c, y_c), (t_e, y_e)


# -----------------------------------------------------------------------------
# 1. Главная картинка — повторение Fig.1 (линейное ОДУ ẏ = y)
# -----------------------------------------------------------------------------
def figure_fig1():
    eq = EQUATIONS["linear"]
    (t_r, y_r), (t_c, y_c), (t_e, y_e) = run_pair(eq, n_macro=3)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t_e, y_e, label="Точное решение y=eᵗ",
            linewidth=2.5, color="#2ca02c")
    ax.plot(t_r, y_r, "o-", label="Классический Эйлер (6 шагов)",
            color="#1f77b4", markersize=6)
    ax.plot(t_c, y_c, "s-", label="Комплексный 2-шаговый Эйлер (3 макрошага)",
            color="#000000", markersize=6)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("Повторение Fig.1 статьи: ẏ = y, y(0)=1")
    ax.legend(loc="upper left")
    out = os.path.join(PLOTS_DIR, "fig1_linear.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# 2. Сетка панелей — повторение Fig.2 (нелинейные / неавтономные)
# -----------------------------------------------------------------------------
def figure_fig2_grid():
    panels = [
        ("square", "ẏ = y²",                10),
        ("nlsin",  "ẏ = 4y·sin³(t)·cos(t)", 20),
        ("cos",    "ẏ = cos(t)",            20),
        ("airy",   "ÿ = t·y",               40),
        ("shm",    "ÿ = −y",                40),
        ("vdp",    "ÿ = μ(1−y²)·ẏ − y",     80),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    axes = axes.ravel()

    for ax, (key, title, n_macro) in zip(axes, panels):
        eq = EQUATIONS[key]
        # Для систем берём первую компоненту (саму y).
        (t_r, y_r), (t_c, y_c), (t_e, y_e) = run_pair(eq, n_macro=n_macro)
        if y_r.ndim == 2:
            y_r = y_r[:, 0]
        if y_c.ndim == 2:
            y_c = y_c[:, 0]
        if y_e is not None and np.ndim(y_e) == 2:
            y_e = y_e[:, 0]

        # Особый случай Ван-дер-Поля: «эталоном» считаем решение комплексным
        # методом с очень мелким шагом (n_macro × 10).
        if key in ("airy", "vdp"):
            t_fine, y_fine = complex_two_step_euler(
                eq.f, eq.y0, eq.t0, eq.tN, n_macro * 20)
            if y_fine.ndim == 2:
                y_fine = y_fine[:, 0]
            ax.plot(t_fine, y_fine, color="#2ca02c", linewidth=2.0,
                    label="«Эталон» (мелкий шаг)")
        elif y_e is not None:
            ax.plot(t_e, y_e, color="#2ca02c", linewidth=2.0,
                    label="Точное решение")

        ax.plot(t_r, y_r, "--", color="#1f77b4", linewidth=1.2,
                label="Классический Эйлер")
        ax.plot(t_c, y_c, "-", color="#000000", linewidth=1.6,
                label="Комплексный 2-шаговый")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel("y")

    # Один общий легенд-блок на всю фигуру
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Повторение Fig.2 статьи: классический vs комплексный 2-шаговый Эйлер",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig2_grid.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# 3. Иллюстрация ПУТИ в комплексной плоскости
# -----------------------------------------------------------------------------
def figure_complex_path():
    dt = 0.5
    n = 6
    # Классический путь — на действительной оси.
    t_real = np.array([(k * dt, 0.0) for k in range(n + 1)])

    # Комплексный путь — зигзаг через w1·dt = dt/2 + i·dt/2.
    pts = [(0.0, 0.0)]
    for k in range(n):
        x_prev, y_prev = pts[-1]
        # шаг w1
        pts.append((x_prev + dt / 2, y_prev + dt / 2))
        # шаг w2 — возвращается на действительную ось
        x_prev, y_prev = pts[-1]
        pts.append((x_prev + dt / 2, y_prev - dt / 2))
    pts = np.array(pts)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(t_real[:, 0], t_real[:, 1], "o-", color="#1f77b4",
            label=f"Классический Эйлер, шаг Δt={dt}")
    ax.plot(pts[:, 0], pts[:, 1], "s-", color="#000000",
            label=f"Комплексный 2-шаговый: w₁·Δt и w₂·Δt")
    # Стрелки-подписи
    ax.annotate("Δt/2 + iΔt/2", xy=(dt / 4, dt / 4 + 0.05),
                xytext=(dt / 4 - 0.4, dt / 2 + 0.25),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("Δt/2 − iΔt/2", xy=(dt / 2 + dt / 4, dt / 4 + 0.05),
                xytext=(dt / 2 + dt / 4 + 0.05, dt / 2 + 0.25),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("Re t")
    ax.set_ylabel("Im t")
    ax.set_title("Пути интегрирования в комплексной плоскости (Δt = 0.5)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    out = os.path.join(PLOTS_DIR, "complex_path.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("Запуск экспериментов из раздела 2 статьи...")
    figure_complex_path()
    figure_fig1()
    figure_fig2_grid()
    print("Готово. Графики в:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
