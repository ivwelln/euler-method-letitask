# -*- coding: utf-8 -*-
"""
convergence_split.py
====================

Делает «облегчённые» графики сходимости:
два subplot'а – отдельно классический Эйлер и отдельно комплексный
2-шаговый. На каждом subplot'е лишь 5 кривых + опорная прямая –
это намного читаемее одного «жирного» графика с 10 линиями.

То же самое для кольцевого теста.
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complex_euler import (classic_euler, complex_two_step_euler,
                           classic_euler_reverse, complex_two_step_euler_reverse,
                           infinity_norm_error)
from equations import EQUATIONS

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


KEYS = ["linear", "square", "nlsin", "cos", "shm"]
N_LIST = [8, 16, 32, 64, 128, 256, 512, 1024]


# ---------------------------------------------------------------------
# 1. Сходимость (split)
# ---------------------------------------------------------------------
def _compute_pairs(eq, n_list):
    """Возвращает dts, err_classic, err_complex для одного уравнения."""
    dts, err_r, err_c = [], [], []
    for n in n_list:
        t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
        t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n)
        ye_c = eq.exact(t_c); ye_r = eq.exact(t_r)
        if y_c.ndim == 2: y_c, ye_c = y_c[:, 0], ye_c[:, 0]
        if y_r.ndim == 2: y_r, ye_r = y_r[:, 0], ye_r[:, 0]
        dts.append((eq.tN - eq.t0) / n)
        err_r.append(infinity_norm_error(y_r, ye_r))
        err_c.append(infinity_norm_error(y_c, ye_c))
    return np.array(dts), np.array(err_r), np.array(err_c)


def figure_convergence_split():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    ax_r, ax_c = axes
    colors = plt.cm.tab10(np.linspace(0, 1, len(KEYS)))

    for color, key in zip(colors, KEYS):
        eq = EQUATIONS[key]
        dts, err_r, err_c = _compute_pairs(eq, N_LIST)
        ax_r.loglog(dts, err_r, "o-", color=color, label=eq.name, markersize=5)
        ax_c.loglog(dts, err_c, "s-", color=color, label=eq.name, markersize=5)

    # Опорные прямые
    dt_ref = np.array([1e-3, 1e0])
    ax_r.loglog(dt_ref, 0.3 * dt_ref ** 1, "k--",
                label="наклон 1 (теория)", linewidth=1.4)
    ax_c.loglog(dt_ref, 0.3 * dt_ref ** 2, "k--",
                label="наклон 2 (теория)", linewidth=1.4)

    for ax, title in zip(axes,
                         ["Классический явный Эйлер (1-й порядок)",
                          "Комплексный 2-шаговый Эйлер (2-й порядок)"]):
        ax.set_xlabel("Δt (макрошаг)")
        ax.set_ylabel("‖y_num − y_exact‖_∞")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Сходимость в log-log координатах: разделение по методам",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "convergence_split.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# ---------------------------------------------------------------------
# 2. Кольцевой тест — раздельно по методам
# ---------------------------------------------------------------------
def _ring_residual(method_forward, method_backward, eq, n):
    _, y_f = method_forward(eq.f, eq.y0, eq.t0, eq.tN, n)
    y_T = y_f[-1]
    _, y_b = method_backward(eq.f, y_T, eq.t0, eq.tN, n)
    if np.ndim(y_b[-1]) == 0:
        return float(abs(y_b[-1] - eq.y0))
    return float(np.max(np.abs(np.asarray(y_b[-1]) - np.asarray(eq.y0))))


def figure_ring_test_split():
    n_list = [8, 16, 32, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    ax_r, ax_c = axes
    colors = plt.cm.tab10(np.linspace(0, 1, len(KEYS)))

    for color, key in zip(colors, KEYS):
        eq = EQUATIONS[key]
        dts, res_r, res_c = [], [], []
        for n in n_list:
            r_r = _ring_residual(classic_euler, classic_euler_reverse, eq, 2 * n)
            r_c = _ring_residual(complex_two_step_euler,
                                 complex_two_step_euler_reverse, eq, n)
            dts.append((eq.tN - eq.t0) / n)
            res_r.append(r_r if r_r > 0 else 1e-16)
            res_c.append(r_c if r_c > 0 else 1e-16)
        ax_r.loglog(dts, res_r, "o-", color=color, label=eq.name, markersize=5)
        ax_c.loglog(dts, res_c, "s-", color=color, label=eq.name, markersize=5)

    dt_ref = np.array([1e-3, 1e0])
    ax_r.loglog(dt_ref, 0.5 * dt_ref ** 1, "k--",
                label="наклон 1 (теория)", linewidth=1.4)
    ax_c.loglog(dt_ref, 0.5 * dt_ref ** 2, "k--",
                label="наклон 2 (теория)", linewidth=1.4)

    for ax, title in zip(axes,
                         ["Классический явный Эйлер (1-й порядок)",
                          "Комплексный 2-шаговый Эйлер (2-й порядок)"]):
        ax.set_xlabel("Δt (макрошаг)")
        ax.set_ylabel("Невязка кольца |y_back(0) − y₀|")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Кольцевой тест: невязка по методам отдельно",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "ring_test_convergence_split.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def main():
    figure_convergence_split()
    figure_ring_test_split()


if __name__ == "__main__":
    main()
