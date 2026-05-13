# -*- coding: utf-8 -*-
"""
convergence.py
==============

Оценка ТОЧНОСТИ (порядка сходимости) методов.

Для каждого тестового ОДУ с известным точным решением считаем максимальную
по сетке погрешность ‖y_num − y_exact‖_∞ при разных шагах Δt и строим
log-log график. Угол наклона прямой даёт порядок сходимости:

  • Классический Эйлер     ⇒ наклон ≈ 1
  • Комплексный 2-шаговый  ⇒ наклон ≈ 2

Запуск:
    python convergence.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python.complex_euler import classic_euler, complex_two_step_euler, infinity_norm_error
from python.equations import EQUATIONS


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Уравнения с известным точным решением для оценки порядка
KEYS_WITH_EXACT = ["linear", "square", "nlsin", "cos", "shm"]


def compute_errors(eq, n_macro_list):
    """Для каждого n_macro считаем оба метода и возвращаем dt-ы и погрешности."""
    dts, err_real, err_cmpl = [], [], []
    for n in n_macro_list:
        # Комплексный — n макрошагов → 2n вычислений f
        t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
        # Классический — 2n шагов для честного сравнения по числу f
        t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n)

        y_e_c = eq.exact(t_c)
        y_e_r = eq.exact(t_r)

        # Для систем (shm) берём только y (первую компоненту).
        if y_c.ndim == 2:
            y_c, y_e_c = y_c[:, 0], y_e_c[:, 0]
        if y_r.ndim == 2:
            y_r, y_e_r = y_r[:, 0], y_e_r[:, 0]

        dts.append((eq.tN - eq.t0) / n)         # «макрошаг» Δt
        err_real.append(infinity_norm_error(y_r, y_e_r))
        err_cmpl.append(infinity_norm_error(y_c, y_e_c))
    return np.array(dts), np.array(err_real), np.array(err_cmpl)


def figure_convergence():
    n_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(KEYS_WITH_EXACT)))

    for color, key in zip(colors, KEYS_WITH_EXACT):
        eq = EQUATIONS[key]
        dts, err_r, err_c = compute_errors(eq, n_list)
        ax.loglog(dts, err_r, "o--", color=color,
                  label=f"{eq.name}: классич.")
        ax.loglog(dts, err_c, "s-",  color=color,
                  label=f"{eq.name}: компл. 2-ш.")

    # Опорные прямые угла наклона 1 и 2
    dt_ref = np.array([1e-3, 1e0])
    ax.loglog(dt_ref, 0.3 * dt_ref ** 1, "k:",  label="наклон 1", linewidth=1)
    ax.loglog(dt_ref, 0.3 * dt_ref ** 2, "k-.", label="наклон 2", linewidth=1)

    ax.set_xlabel("Δt (макрошаг)")
    ax.set_ylabel("‖y_num − y_exact‖_∞")
    ax.set_title("Сходимость: классический Эйлер vs комплексный 2-шаговый")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.4)
    out = os.path.join(PLOTS_DIR, "convergence.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def print_orders():
    """Печатает численные оценки порядка для отчёта."""
    print()
    print(f"{'Уравнение':12s} | {'метод':12s} | оценка порядка p")
    print("-" * 50)
    n_list = [32, 64, 128, 256, 512]
    for key in KEYS_WITH_EXACT:
        eq = EQUATIONS[key]
        dts, err_r, err_c = compute_errors(eq, n_list)
        # log-log регрессия по последним 4 точкам
        p_r = np.polyfit(np.log(dts[-4:]), np.log(err_r[-4:]), 1)[0]
        p_c = np.polyfit(np.log(dts[-4:]), np.log(err_c[-4:]), 1)[0]
        print(f"{eq.name:12s} | {'classic':12s} | {p_r:5.3f}")
        print(f"{eq.name:12s} | {'complex 2-st':12s} | {p_c:5.3f}")


def main():
    print("Оценка порядка точности методов...")
    figure_convergence()
    print_orders()


if __name__ == "__main__":
    main()
