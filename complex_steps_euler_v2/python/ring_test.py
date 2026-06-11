# -*- coding: utf-8 -*-
"""
ring_test.py
============

КОЛЬЦЕВОЙ ТЕСТ (round-trip): идея проверки качества численного метода
через «интегрирование туда и обратно».

Схема эксперимента:
  1. Стартуем из y₀ в момент t = 0;
  2. Интегрируем выбранным методом ВПЕРЁД до момента t = T, получаем y_T;
  3. Из y_T интегрируем ТЕМ ЖЕ методом НАЗАД с отрицательным шагом −Δt
     до момента t = 0, получаем y_back;
  4. Сравниваем |y_back − y₀| — это «невязка кольца».

Что это даёт:
  • У идеального (точного) метода невязка равна 0 — мы прошли по замкнутому
    контуру в фазовом пространстве и вернулись в исходную точку.
  • У реальных методов невязка ≠ 0 и имеет порядок O(Δt^p), где p — порядок
    метода. Поэтому в log-log координатах прямая с наклоном p.
  • Это очень наглядный тест: его легко формулировать на пальцах
    («поедешь — приедешь? вернёшься — приедешь в ту же точку?»), а
    численно он выявляет именно ОБЩУЮ накопленную ошибку метода.

Здесь мы:
  • строим невязки кольца для классического и комплексного методов;
  • показываем log-log график их зависимости от Δt;
  • визуализируем «петлю» решения в плоскости (t, y) для каждого метода —
    видно, замыкается она или нет.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complex_euler import (
    classic_euler, classic_euler_reverse,
    complex_two_step_euler, complex_two_step_euler_reverse,
)
from equations import EQUATIONS


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Один прогон round-trip и возврат невязки
# -----------------------------------------------------------------------------
def ring_residual(method_forward, method_backward, eq, n: int):
    """Считает |y(0) − y_back(0)| для заданного метода и сетки из n шагов."""
    # Forward
    t_f, y_f = method_forward(eq.f, eq.y0, eq.t0, eq.tN, n)
    y_T = y_f[-1] if np.ndim(y_f) == 1 else y_f[-1]
    # Backward: стартуем из конечного значения, шагаем назад
    t_b, y_b = method_backward(eq.f, y_T, eq.t0, eq.tN, n)
    y_0_back = y_b[-1]
    # Невязка кольца
    if np.ndim(y_0_back) == 0:
        return float(abs(y_0_back - eq.y0))
    # для систем — берём норму вектора
    return float(np.max(np.abs(np.asarray(y_0_back) - np.asarray(eq.y0))))


# -----------------------------------------------------------------------------
# Главный график: log-log невязки кольца от Δt
# -----------------------------------------------------------------------------
def figure_ring_convergence():
    """Невязка кольца от макрошага Δt в log-log координатах."""
    keys = ["linear", "square", "nlsin", "cos", "shm"]
    n_list = [8, 16, 32, 64, 128, 256, 512]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for color, key in zip(colors, keys):
        eq = EQUATIONS[key]
        dts, res_r, res_c = [], [], []
        for n in n_list:
            # Классическому даём 2n, чтобы по числу вызовов f было то же,
            # что у комплексного с n макрошагами.
            r_r = ring_residual(classic_euler, classic_euler_reverse, eq, 2 * n)
            r_c = ring_residual(complex_two_step_euler,
                                complex_two_step_euler_reverse, eq, n)
            dts.append((eq.tN - eq.t0) / n)
            res_r.append(r_r if r_r > 0 else 1e-16)
            res_c.append(r_c if r_c > 0 else 1e-16)

        ax.loglog(dts, res_r, "o--", color=color,
                  label=f"{eq.name}: классич.")
        ax.loglog(dts, res_c, "s-",  color=color,
                  label=f"{eq.name}: компл. 2-ш.")

    # Опорные прямые
    dt_ref = np.array([1e-3, 1e0])
    ax.loglog(dt_ref, 0.5 * dt_ref ** 1, "k:",
              label="наклон 1", linewidth=1)
    ax.loglog(dt_ref, 0.5 * dt_ref ** 2, "k-.",
              label="наклон 2", linewidth=1)

    ax.set_xlabel("Δt (макрошаг)")
    ax.set_ylabel("Невязка кольца |y_back(0) − y₀|")
    ax.set_title("Кольцевой тест: «невязка кольца» в log-log координатах\n"
                 "(классический — пунктир, комплексный 2-шаговый — сплошная)")
    ax.legend(bbox_to_anchor=(0.5, -0.15), fontsize=8, loc="upper center", ncol=2)
    ax.grid(True, which="both", alpha=0.4)

    out = os.path.join(PLOTS_DIR, "ring_test_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# Визуализация петли решения в плоскости (t, y) для одного уравнения
# -----------------------------------------------------------------------------
def figure_ring_path(key: str = "linear", n_macro: int = 5):
    """Для выбранного уравнения рисует прямой и обратный путь —
    видно, замыкается петля или нет."""
    eq = EQUATIONS[key]
    # Классический: вперёд → назад
    t_rf, y_rf = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n_macro)
    y_T_r = y_rf[-1] if y_rf.ndim == 1 else y_rf[-1]
    t_rb, y_rb = classic_euler_reverse(eq.f, y_T_r, eq.t0, eq.tN, 2 * n_macro)

    # Комплексный
    t_cf, y_cf = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n_macro)
    y_T_c = y_cf[-1] if y_cf.ndim == 1 else y_cf[-1]
    t_cb, y_cb = complex_two_step_euler_reverse(eq.f, y_T_c, eq.t0, eq.tN, n_macro)

    # Для системы берём первую компоненту
    if y_rf.ndim == 2:
        y_rf, y_rb = y_rf[:, 0], y_rb[:, 0]
    if y_cf.ndim == 2:
        y_cf, y_cb = y_cf[:, 0], y_cb[:, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Левая панель — классический
    ax1.plot(t_rf, y_rf, "o-", color="#1f77b4", label=f"вперёд (2·n={2*n_macro})")
    ax1.plot(t_rb, y_rb, "s--", color="#d62728", label="обратно")
    # Подсветка начальной точки и куда вернулись
    ax1.scatter([eq.t0], [eq.y0 if np.ndim(eq.y0)==0 else eq.y0[0]],
                color="black", zorder=5, s=70, label="старт y₀")
    ax1.scatter([eq.t0], [y_rb[-1]], color="orange", zorder=5, s=70,
                label=f"y_back ({abs(y_rb[-1]-(eq.y0 if np.ndim(eq.y0)==0 else eq.y0[0])):.2e})")
    ax1.set_title(f"Классический Эйлер: round-trip для {eq.name}")
    ax1.set_xlabel("t"); ax1.set_ylabel("y")
    ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

    # Правая панель — комплексный
    ax2.plot(t_cf, y_cf, "o-", color="#1f77b4", label=f"вперёд (n={n_macro})")
    ax2.plot(t_cb, y_cb, "s--", color="#d62728", label="обратно")
    ax2.scatter([eq.t0], [eq.y0 if np.ndim(eq.y0)==0 else eq.y0[0]],
                color="black", zorder=5, s=70, label="старт y₀")
    ax2.scatter([eq.t0], [y_cb[-1]], color="orange", zorder=5, s=70,
                label=f"y_back ({abs(y_cb[-1]-(eq.y0 if np.ndim(eq.y0)==0 else eq.y0[0])):.2e})")
    ax2.set_title(f"Комплексный 2-шаговый Эйлер: round-trip для {eq.name}")
    ax2.set_xlabel("t"); ax2.set_ylabel("y")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

    fig.suptitle("Кольцевой тест: вперёд → обратно. Видно, насколько метод «возвращает» к y₀",
                 fontsize=12)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, f"ring_test_path_{key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# Таблица улучшения по уравнениям
# -----------------------------------------------------------------------------
def print_summary_table():
    """Печатает таблицу: невязка классического / невязка комплексного при n=64."""
    n = 64
    print("\n=== Сводка кольцевого теста (n=64) ===")
    print(f"{'Уравнение':25s} | {'классич.':>12s} | {'компл.':>12s} | {'улучшение':>10s}")
    print("-" * 70)
    for key in ["linear", "square", "nlsin", "cos", "shm"]:
        eq = EQUATIONS[key]
        r_r = ring_residual(classic_euler, classic_euler_reverse, eq, 2 * n)
        r_c = ring_residual(complex_two_step_euler,
                            complex_two_step_euler_reverse, eq, n)
        impr = r_r / r_c if r_c > 0 else float('inf')
        print(f"{eq.name:25s} | {r_r:12.4e} | {r_c:12.4e} | {impr:8.2f}×")


def main():
    print("Запуск кольцевого теста...")
    figure_ring_convergence()
    figure_ring_path("linear", n_macro=4)
    figure_ring_path("shm", n_macro=20)
    print_summary_table()


if __name__ == "__main__":
    main()
