# -*- coding: utf-8 -*-
"""
verify.py
=========

Финальная проверка корректности реализации:

1. Линейный случай ẏ=y проверяется по точной формуле раздела 2 статьи:
       y_real(Δt)  = 1 + Δt + Δt²/4   (классический Эйлер с 2 шагами Δt/2)
       y_compl(Δt) = 1 + Δt + Δt²/2   (комплексный 2-шаговый Эйлер)
   После одного «макрошага» Δt значения должны совпасть с этими формулами
   до машинной точности.

2. Условие 2-го порядка на веса:  w1 + w2 = 1  и  w1·w2 = 1/2 .

3. Эмпирический порядок сходимости (log-log регрессия по 5 точкам) для
   каждого уравнения с известным точным решением — должен быть близок
   к 1 (классический) и к 2 (комплексный).

4. Функции устойчивости должны давать ровно 1 на собственной границе
   (для классического: z = -2;  для комплексного 2-шагового: z = -2 + 0i
   тоже даёт |Φ| = 1, но проверим в другой точке).
"""

from __future__ import annotations

import sys
import numpy as np

from python.complex_euler import (
    classic_euler, complex_two_step_euler,
    stability_function_classic, stability_function_complex2,
    W1, W2,
)
from python.equations import EQUATIONS


N_FAIL = 0


def check(name, cond, msg=""):
    global N_FAIL
    if cond:
        print(f"  [OK]  {name}")
    else:
        print(f"  [FAIL] {name}  {msg}")
        N_FAIL += 1


def test_linear_one_step_formula():
    print("\n=== 1. Сверка с формулами раздела 2 на ẏ=y ===")
    dt = 0.5

    # один макрошаг классического Эйлера с двумя подшагами Δt/2:
    # y1 = 1*(1 + dt/2)*(1 + dt/2)
    y_th_real = 1 * (1 + dt/2) * (1 + dt/2)
    # один макрошаг комплексного:
    # y1 = 1*(1 + (dt/2 + i*dt/2))(1 + (dt/2 - i*dt/2))
    y_th_compl = 1 * (1 + W1*dt) * (1 + W2*dt)

    # численно: один макрошаг = шаг длины dt;
    # для классического это 2 шага Δt/2, для комплексного — 1 макрошаг
    _, yc = complex_two_step_euler(lambda t, y: y, 1.0, 0.0, dt, 1)
    _, yr = classic_euler(lambda t, y: y, 1.0, 0.0, dt, 2)

    check("classic numerically == теоретическая формула",
          abs(yr[-1] - y_th_real) < 1e-12,
          f"num={yr[-1]}, theory={y_th_real}")
    check("complex numerically == теоретическая формула",
          abs(yc[-1] - y_th_compl.real) < 1e-12,
          f"num={yc[-1]}, theory={y_th_compl.real}")
    check("формула: y_real(dt)  = 1 + dt + dt²/4",
          abs(y_th_real  - (1 + dt + dt**2/4)) < 1e-12)
    check("формула: y_compl(dt) = 1 + dt + dt²/2",
          abs(y_th_compl - (1 + dt + dt**2/2)) < 1e-12)


def test_weights():
    print("\n=== 2. Условия 2-го порядка на w₁, w₂ ===")
    check("w1 + w2 == 1",   abs(W1 + W2 - 1)   < 1e-15)
    check("w1 * w2 == 1/2", abs(W1 * W2 - 0.5) < 1e-15)


def test_orders():
    print("\n=== 3. Эмпирические порядки сходимости ===")
    n_list = [32, 64, 128, 256, 512]
    keys = ["linear", "square", "nlsin", "cos", "shm"]
    for key in keys:
        eq = EQUATIONS[key]
        dts, err_r, err_c = [], [], []
        for n in n_list:
            t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
            t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n)
            y_e_c = eq.exact(t_c)
            y_e_r = eq.exact(t_r)
            if y_c.ndim == 2:
                y_c, y_e_c = y_c[:, 0], y_e_c[:, 0]
            if y_r.ndim == 2:
                y_r, y_e_r = y_r[:, 0], y_e_r[:, 0]
            dts.append((eq.tN - eq.t0) / n)
            err_r.append(np.max(np.abs(y_r - y_e_r)))
            err_c.append(np.max(np.abs(y_c - y_e_c)))
        p_r = np.polyfit(np.log(dts), np.log(err_r), 1)[0]
        p_c = np.polyfit(np.log(dts), np.log(err_c), 1)[0]
        check(f"[{eq.name:25s}] classic ≈ 1   (полученное {p_r:.3f})",
              0.85 < p_r < 1.40)
        check(f"[{eq.name:25s}] complex ≈ 2   (полученное {p_c:.3f})",
              1.85 < p_c < 2.15)


def test_stability_functions():
    print("\n=== 4. Функции устойчивости в характерных точках ===")
    # На границе областей |Φ|=1
    check("classic |Φ(-2)|     == 1",
          abs(abs(stability_function_classic(-2.0)) - 1.0) < 1e-12)
    # Комплексный 2-шаговый на z=-2: Φ=1+(-2)+2 = 1, |Φ|=1
    check("complex2 |Φ(-2)|    == 1",
          abs(abs(stability_function_complex2(-2.0)) - 1.0) < 1e-12)
    # Точка в начале координат — устойчивы оба
    check("classic |Φ(0)|      == 1",
          abs(abs(stability_function_classic(0.0)) - 1.0) < 1e-12)
    check("complex2 |Φ(0)|     == 1",
          abs(abs(stability_function_complex2(0.0)) - 1.0) < 1e-12)


def main():
    print("Финальная проверка реализации")
    print("=" * 60)
    test_linear_one_step_formula()
    test_weights()
    test_orders()
    test_stability_functions()
    print("\n" + "=" * 60)
    if N_FAIL == 0:
        print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✓")
        sys.exit(0)
    else:
        print(f"ОШИБОК: {N_FAIL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
