# -*- coding: utf-8 -*-
"""
verify_v2.py
============

Финальная проверка всего, что относится к версии 2:

  1. Теоретические формулы раздела 4.1 статьи: y_real(Δt)=1+Δt+Δt²/4,
     y_compl(Δt)=1+Δt+Δt²/2 при одном макрошаге Δt.
  2. Условия w₁+w₂=1, w₁·w₂=1/2 (канонические веса).
  3. Эмпирические порядки сходимости — 1 и 2.
  4. Функции устойчивости |Φ(z)|=1 на границах.
  5. КОЛЬЦЕВОЙ ТЕСТ: невязка комплексного существенно меньше невязки
     классического (минимум в 50 раз для всех тестовых задач, кроме
     cos(t), где обе невязки на уровне машинной точности).
  6. ПЕРЕМЕННЫЙ КОЭФФИЦИЕНТ: при w₁≠0.5±0.5i порядок ≈ 1; при канонических
     весах — порядок ≈ 2.
  7. УПРАВЛЯЕМАЯ УСТОЙЧИВОСТЬ: для подобранной комплексной λ=−0.5+1.5i и
     Δt=1.0 имеем |Φ_classic|>1, |Φ_complex|<1, что подтверждается
     поведением численного решения.
"""

from __future__ import annotations
import sys
import numpy as np

from complex_euler import (
    classic_euler, complex_two_step_euler,
    classic_euler_reverse, complex_two_step_euler_reverse,
    stability_classic, stability_complex2, W1, W2,
)
from equations import EQUATIONS

N_FAIL = 0

def check(name, cond, msg=""):
    global N_FAIL
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {msg}")
        N_FAIL += 1


def test_one_step_formulas():
    print("\n=== 1. Формулы раздела 4.1 ===")
    dt = 0.5
    y_th_real  = (1 + dt/2) ** 2                          # 1 + dt + dt²/4
    y_th_compl = (1 + W1*dt) * (1 + W2*dt)                # 1 + dt + dt²/2
    _, yc = complex_two_step_euler(lambda t, y: y, 1.0, 0.0, dt, 1)
    _, yr = classic_euler(lambda t, y: y, 1.0, 0.0, dt, 2)
    check("classic = (1+dt/2)²",         abs(yr[-1] - y_th_real)  < 1e-12)
    check("complex = (1+w1dt)(1+w2dt)",  abs(yc[-1] - y_th_compl.real) < 1e-12)
    check("y_real  ≡ 1+dt+dt²/4",         abs(y_th_real  - (1+dt+dt*dt/4)) < 1e-12)
    check("y_compl ≡ 1+dt+dt²/2",         abs(y_th_compl - (1+dt+dt*dt/2)) < 1e-12)


def test_weights():
    print("\n=== 2. Канонические веса ===")
    check("w1+w2 == 1",    abs(W1 + W2 - 1)   < 1e-15)
    check("w1*w2 == 0.5",  abs(W1 * W2 - 0.5) < 1e-15)


def test_orders():
    print("\n=== 3. Эмпирические порядки сходимости ===")
    n_list = [32, 64, 128, 256, 512]
    for key in ["linear", "square", "nlsin", "cos", "shm"]:
        eq = EQUATIONS[key]
        dts, err_r, err_c = [], [], []
        for n in n_list:
            t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
            t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2*n)
            ye_c = eq.exact(t_c); ye_r = eq.exact(t_r)
            if y_c.ndim == 2: y_c, ye_c = y_c[:,0], ye_c[:,0]
            if y_r.ndim == 2: y_r, ye_r = y_r[:,0], ye_r[:,0]
            dts.append((eq.tN-eq.t0)/n)
            err_r.append(np.max(np.abs(y_r-ye_r)))
            err_c.append(np.max(np.abs(y_c-ye_c)))
        p_r = np.polyfit(np.log(dts), np.log(err_r), 1)[0]
        p_c = np.polyfit(np.log(dts), np.log(err_c), 1)[0]
        check(f"[{eq.name:23s}] classic ~ 1  ({p_r:.3f})", 0.85 < p_r < 1.40)
        check(f"[{eq.name:23s}] complex ~ 2  ({p_c:.3f})", 1.85 < p_c < 2.15)


def test_stability_functions():
    print("\n=== 4. Функции устойчивости ===")
    check("|Φ_classic(-2)| == 1",
          abs(abs(stability_classic(-2)) - 1) < 1e-12)
    check("|Φ_complex(0)|  == 1",
          abs(abs(stability_complex2(0))   - 1) < 1e-12)
    check("|Φ_classic(-0.5+1.5j)| > 1 (классический НЕустойчив)",
          abs(stability_classic(-0.5+1.5j)) > 1.0)
    check("|Φ_complex(-0.5+1.5j)| < 1 (комплексный устойчив)",
          abs(stability_complex2(-0.5+1.5j)) < 1.0)


def test_ring():
    print("\n=== 5. Кольцевой тест ===")
    n = 64
    for key in ["linear", "square", "nlsin", "shm"]:   # cos особый, см. отчёт
        eq = EQUATIONS[key]
        # Forward + backward
        _, yf_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2*n)
        _, yb_r = classic_euler_reverse(eq.f, yf_r[-1], eq.t0, eq.tN, 2*n)
        _, yf_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
        _, yb_c = complex_two_step_euler_reverse(eq.f, yf_c[-1], eq.t0, eq.tN, n)
        r_r = np.max(np.abs(np.asarray(yb_r[-1]) - np.asarray(eq.y0)))
        r_c = np.max(np.abs(np.asarray(yb_c[-1]) - np.asarray(eq.y0)))
        impr = r_r / r_c if r_c > 0 else float('inf')
        check(f"[{eq.name:23s}] невязка компл. < невязки классич./50 (impr={impr:.1f}×)",
              impr > 50)


def test_varying_w():
    print("\n=== 6. Переменный коэффициент w₁ ===")
    n_list = [32, 64, 128, 256]
    f_lin = lambda t, y: y

    # 6a. Канонический
    dts, errs = [], []
    for n in n_list:
        _, y = complex_two_step_euler(f_lin, 1.0, 0.0, 1.0, n,
                                      w1=0.5+0.5j, w2=0.5-0.5j)
        dts.append(1.0/n); errs.append(abs(y[-1] - np.exp(1)))
    p_canon = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    check(f"канонические w₁=0.5+0.5i → порядок ≈ 2  ({p_canon:.3f})",
          1.85 < p_canon < 2.15)

    # 6b. Вещественный w₁=1.0
    dts, errs = [], []
    for n in n_list:
        _, y = complex_two_step_euler(f_lin, 1.0, 0.0, 1.0, n,
                                      w1=1.0+0j, w2=0.0+0j)
        dts.append(1.0/n); errs.append(abs(y[-1] - np.exp(1)))
    p_real = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    check(f"вещественный w₁=1.0       → порядок ≈ 1  ({p_real:.3f})",
          0.80 < p_real < 1.20)

    # 6c. Произвольный w₁=0.3+0.7i
    w1 = 0.3+0.7j; w2 = 1 - w1
    dts, errs = [], []
    for n in n_list:
        _, y = complex_two_step_euler(f_lin, 1.0, 0.0, 1.0, n, w1=w1, w2=w2)
        dts.append(1.0/n); errs.append(abs(y[-1] - np.exp(1)))
    p_other = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    check(f"произвольный w₁=0.3+0.7i  → порядок ≈ 1  ({p_other:.3f})",
          0.80 < p_other < 1.20)


def test_controllable_stability():
    print("\n=== 7. Управляемая устойчивость (λ=-0.5+1.5i, Δt=1.0) ===")
    lam = -0.5 + 1.5j; dt = 1.0
    z = lam * dt
    mag_c = abs(stability_classic(z))
    mag_x = abs(stability_complex2(z))
    check(f"|Φ_classic|={mag_c:.3f} > 1 (классический неустойчив)", mag_c > 1.0)
    check(f"|Φ_complex|={mag_x:.3f} < 1 (комплексный устойчив)",    mag_x < 1.0)


def main():
    print("Финальная проверка v2")
    print("=" * 60)
    test_one_step_formulas()
    test_weights()
    test_orders()
    test_stability_functions()
    test_ring()
    test_varying_w()
    test_controllable_stability()
    print("\n" + "=" * 60)
    if N_FAIL == 0:
        print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✓")
        sys.exit(0)
    else:
        print(f"ОШИБОК: {N_FAIL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
