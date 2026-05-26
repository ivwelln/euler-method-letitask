# -*- coding: utf-8 -*-
"""
snapshots.py
============

Делает СТАТИЧЕСКИЕ снимки-сценарии, которые повторяют типичные кадры
интерактивных GUI. Эти PNG нужны в отчёте как иллюстрации.

Сценарии:

A. Управляемая устойчивость (controllable_stability):
   A1. λ=−0.8, Δt=1.0  — точка внутри обеих областей: оба метода устойчивы.
   A2. λ=−1.5, Δt=1.5  — точка ЗА единичным кругом классического Эйлера,
                         но всё ещё внутри овала комплексного метода.
                         Классический «взрывается», комплексный устойчив.
   A3. λ=−2.5, Δt=1.0  — точка ВЫСКОЧИЛА из обеих областей: оба
                         расходятся, но с разной скоростью.

B. Изменение комплексного коэффициента (varying_coefficient):
   B1. w₁=0.5+0.5i  — канонический выбор, 2-й порядок.
   B2. w₁=1.0 (вещественный)  — это просто Эйлер ½+½ = два шага Δt/2,
                                1-й порядок (потерян 2-й).
   B3. w₁=0.3+0.7i  — w₁+w₂=1 выполнено, но w₁·w₂=0.51 — почти 2-й порядок,
                       немного изменена область устойчивости.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from complex_euler import (classic_euler, complex_two_step_euler,
                           stability_classic, stability_complex2)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Сетка для границ устойчивости
_X = np.linspace(-4.5, 2.5, 600)
_Y = np.linspace(-3.5, 3.5, 600)
_XM, _YM = np.meshgrid(_X, _Y)
_ZM = _XM + 1j * _YM
_PHI_CLASSIC = np.abs(stability_classic(_ZM))


# -----------------------------------------------------------------------------
# Сценарий A: управляемая устойчивость
# -----------------------------------------------------------------------------
def _snapshot_A(lam: complex, dt: float, label: str, fname: str):
    """Один кадр: λΔt на областях + численное решение y(t) = e^{λt}."""
    fig = plt.figure(figsize=(12.5, 5.0))

    # Левая панель — комплексная плоскость
    ax_stab = fig.add_axes([0.05, 0.13, 0.40, 0.78])
    ax_stab.set_aspect("equal")
    phi_c = np.abs(stability_complex2(_ZM))
    ax_stab.contourf(_XM, _YM, phi_c, levels=[0, 1],
                     colors=["#8db9f0"], alpha=0.55)
    ax_stab.contourf(_XM, _YM, _PHI_CLASSIC, levels=[0, 1],
                     colors=["#9ad08f"], alpha=0.65)
    ax_stab.contour(_XM, _YM, _PHI_CLASSIC, levels=[1],
                    colors=["#1a7a3a"], linewidths=2)
    ax_stab.contour(_XM, _YM, phi_c, levels=[1],
                    colors=["#1f3a6e"], linewidths=2)
    ax_stab.axhline(0, color="grey", linewidth=0.5)
    ax_stab.axvline(0, color="grey", linewidth=0.5)
    z = lam * dt
    ax_stab.plot(z.real, z.imag, "o", color="#d62728", markersize=12,
                 markeredgecolor="black")
    ax_stab.annotate(f"z = λ·Δt = {z:.2f}", (z.real, z.imag),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax_stab.set_xlim(-4.5, 2.5)
    ax_stab.set_ylim(-3.5, 3.5)
    ax_stab.set_xlabel("Re(z)"); ax_stab.set_ylabel("Im(z)")
    ax_stab.set_title("Положение λ·Δt на областях устойчивости")
    ax_stab.legend(handles=[
        mpatches.Patch(color="#9ad08f", label="классический Эйлер: |1+z|≤1"),
        mpatches.Patch(color="#8db9f0", label="комплексный 2-шаг.: |1+z+z²/2|≤1"),
    ], loc="upper left", fontsize=8)

    # Правая панель — численное решение
    ax_sol = fig.add_axes([0.52, 0.13, 0.43, 0.78])
    N = 25
    T = N * dt
    f = lambda t, y: lam * y

    with np.errstate(over='ignore', invalid='ignore'):
        try:
            t_c, y_c = complex_two_step_euler(f, 1.0, 0.0, T, N)
            t_r, y_r = classic_euler(f, 1.0, 0.0, T, 2 * N)
        except (OverflowError, FloatingPointError):
            y_c = y_r = None
    t_e = np.linspace(0, T, 200)
    y_e = np.real(np.exp(lam * t_e))

    ax_sol.plot(t_e, y_e, color="#2ca02c", linewidth=2.2,
                label=r"точное Re(e^{λt})")
    if y_r is not None:
        y_r_plot = np.where(np.isfinite(y_r), y_r, np.nan)
        ax_sol.plot(t_r, y_r_plot, "o--", color="#1f77b4",
                    markersize=4, label="классический Эйлер")
    if y_c is not None:
        y_c_plot = np.where(np.isfinite(y_c), y_c, np.nan)
        ax_sol.plot(t_c, y_c_plot, "s-", color="#000000",
                    markersize=4, label="комплексный 2-шаг.")
    # подписи устойчивости
    mag_r = abs(stability_classic(z))
    mag_c = abs(stability_complex2(z))
    v_r = "устойчив" if mag_r <= 1.0 + 1e-12 else "НЕУСТОЙЧИВ"
    v_c = "устойчив" if mag_c <= 1.0 + 1e-12 else "НЕУСТОЙЧИВ"
    ax_sol.set_title(f"{label}\n"
                     f"|Φ_classic|={mag_r:.3f} ({v_r}),  "
                     f"|Φ_complex|={mag_c:.3f} ({v_c})",
                     fontsize=10)
    ax_sol.set_xlabel("t"); ax_sol.set_ylabel("y(t)")
    ax_sol.grid(True, alpha=0.3)
    ax_sol.legend(loc="best", fontsize=9)
    # автомасштаб: вертикально не более ±1000, чтобы overflow не убил картинку
    ymax = np.nanpercentile(np.abs(np.concatenate(
        [y_e,
         y_r_plot if y_r is not None else np.array([1.0]),
         y_c_plot if y_c is not None else np.array([1.0])])), 99)
    if not np.isfinite(ymax) or ymax > 1000:
        ymax = 1000
    ax_sol.set_ylim(-2 * ymax - 1, 2 * ymax + 1)

    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def figures_A_controllable_stability():
    _snapshot_A(lam=-0.8 + 0.0j, dt=1.0,
                label="Сценарий A1: λ=−0.8, Δt=1.0 — оба метода устойчивы",
                fname="cs_A1_inside.png")
    _snapshot_A(lam=-0.5 + 1.5j, dt=1.0,
                label="Сценарий A2: λ=−0.5+1.5i, Δt=1.0 — классический НЕустойчив,\nкомплексный 2-шаговый — ЕЩЁ устойчив",
                fname="cs_A2_classic_unstable.png")
    _snapshot_A(lam=-2.5 + 0.0j, dt=1.0,
                label="Сценарий A3: λ=−2.5, Δt=1.0 — оба метода неустойчивы",
                fname="cs_A3_both_unstable.png")


# -----------------------------------------------------------------------------
# Сценарий B: изменение комплексного коэффициента
# -----------------------------------------------------------------------------
def _snapshot_B(w1: complex, label: str, fname: str):
    """Один кадр: путь, область устойчивости, решение ẏ=y."""
    w2 = 1.0 - w1
    dt = 0.5

    fig = plt.figure(figsize=(14.0, 4.5))

    # Левая ось — путь
    ax_path = fig.add_axes([0.04, 0.15, 0.27, 0.70])
    ax_path.set_aspect("equal")
    pts = [(0.0, 0.0)]
    for k in range(4):
        x_prev, y_prev = pts[-1]
        pts.append((x_prev + (w1 * dt).real, y_prev + (w1 * dt).imag))
        x_prev, y_prev = pts[-1]
        pts.append((x_prev + (w2 * dt).real, y_prev + (w2 * dt).imag))
    pts = np.array(pts)
    ax_path.plot(pts[:, 0], pts[:, 1], "o-", color="#000000")
    ax_path.axhline(0, color="grey", linewidth=0.5)
    ax_path.set_xlabel("Re t"); ax_path.set_ylabel("Im t")
    ax_path.grid(True, alpha=0.3)
    ax_path.set_title(f"Путь:  w₁={w1:.2f},  w₂={w2:.2f}")
    ylim = max(0.6, abs((w1 * dt).imag) * 1.6)
    ax_path.set_ylim(-ylim, ylim)
    ax_path.set_xlim(-0.3, 4 * dt + 0.3)

    # Центральная ось — область устойчивости
    ax_stab = fig.add_axes([0.36, 0.15, 0.27, 0.70])
    ax_stab.set_aspect("equal")
    phi = np.abs(stability_complex2(_ZM, w1, w2))
    ax_stab.contourf(_XM, _YM, phi, levels=[0, 1],
                     colors=["#8db9f0"], alpha=0.6)
    ax_stab.contour(_XM, _YM, phi, levels=[1],
                    colors=["#1f3a6e"], linewidths=2)
    # для сравнения — круг классического Эйлера
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_stab.plot(-1 + np.cos(theta), np.sin(theta),
                 color="#1a7a3a", linewidth=1.5,
                 label="классич. Эйлер")
    ax_stab.axhline(0, color="grey", linewidth=0.5)
    ax_stab.axvline(0, color="grey", linewidth=0.5)
    ax_stab.set_xlim(-4.0, 1.5)
    ax_stab.set_ylim(-2.5, 2.5)
    ax_stab.set_xlabel("Re(z)"); ax_stab.set_ylabel("Im(z)")
    ax_stab.set_title(f"Область устойчивости при w₁={w1:.2f}")
    ax_stab.legend(loc="upper left", fontsize=8)

    # Правая ось — численное решение ẏ=y
    ax_sol = fig.add_axes([0.68, 0.15, 0.29, 0.70])
    N = 12
    T = N * dt
    f = lambda t, y: y
    t_c, y_c = complex_two_step_euler(f, 1.0, 0.0, T, N, w1=w1, w2=w2)
    t_r, y_r = classic_euler(f, 1.0, 0.0, T, 2 * N)
    t_e = np.linspace(0, T, 200)
    y_e = np.exp(t_e)
    ax_sol.plot(t_e, y_e, color="#2ca02c", linewidth=2.0, label="точное eᵗ")
    ax_sol.plot(t_r, y_r, "o--", color="#1f77b4", markersize=4,
                label="классический")
    ax_sol.plot(t_c, y_c, "s-", color="#000000", markersize=4,
                label=f"2-шаг. (w₁={w1:.2f})")
    ax_sol.set_xlabel("t"); ax_sol.set_ylabel("y")
    ax_sol.grid(True, alpha=0.3)
    err_c = np.max(np.abs(y_c - np.exp(t_c)))
    err_r = np.max(np.abs(y_r - np.exp(t_r)))
    prod_check = abs(w1 * w2 - 0.5)
    order_hint = "2-й порядок" if prod_check < 1e-3 else "1-й порядок"
    ax_sol.set_title(
        f"ẏ=y, y(0)=1\n‖err‖∞: класс.={err_r:.2e}, 2-ш.={err_c:.2e}\n"
        f"|w₁w₂−½|={prod_check:.2e} ⇒ {order_hint}",
        fontsize=9)
    ax_sol.legend(loc="best", fontsize=8)

    fig.suptitle(label, fontsize=12)

    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def figures_B_varying_coefficient():
    _snapshot_B(w1=0.5 + 0.5j,
                label="Сценарий B1: w₁ = 0.5 + 0.5i — КАНОНИЧЕСКИЙ выбор, 2-й порядок",
                fname="vc_B1_canonical.png")
    _snapshot_B(w1=1.0 + 0.0j,
                label="Сценарий B2: w₁ = 1.0 (вещественный) — путь вырождается, остаётся только 1-й порядок",
                fname="vc_B2_real.png")
    _snapshot_B(w1=0.3 + 0.7j,
                label="Сценарий B3: w₁ = 0.3 + 0.7i — другой комплексный путь",
                fname="vc_B3_other.png")


# -----------------------------------------------------------------------------
# Сценарий C: график порядка как функции от Im(w₁) при фиксированном Re(w₁)=0.5
# -----------------------------------------------------------------------------
def figure_C_order_vs_imag():
    """Эмпирическая оценка порядка как функция Im(w₁)."""
    re_w1 = 0.5
    im_values = np.linspace(0.0, 1.0, 21)
    orders = []
    eq_f = lambda t, y: y
    n_list = [32, 64, 128, 256]

    for im in im_values:
        w1 = re_w1 + 1j * im
        w2 = 1.0 - w1
        dts, errs = [], []
        for n in n_list:
            _, y_c = complex_two_step_euler(eq_f, 1.0, 0.0, 1.0, n, w1=w1, w2=w2)
            dts.append(1.0 / n)
            errs.append(abs(y_c[-1] - np.exp(1.0)))
        # log-log регрессия
        p = np.polyfit(np.log(dts), np.log(np.array(errs) + 1e-16), 1)[0]
        orders.append(p)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(im_values, orders, "o-", color="#000000", markersize=6)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="1-й порядок")
    ax.axhline(2.0, color="#1f3a6e", linestyle="--", linewidth=1,
               label="2-й порядок (теоретическая граница)")
    ax.axvline(0.5, color="#d62728", linestyle=":", linewidth=1.2,
               label="канонический Im(w₁)=0.5")
    ax.set_xlabel("Im(w₁)   (при Re(w₁)=0.5, w₂=1−w₁)")
    ax.set_ylabel("Эмпирический порядок p")
    ax.set_title("Зависимость порядка точности от мнимой части коэффициента w₁\n"
                 "Только при w₁ = 0.5 ± 0.5i достигается 2-й порядок")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out = os.path.join(PLOTS_DIR, "vc_order_vs_imag.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
def main():
    print("Сценарии A: управляемая устойчивость...")
    figures_A_controllable_stability()
    print("Сценарии B: переменный коэффициент...")
    figures_B_varying_coefficient()
    print("Сценарий C: график порядка...")
    figure_C_order_vs_imag()


if __name__ == "__main__":
    main()
