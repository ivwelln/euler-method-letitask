# -*- coding: utf-8 -*-
"""
stability_regions.py
====================

Построение и сравнение ОБЛАСТЕЙ АБСОЛЮТНОЙ УСТОЙЧИВОСТИ для:
  • классического Эйлера: Φ(z) = 1 + z
  • комплексного 2-шагового Эйлера: Φ(z) = 1 + z + z²/2
  • (для справки) 3-шагового комплексного Эйлера: Φ(z) = 1 + z + z²/2 + z³/6

Идея:
  Применим метод к модельной задаче ẏ = λy. На каждом макрошаге Δt численное
  решение умножается на «коэффициент усиления» Φ(λΔt). Метод считается
  «абсолютно устойчивым» в точке z = λΔt, если |Φ(z)| ≤ 1 — иначе ошибки от
  предыдущих шагов растут от шага к шагу и решение «разносит».

Эта тема — раздел 6 статьи: «walking into the complex plane to get a larger
stability region». Здесь мы воспроизводим базовые (не оптимизированные)
области, описанные в начале раздела 6 (Fig.11 статьи).

Запуск:
    python stability_regions.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python.complex_euler import (
    stability_function_classic,
    stability_function_complex2,
    stability_function_complex3,
)


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Сетка точек на комплексной плоскости z = x + i·y
# -----------------------------------------------------------------------------
def _meshgrid(xlim=(-4.0, 1.5), ylim=(-3.0, 3.0), n=600):
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z


# -----------------------------------------------------------------------------
# Основной график — сравнение трёх областей устойчивости
# -----------------------------------------------------------------------------
def figure_stability_regions():
    X, Y, Z = _meshgrid()

    phi1 = np.abs(stability_function_classic(Z))
    phi2 = np.abs(stability_function_complex2(Z))
    phi3 = np.abs(stability_function_complex3(Z))

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Заливка областей |Φ(z)| ≤ 1 разной прозрачностью
    ax.contourf(X, Y, phi3, levels=[0, 1], colors=["#fcd28b"], alpha=0.55)
    ax.contourf(X, Y, phi2, levels=[0, 1], colors=["#8db9f0"], alpha=0.55)
    ax.contourf(X, Y, phi1, levels=[0, 1], colors=["#9ad08f"], alpha=0.65)

    # Контурные границы (более тёмные)
    ax.contour(X, Y, phi1, levels=[1], colors=["#1a7a3a"], linewidths=2)
    ax.contour(X, Y, phi2, levels=[1], colors=["#1f3a6e"], linewidths=2)
    ax.contour(X, Y, phi3, levels=[1], colors=["#a05a00"], linewidths=2)

    # Оси
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    # Легенда — через прокси-объекты, чтобы цвета совпадали с заливкой
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#9ad08f", label="Классический Эйлер: |1+z|≤1"),
        mpatches.Patch(color="#8db9f0", label="Комплексный 2-шаг.: |1+z+z²/2|≤1"),
        mpatches.Patch(color="#fcd28b", label="3-шаговый (для справки): |1+z+z²/2+z³/6|≤1"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    ax.set_xlim(-4.0, 1.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z),   z = λ·Δt")
    ax.set_ylabel("Im(z)")
    ax.set_title("Области абсолютной устойчивости методов\n(точка z=λΔt должна быть внутри)")

    out = os.path.join(PLOTS_DIR, "stability_regions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


# -----------------------------------------------------------------------------
# Демонстрация на ẏ = λy с заданным λ — попадает ли λΔt в область устойчивости?
# Полезна для пояснительной записки: можно показать, какой Δt ещё «спасает»
# классический Эйлер, а какой уже нет.
# -----------------------------------------------------------------------------
def figure_stability_demo(lam: complex = -2.0 + 0.0j):
    """Демонстрация: для разных Δt отмечаем положение λΔt на плоскости."""
    X, Y, Z = _meshgrid(xlim=(-4.5, 1.0), ylim=(-2.0, 2.0))
    phi1 = np.abs(stability_function_classic(Z))
    phi2 = np.abs(stability_function_complex2(Z))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.contour(X, Y, phi1, levels=[1], colors=["#1a7a3a"], linewidths=2)
    ax.contour(X, Y, phi2, levels=[1], colors=["#1f3a6e"], linewidths=2)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    # Проба нескольких шагов Δt
    dts = [0.5, 1.0, 1.5, 2.0, 2.5]
    for dt in dts:
        z = lam * dt
        ax.plot(z.real, z.imag, "o", color="#d62728", markersize=7)
        ax.annotate(f"Δt={dt}", (z.real, z.imag),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)

    ax.set_xlim(-4.5, 1.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"ẏ = λy, λ = {lam}: где находится λΔt относительно границ устойчивости")
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#1a7a3a", label="граница классич. Эйлера |1+z|=1"),
        mpatches.Patch(color="#1f3a6e", label="граница комплексн. 2-шагового |1+z+z²/2|=1"),
    ], loc="upper left")
    out = os.path.join(PLOTS_DIR, "stability_demo.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] сохранено: {out}")


def main():
    print("Построение областей устойчивости...")
    figure_stability_regions()
    figure_stability_demo(lam=-2.0)


if __name__ == "__main__":
    main()
