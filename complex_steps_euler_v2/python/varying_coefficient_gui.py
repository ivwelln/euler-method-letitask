# -*- coding: utf-8 -*-
"""
varying_coefficient_gui.py
==========================

Интерактивный GUI для исследования ВЛИЯНИЯ КОМПЛЕКСНОГО КОЭФФИЦИЕНТА w₁
на 2-шаговый метод. Пользователь меняет Re(w₁) и Im(w₁) ползунками;
w₂ = 1 − w₁ берётся автоматически (так выполняется условие 1-го порядка
w₁ + w₂ = 1, и метод как минимум 1-го порядка).

Условие 2-го порядка:  w₁ · w₂ = 1/2.  Подставляя w₂ = 1 − w₁, получаем
квадратное уравнение w₁(1 − w₁) = 1/2, т. е. w₁² − w₁ + 1/2 = 0.
Его корни:  w₁ = 1/2 ± i/2.  Любой ДРУГОЙ выбор w₁ даёт лишь 1-й порядок.

Три панели:
  • Слева — путь в комплексной плоскости (w₁·Δt, потом w₂·Δt = (1−w₁)·Δt).
  • Посередине — область устойчивости Φ(z) = (1 + w₁ z)(1 + w₂ z).
  • Справа — численное решение ẏ = y и сравнение с точным e^t.

Под графиками — текущие значения w₁, w₂, |w₁·w₂ − 0.5| (мера отступления
от 2-го порядка) и эмпирическая оценка погрешности.

Запуск:
    python varying_coefficient_gui.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from complex_euler import (classic_euler, complex_two_step_euler,
                           stability_complex2, infinity_norm_error)

# Сетка для области устойчивости
_X = np.linspace(-4.0, 2.0, 500)
_Y = np.linspace(-3.0, 3.0, 500)
_XM, _YM = np.meshgrid(_X, _Y)
_ZM = _XM + 1j * _YM


class VaryingCoefficientApp:
    def __init__(self):
        self.fig = plt.figure(figsize=(14.0, 7.5))
        self.fig.suptitle("Влияние комплексного коэффициента w₁ на метод "
                          "(w₂ = 1 − w₁)", fontsize=13)

        # Три оси: путь, область устойчивости, решение
        self.ax_path  = self.fig.add_axes([0.04, 0.30, 0.27, 0.60])
        self.ax_stab  = self.fig.add_axes([0.36, 0.30, 0.27, 0.60])
        self.ax_sol   = self.fig.add_axes([0.68, 0.30, 0.29, 0.60])

        # Слайдеры
        ax_re = self.fig.add_axes([0.10, 0.15, 0.55, 0.03])
        ax_im = self.fig.add_axes([0.10, 0.10, 0.55, 0.03])
        ax_dt = self.fig.add_axes([0.10, 0.05, 0.55, 0.03])
        self.s_re = Slider(ax_re, "Re(w₁)", 0.0, 1.0, valinit=0.5, valstep=0.01)
        self.s_im = Slider(ax_im, "Im(w₁)", -1.0, 1.0, valinit=0.5, valstep=0.01)
        self.s_dt = Slider(ax_dt, "Δt",     0.05, 1.5, valinit=0.5, valstep=0.05)
        for s in (self.s_re, self.s_im, self.s_dt):
            s.on_changed(lambda _v: self._redraw())

        # Кнопка «Канонические» — устанавливает w₁ = 0.5 + 0.5i
        ax_btn = self.fig.add_axes([0.70, 0.05, 0.10, 0.04])
        self.btn_canon = Button(ax_btn, "канон. 2-й")
        self.btn_canon.on_clicked(self._set_canonical)

        # Поле статистики
        self.text_ax = self.fig.add_axes([0.82, 0.02, 0.16, 0.20])
        self.text_ax.axis("off")
        self.text_handle = self.text_ax.text(
            0.02, 0.95, "", transform=self.text_ax.transAxes,
            family="monospace", fontsize=10, va="top")

        self._redraw()

    def _set_canonical(self, _evt):
        self.s_re.set_val(0.5)
        self.s_im.set_val(0.5)
        # _redraw сработает через on_changed

    def _redraw(self):
        w1 = self.s_re.val + 1j * self.s_im.val
        w2 = 1.0 - w1
        dt = self.s_dt.val

        # --- Левая ось: путь интегрирования в комплексной плоскости ---
        self.ax_path.clear()
        self.ax_path.set_aspect("equal")
        self.ax_path.set_title("Путь в комплексной плоскости\n"
                               f"w₁={w1:.2f},  w₂={w2:.2f}")
        self.ax_path.set_xlabel("Re t");  self.ax_path.set_ylabel("Im t")
        self.ax_path.grid(True, alpha=0.3)
        # Несколько макрошагов подряд
        pts = [(0.0, 0.0)]
        for k in range(5):
            x_prev, y_prev = pts[-1]
            pts.append((x_prev + (w1 * dt).real, y_prev + (w1 * dt).imag))
            x_prev, y_prev = pts[-1]
            pts.append((x_prev + (w2 * dt).real, y_prev + (w2 * dt).imag))
        pts = np.array(pts)
        self.ax_path.plot(pts[:, 0], pts[:, 1], "o-", color="#000000")
        self.ax_path.axhline(0, color="grey", linewidth=0.5)
        # Подсвечиваем вещественную ось
        self.ax_path.set_xlim(-0.5, 5 * dt + 0.5)
        ylim = max(0.5, abs((w1 * dt).imag) * 1.5)
        self.ax_path.set_ylim(-ylim, ylim)

        # --- Центральная ось: область устойчивости Φ(z) = (1+w1 z)(1+w2 z) ---
        self.ax_stab.clear()
        self.ax_stab.set_aspect("equal")
        self.ax_stab.set_title("Область устойчивости |Φ(z)|≤1\n"
                               f"Φ(z) = (1+w₁z)(1+w₂z)")
        self.ax_stab.set_xlabel("Re(z)");  self.ax_stab.set_ylabel("Im(z)")
        self.ax_stab.set_xlim(-4.0, 2.0)
        self.ax_stab.set_ylim(-3.0, 3.0)
        self.ax_stab.grid(True, alpha=0.3)
        try:
            phi = np.abs(stability_complex2(_ZM, w1, w2))
            self.ax_stab.contourf(_XM, _YM, phi, levels=[0, 1],
                                  colors=["#8db9f0"], alpha=0.6)
            self.ax_stab.contour(_XM, _YM, phi, levels=[1],
                                 colors=["#1f3a6e"], linewidths=2)
        except Exception:
            pass
        # для сравнения — граница классического Эйлера
        theta = np.linspace(0, 2 * np.pi, 200)
        self.ax_stab.plot(-1 + np.cos(theta), np.sin(theta),
                          color="#1a7a3a", linewidth=1.5,
                          label="классический Эйлер")
        self.ax_stab.axhline(0, color="grey", linewidth=0.5)
        self.ax_stab.axvline(0, color="grey", linewidth=0.5)
        self.ax_stab.legend(loc="upper left", fontsize=8)

        # --- Правая ось: численное решение ẏ = y ---
        self.ax_sol.clear()
        self.ax_sol.set_title("Численное решение ẏ = y,  y(0)=1")
        self.ax_sol.set_xlabel("t");  self.ax_sol.set_ylabel("y")
        self.ax_sol.grid(True, alpha=0.3)
        N = 12
        T = N * dt
        f = lambda t, y: y
        t_c, y_c = complex_two_step_euler(f, 1.0, 0.0, T, N, w1=w1, w2=w2)
        t_r, y_r = classic_euler(f, 1.0, 0.0, T, 2 * N)
        t_e = np.linspace(0, T, 200)
        y_e = np.exp(t_e)
        self.ax_sol.plot(t_e, y_e, color="#2ca02c", linewidth=2.0,
                         label="точное eᵗ")
        self.ax_sol.plot(t_r, y_r, "o--", color="#1f77b4", markersize=4,
                         label="классический Эйлер")
        self.ax_sol.plot(t_c, y_c, "s-",  color="#000000", markersize=4,
                         label=f"2-шаг. с w₁={w1:.2f}")
        self.ax_sol.legend(loc="best", fontsize=8)

        # --- Статистика ---
        sum_check = abs(w1 + w2 - 1)               # ≈ 0 всегда (по построению)
        prod_check = abs(w1 * w2 - 0.5)            # должно быть ≈ 0 для 2-го порядка
        err_c = infinity_norm_error(y_c, np.exp(t_c))
        err_r = infinity_norm_error(y_r, np.exp(t_r))
        order_hint = "2-й порядок" if prod_check < 1e-3 else \
                     ("≈1-й порядок" if sum_check < 1e-3 else "не Эйлер")
        lines = [
            f"w₁ = {w1:.3f}",
            f"w₂ = {w2:.3f}",
            f"|w₁+w₂−1| = {sum_check:.1e}",
            f"|w₁w₂−½| = {prod_check:.1e}",
            f"⇒ {order_hint}",
            "",
            "Погрешность на ẏ=y:",
            f"  классич.: {err_r:.2e}",
            f"  2-шаг.  : {err_c:.2e}",
        ]
        self.text_handle.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()


def main():
    _ = VaryingCoefficientApp()
    plt.show()


if __name__ == "__main__":
    main()
