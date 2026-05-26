# -*- coding: utf-8 -*-
"""
controllable_stability_gui.py
=============================

Интерактивный GUI для демонстрации УПРАВЛЯЕМОЙ УСТОЙЧИВОСТИ.

Левая панель — комплексная плоскость с границами устойчивости (классический
Эйлер и 2-шаговый комплексный). Точка z = λ·Δt отмечается красным маркером.
Положение точки определяется ползунками Re(λ), Im(λ) и Δt.

Правая панель — численное решение пробной задачи ẏ = λ·y с тем же λ и Δt
обоими методами. Видно, растёт ли решение, затухает или осциллирует, а
также явное «выгорание» (overflow), если точка λΔt вылетает за границы.

Под графиками выводятся численные значения |Φ_classic(z)| и |Φ_complex(z)|
— коэффициенты усиления одного шага. Если они > 1, метод неустойчив.

Запуск:
    python controllable_stability_gui.py
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np

from complex_euler import classic_euler, complex_two_step_euler
from complex_euler import stability_classic, stability_complex2


# Сетка для построения границ устойчивости
_X = np.linspace(-4.5, 2.5, 600)
_Y = np.linspace(-3.5, 3.5, 600)
_XM, _YM = np.meshgrid(_X, _Y)
_ZM = _XM + 1j * _YM


class ControllableStabilityApp:
    def __init__(self):
        self.fig = plt.figure(figsize=(13.0, 7.0))
        self.fig.suptitle("Управляемая устойчивость:  ẏ = λ·y,  z = λ·Δt",
                          fontsize=13)

        # Левая ось — комплексная плоскость
        self.ax_stab = self.fig.add_axes([0.05, 0.30, 0.42, 0.62])
        self.ax_stab.set_aspect("equal")
        self.ax_stab.set_xlim(-4.5, 2.5)
        self.ax_stab.set_ylim(-3.5, 3.5)
        self.ax_stab.set_xlabel("Re(z)")
        self.ax_stab.set_ylabel("Im(z)")
        self.ax_stab.set_title("Точка z = λ·Δt на фоне областей устойчивости")
        self.ax_stab.grid(True, alpha=0.3)

        # Заливка областей |Φ| ≤ 1
        phi1 = np.abs(stability_classic(_ZM))
        phi2 = np.abs(stability_complex2(_ZM))
        self.ax_stab.contourf(_XM, _YM, phi2, levels=[0, 1],
                              colors=["#8db9f0"], alpha=0.55)
        self.ax_stab.contourf(_XM, _YM, phi1, levels=[0, 1],
                              colors=["#9ad08f"], alpha=0.65)
        self.ax_stab.contour(_XM, _YM, phi1, levels=[1],
                             colors=["#1a7a3a"], linewidths=2)
        self.ax_stab.contour(_XM, _YM, phi2, levels=[1],
                             colors=["#1f3a6e"], linewidths=2)
        self.ax_stab.axhline(0, color="grey", linewidth=0.5)
        self.ax_stab.axvline(0, color="grey", linewidth=0.5)

        # Маркер точки λΔt (динамический)
        self.point_marker, = self.ax_stab.plot([], [], "o",
                                               color="#d62728", markersize=11,
                                               markeredgecolor="black")

        # Правая ось — численное решение пробной задачи
        self.ax_sol = self.fig.add_axes([0.55, 0.30, 0.40, 0.62])
        self.ax_sol.set_xlabel("t")
        self.ax_sol.set_ylabel("y(t)")
        self.ax_sol.set_title("Численное решение ẏ = λ·y, y(0)=1")
        self.ax_sol.grid(True, alpha=0.3)

        # Слайдеры внизу: Re(λ), Im(λ), Δt
        ax_re   = self.fig.add_axes([0.10, 0.18, 0.55, 0.03])
        ax_im   = self.fig.add_axes([0.10, 0.13, 0.55, 0.03])
        ax_dt   = self.fig.add_axes([0.10, 0.08, 0.55, 0.03])
        self.s_re = Slider(ax_re, "Re(λ)", -3.0, 1.0, valinit=-1.0,
                           valstep=0.05)
        self.s_im = Slider(ax_im, "Im(λ)", -2.0, 2.0, valinit=0.0,
                           valstep=0.05)
        self.s_dt = Slider(ax_dt, "Δt",     0.05, 3.0, valinit=1.0,
                           valstep=0.05)
        for s in (self.s_re, self.s_im, self.s_dt):
            s.on_changed(lambda _v: self._redraw())

        # Поле статистики справа от слайдеров
        self.text_ax = self.fig.add_axes([0.70, 0.05, 0.27, 0.20])
        self.text_ax.axis("off")
        self.text_handle = self.text_ax.text(
            0.02, 0.95, "", transform=self.text_ax.transAxes,
            family="monospace", fontsize=10, va="top")

        self._redraw()

    # ----- логика перерисовки ------------------------------------------------
    def _redraw(self):
        lam_re = self.s_re.val
        lam_im = self.s_im.val
        dt     = self.s_dt.val
        lam    = lam_re + 1j * lam_im
        z      = lam * dt

        # 1. Двигаем маркер
        self.point_marker.set_data([z.real], [z.imag])

        # 2. Перерисовываем правую ось — решение
        self.ax_sol.clear()
        self.ax_sol.set_xlabel("t")
        self.ax_sol.set_ylabel("y(t)")
        self.ax_sol.set_title(f"ẏ = ({lam:.2f})·y,  y(0)=1,  Δt={dt:.2f}")
        self.ax_sol.grid(True, alpha=0.3)

        # Возьмём 25 макрошагов, отрезок [0, 25*dt]
        N = 25
        T = N * dt

        f = lambda t, y: lam * y
        # Для линейной задачи у комплексного метода 1-й порядок не теряется;
        # для устойчивого случая решение — exp(λt).
        # Берём complex_two_step_euler без real-проекции — он сам её сделает.
        try:
            t_c, y_c = complex_two_step_euler(f, 1.0, 0.0, T, N)
            # классический даёт 2N шагов того же Δt/2
            t_r, y_r = classic_euler(f, 1.0, 0.0, T, 2 * N)
            t_e = np.linspace(0, T, 200)
            y_e = np.real(np.exp(lam * t_e))   # точное решение
        except (OverflowError, FloatingPointError):
            self.ax_sol.text(0.5, 0.5, "OVERFLOW",
                             transform=self.ax_sol.transAxes,
                             ha="center", fontsize=20, color="red")
            return

        # Маскируем NaN / Inf для красивого построения
        y_c = np.where(np.isfinite(y_c), y_c, np.nan)
        y_r = np.where(np.isfinite(y_r), y_r, np.nan)

        self.ax_sol.plot(t_e, y_e, color="#2ca02c", linewidth=2.0,
                         label="Точное Re(e^{λt})")
        self.ax_sol.plot(t_r, y_r, "o--", color="#1f77b4", markersize=4,
                         label="Классический Эйлер")
        self.ax_sol.plot(t_c, y_c, "s-",  color="#000000", markersize=4,
                         label="Комплексный 2-шаговый")
        self.ax_sol.legend(loc="best", fontsize=9)

        # 3. Статистика — |Φ|, устойчивость
        phi_r = stability_classic(z)
        phi_c = stability_complex2(z)
        mag_r = abs(phi_r)
        mag_c = abs(phi_c)

        def verdict(mag):
            if mag <= 1.0 + 1e-12:
                return "УСТОЙЧИВ"
            return "НЕУСТОЙЧИВ"

        lines = [
            f"z = λ·Δt = {z:.3f}",
            "",
            f"|Φ_classic(z)| = {mag_r:.4f} → {verdict(mag_r)}",
            f"|Φ_complex(z)| = {mag_c:.4f} → {verdict(mag_c)}",
            "",
            "Метод устойчив, если |Φ(z)| ≤ 1.",
            "Точка должна лежать ВНУТРИ",
            "зелёной (классический) или",
            "голубой (комплексный) области.",
        ]
        self.text_handle.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()


def main():
    _ = ControllableStabilityApp()
    plt.show()


if __name__ == "__main__":
    main()
