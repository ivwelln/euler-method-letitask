# -*- coding: utf-8 -*-
"""
gui.py
======

Интерактивное окно matplotlib с ПОЛЗУНКАМИ для управляемого эксперимента
(аналог LabVIEW-VI: можно «крутить» Δt и переключать уравнение,
сразу видеть изменения).

Управление:
  • RadioButtons слева — выбор уравнения (linear, square, nlsin, cos, shm).
  • Slider «n_macro»   — число макрошагов комплексного 2-шагового метода
                         (классическому даётся 2·n_macro шагов).
  • Кнопка «Refresh»   — пересчитать и перерисовать.

Запуск:
    python gui.py
"""

from __future__ import annotations

import matplotlib
# Здесь нам нужен интерактивный backend; пусть matplotlib выберет сам.
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import numpy as np

from complex_euler import classic_euler, complex_two_step_euler, infinity_norm_error
from equations import EQUATIONS


# Только уравнения, для которых есть точное решение
EQ_KEYS = ["linear", "square", "nlsin", "cos", "shm"]


class InteractiveExperiment:
    def __init__(self):
        # Главное окно
        self.fig = plt.figure(figsize=(10.5, 6.5))
        self.fig.suptitle("Комплексный 2-шаговый Эйлер vs Классический Эйлер",
                          fontsize=12)

        # Основной график
        self.ax = self.fig.add_axes([0.30, 0.20, 0.66, 0.70])
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.3)

        # Радио-кнопки выбора уравнения
        rax = self.fig.add_axes([0.02, 0.50, 0.20, 0.40])
        rax.set_title("Уравнение", fontsize=10)
        self.radio = RadioButtons(rax, EQ_KEYS, active=0)
        self.radio.on_clicked(self._on_equation)

        # Слайдер числа макрошагов
        sax = self.fig.add_axes([0.30, 0.07, 0.55, 0.04])
        self.slider = Slider(sax, "n_macro", 2, 200, valinit=10, valstep=1)
        self.slider.on_changed(self._on_slider)

        # Кнопка Refresh
        bax = self.fig.add_axes([0.88, 0.07, 0.08, 0.04])
        self.button = Button(bax, "Refresh")
        self.button.on_clicked(self._on_refresh)

        # Поле со статистикой
        self.text_ax = self.fig.add_axes([0.02, 0.05, 0.22, 0.40])
        self.text_ax.axis("off")
        self.text_handle = self.text_ax.text(
            0.02, 0.95, "", transform=self.text_ax.transAxes,
            family="monospace", fontsize=9, va="top")

        self.current_key = EQ_KEYS[0]
        self._redraw()

    # --- обработчики ---------------------------------------------------------
    def _on_equation(self, key):
        self.current_key = key
        self._redraw()

    def _on_slider(self, _val):
        self._redraw()

    def _on_refresh(self, _evt):
        self._redraw()

    # --- основной перерасчёт -------------------------------------------------
    def _redraw(self):
        eq = EQUATIONS[self.current_key]
        n = int(self.slider.val)

        t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
        t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n)
        t_e = np.linspace(eq.t0, eq.tN, 400)
        y_e = eq.exact(t_e) if eq.exact is not None else None

        # для систем берём первую компоненту
        if y_c.ndim == 2:
            y_c = y_c[:, 0]
        if y_r.ndim == 2:
            y_r = y_r[:, 0]
        if y_e is not None and np.ndim(y_e) == 2:
            y_e = y_e[:, 0]

        # === перерисовка графика ===
        self.ax.clear()
        if y_e is not None:
            self.ax.plot(t_e, y_e, color="#2ca02c",
                         linewidth=2.0, label="Точное решение")
        self.ax.plot(t_r, y_r, "o--", color="#1f77b4",
                     markersize=4, label=f"Классич. Эйлер ({2*n} шагов)")
        self.ax.plot(t_c, y_c, "s-", color="#000000",
                     markersize=4, label=f"Компл. 2-шаговый ({n} макрошагов)")
        self.ax.set_title(eq.title, fontsize=11)
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best", fontsize=9)

        # === статистика погрешности ===
        info_lines = []
        if eq.exact is not None:
            y_e_c = eq.exact(t_c)
            y_e_r = eq.exact(t_r)
            if y_e_c.ndim == 2:
                y_e_c = y_e_c[:, 0]
            if y_e_r.ndim == 2:
                y_e_r = y_e_r[:, 0]
            err_r = infinity_norm_error(y_r, y_e_r)
            err_c = infinity_norm_error(y_c, y_e_c)
            dt = (eq.tN - eq.t0) / n
            info_lines.append(f"Δt (макро) = {dt:.4g}")
            info_lines.append(f"")
            info_lines.append(f"‖err‖_∞:")
            info_lines.append(f"  классический : {err_r:.3e}")
            info_lines.append(f"  комплексный  : {err_c:.3e}")
            info_lines.append(f"")
            if err_c > 0:
                info_lines.append(f"улучшение ×{err_r/err_c:7.2f}")
        else:
            info_lines.append("(точное решение неизв.)")
        self.text_handle.set_text("\n".join(info_lines))

        self.fig.canvas.draw_idle()


def main():
    _ = InteractiveExperiment()
    plt.show()


if __name__ == "__main__":
    main()
