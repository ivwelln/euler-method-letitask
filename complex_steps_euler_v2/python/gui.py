# -*- coding: utf-8 -*-
"""
gui.py
======

Главный интерактивный GUI второй версии работы. Объединяет три режима
исследования через выпадающее меню (RadioButtons слева):

  1. «Сравнение методов» — выбор уравнения (ẏ=y, ẏ=y², …), слайдер n_macro,
     показывает кривые точное / классический / комплексный + норма погрешности.
  2. «Управляемая устойчивость» — слайдеры Re(λ), Im(λ), Δt; точка λΔt на
     фоне областей устойчивости + численное решение ẏ=λy.
  3. «Переменный комплексный коэффициент» — слайдеры Re(w₁), Im(w₁), Δt;
     путь в комплексной плоскости + область устойчивости + численное
     решение ẏ=y с заданными весами.

Запуск:
    python gui.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np

# Импортируем готовые «движки» режимов
from controllable_stability_gui import ControllableStabilityApp
from varying_coefficient_gui  import VaryingCoefficientApp


# Простейший «выбиратель» режима — три отдельных окна по запросу.
def main():
    print("Главное меню v2. Выберите режим:")
    print("  1 — Сравнение методов на наборе ОДУ")
    print("  2 — Управляемая устойчивость (слайдеры Re(λ), Im(λ), Δt)")
    print("  3 — Переменный комплексный коэффициент (слайдеры Re(w₁), Im(w₁))")
    print("Введите 1, 2 или 3 (Enter = 1):")
    try:
        choice = input("> ").strip() or "1"
    except EOFError:
        choice = "1"

    if choice == "2":
        _ = ControllableStabilityApp()
        plt.show()
    elif choice == "3":
        _ = VaryingCoefficientApp()
        plt.show()
    else:
        # Режим 1 — тот же простой сравнитель, что в v1 (импортируем
        # из experiments.py). Здесь — мини-GUI на выбор уравнения.
        from matplotlib.widgets import Slider
        from complex_euler import (classic_euler, complex_two_step_euler,
                                   infinity_norm_error)
        from equations import EQUATIONS
        EQ_KEYS = ["linear", "square", "nlsin", "cos", "shm"]

        fig = plt.figure(figsize=(10.5, 6.5))
        fig.suptitle("Сравнение методов на наборе ОДУ", fontsize=12)
        ax = fig.add_axes([0.30, 0.20, 0.66, 0.70])

        rax = fig.add_axes([0.02, 0.50, 0.20, 0.40])
        rax.set_title("Уравнение", fontsize=10)
        radio = RadioButtons(rax, EQ_KEYS, active=0)

        sax = fig.add_axes([0.30, 0.07, 0.55, 0.04])
        slider = Slider(sax, "n_macro", 2, 200, valinit=10, valstep=1)

        text_ax = fig.add_axes([0.02, 0.05, 0.22, 0.40])
        text_ax.axis("off")
        th = text_ax.text(0.02, 0.95, "", transform=text_ax.transAxes,
                          family="monospace", fontsize=9, va="top")

        state = {"key": EQ_KEYS[0]}

        def redraw():
            eq = EQUATIONS[state["key"]]
            n = int(slider.val)
            t_c, y_c = complex_two_step_euler(eq.f, eq.y0, eq.t0, eq.tN, n)
            t_r, y_r = classic_euler(eq.f, eq.y0, eq.t0, eq.tN, 2 * n)
            t_e = np.linspace(eq.t0, eq.tN, 400)
            y_e = eq.exact(t_e) if eq.exact is not None else None
            if y_c.ndim == 2: y_c = y_c[:, 0]
            if y_r.ndim == 2: y_r = y_r[:, 0]
            if y_e is not None and np.ndim(y_e) == 2: y_e = y_e[:, 0]
            ax.clear()
            if y_e is not None:
                ax.plot(t_e, y_e, color="#2ca02c", linewidth=2.0,
                        label="точное решение")
            ax.plot(t_r, y_r, "o--", color="#1f77b4", markersize=4,
                    label=f"классич. ({2*n} ш.)")
            ax.plot(t_c, y_c, "s-", color="#000000", markersize=4,
                    label=f"компл. 2-шаг. ({n} макрош.)")
            ax.set_title(eq.title, fontsize=11)
            ax.set_xlabel("t"); ax.set_ylabel("y")
            ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
            info = []
            if eq.exact is not None:
                ye_c = eq.exact(t_c); ye_r = eq.exact(t_r)
                if ye_c.ndim == 2: ye_c = ye_c[:, 0]
                if ye_r.ndim == 2: ye_r = ye_r[:, 0]
                er = infinity_norm_error(y_r, ye_r)
                ec = infinity_norm_error(y_c, ye_c)
                info.append(f"Δt = {(eq.tN-eq.t0)/n:.4g}")
                info.append("")
                info.append(f"‖err‖∞:")
                info.append(f"  классич.: {er:.3e}")
                info.append(f"  компл.  : {ec:.3e}")
                if ec > 0:
                    info.append("")
                    info.append(f"улучшение ×{er/ec:7.2f}")
            th.set_text("\n".join(info))
            fig.canvas.draw_idle()

        def on_key(key):
            state["key"] = key
            redraw()
        def on_slider(_v):
            redraw()
        radio.on_clicked(on_key); slider.on_changed(on_slider)
        redraw()
        plt.show()


if __name__ == "__main__":
    main()
