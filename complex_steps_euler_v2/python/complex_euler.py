# -*- coding: utf-8 -*-
"""
complex_euler.py  (v2 — расширенная)
====================================

Те же два метода, что в v1 (классический Эйлер и 2-шаговый комплексный
Эйлер), но реализация комплексного метода теперь принимает ПРОИЗВОЛЬНЫЕ
веса w1, w2. Это нужно для трёх новых экспериментов второй версии:

  • controllable_stability_gui.py — пользователь меняет λ и Δt;
  • varying_coefficient_gui.py   — пользователь меняет w₁ (а w₂ = 1 − w₁);
  • ring_test.py                  — round-trip туда-обратно.

«Канонические» веса 2-го порядка по-прежнему:
    w1 = 1/2 + i/2,   w2 = 1/2 − i/2.
Условия 2-го порядка:
    w1 + w2 = 1,      w1 · w2 = 1/2.

Если эти условия НАРУШЕНЫ, метод теряет 2-й порядок и становится 1-м.
Если нарушено даже первое — метод вообще «не Эйлер», по сути произвольная
схема. Функция print-предупреждает об отклонениях, но позволяет считать —
это удобно для демонстрационных экспериментов.
"""

from __future__ import annotations

import numpy as np


# Канонические веса 2-го порядка
W1 = 0.5 + 0.5j
W2 = 0.5 - 0.5j


# -----------------------------------------------------------------------------
# 1. Классический явный метод Эйлера
# -----------------------------------------------------------------------------
def classic_euler(f, y0, t0: float, tN: float, n_steps: int):
    """y_{k+1} = y_k + Δt · f(t_k, y_k). Глобальный порядок 1.

    Автоматически выбирает тип данных: если правая часть возвращает
    комплексные значения (например, для ẏ=λy с комплексным λ), массив
    решения тоже хранится в complex.
    """
    dt = (tN - t0) / n_steps
    t = np.linspace(t0, tN, n_steps + 1)

    y0_arr = np.atleast_1d(np.asarray(y0))
    # Пробуем оценить тип: вычисляем f(t0, y0) и смотрим на dtype результата.
    try:
        probe = np.asarray(f(t0, y0_arr if y0_arr.size > 1 else y0_arr[0]))
        is_complex = np.iscomplexobj(probe) or np.iscomplexobj(y0_arr)
    except Exception:
        is_complex = np.iscomplexobj(y0_arr)
    dtype = complex if is_complex else float

    y0_arr = y0_arr.astype(dtype)
    y = np.empty((n_steps + 1, y0_arr.size), dtype=dtype)
    y[0] = y0_arr

    for k in range(n_steps):
        y[k + 1] = y[k] + dt * np.asarray(f(t[k], y[k]), dtype=dtype)

    if y.shape[1] == 1:
        return t, y[:, 0]
    return t, y


# -----------------------------------------------------------------------------
# 2. Комплексный 2-шаговый Эйлер с ПРОИЗВОЛЬНЫМИ весами
# -----------------------------------------------------------------------------
def complex_two_step_euler(f, y0, t0: float, tN: float, n_macro: int,
                           w1: complex = W1, w2: complex | None = None,
                           keep_imag: bool = False,
                           warn: bool = False):
    """2-шаговый Эйлер с комплексными весами w1, w2.

    Параметры
    ---------
    w1 : complex
        Первый комплексный полу-шаг (множитель при Δt).
    w2 : complex или None
        Второй полу-шаг. Если None — берётся (1 − w1), чтобы гарантировать
        как минимум 1-й порядок (условие w1 + w2 = 1).
    keep_imag : bool
        Если True — после второго подшага НЕ берётся Re(·). Полезно
        для исследований; для практики оставляйте False.
    warn : bool
        Печатать предупреждение, если веса нарушают условия 2-го порядка.
    """
    if w2 is None:
        w2 = 1.0 - w1

    if warn:
        eps = 1e-9
        s = w1 + w2
        p = w1 * w2
        if abs(s - 1) > eps:
            print(f"[complex2] w1+w2 = {s} (ожидалось 1) — потерян 1-й порядок")
        if abs(p - 0.5) > eps:
            print(f"[complex2] w1·w2 = {p} (ожидалось 0.5) — потерян 2-й порядок")

    dt = (tN - t0) / n_macro
    t = np.linspace(t0, tN, n_macro + 1)

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=complex))
    out = np.empty((n_macro + 1, y0_arr.size),
                   dtype=complex if keep_imag else float)
    out[0] = y0_arr if keep_imag else y0_arr.real

    yc = y0_arr.copy()
    for k in range(n_macro):
        tk = t[k]
        y_star = yc + w1 * dt * np.asarray(f(tk, yc), dtype=complex)
        t_star = tk + w1 * dt
        yc_new = y_star + w2 * dt * np.asarray(f(t_star, y_star), dtype=complex)
        if not keep_imag:
            yc = yc_new.real.astype(complex)
            out[k + 1] = yc.real
        else:
            yc = yc_new
            out[k + 1] = yc

    if out.shape[1] == 1:
        return t, out[:, 0]
    return t, out


# -----------------------------------------------------------------------------
# 3. Обратный шаг (для round-trip / кольцевого теста)
# -----------------------------------------------------------------------------
def classic_euler_reverse(f, y0, t0: float, tN: float, n_steps: int):
    """Тот же Эйлер, но dt отрицательный — идём от tN к t0."""
    # Просто переворачиваем направление: интегрируем от tN к t0
    # с отрицательным шагом. Удобно реализовать через замену t' = -t.
    dt = (tN - t0) / n_steps          # это положительное число (tN > t0)
    # шаг назад имеет длину -dt
    t = np.linspace(tN, t0, n_steps + 1)

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    y = np.empty((n_steps + 1, y0_arr.size), dtype=float)
    y[0] = y0_arr

    for k in range(n_steps):
        y[k + 1] = y[k] + (-dt) * np.asarray(f(t[k], y[k]), dtype=float)

    if y.shape[1] == 1:
        return t, y[:, 0]
    return t, y


def complex_two_step_euler_reverse(f, y0, t0: float, tN: float, n_macro: int,
                                   w1: complex = W1, w2: complex | None = None):
    """То же, но идём от tN к t0 с шагом dt = -(tN-t0)/n_macro."""
    if w2 is None:
        w2 = 1.0 - w1
    dt = -(tN - t0) / n_macro          # отрицательный
    t = np.linspace(tN, t0, n_macro + 1)

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=complex))
    out = np.empty((n_macro + 1, y0_arr.size), dtype=float)
    out[0] = y0_arr.real

    yc = y0_arr.copy()
    for k in range(n_macro):
        tk = t[k]
        y_star = yc + w1 * dt * np.asarray(f(tk, yc), dtype=complex)
        t_star = tk + w1 * dt
        yc_new = y_star + w2 * dt * np.asarray(f(t_star, y_star), dtype=complex)
        yc = yc_new.real.astype(complex)
        out[k + 1] = yc.real

    if out.shape[1] == 1:
        return t, out[:, 0]
    return t, out


# -----------------------------------------------------------------------------
# 4. Функции устойчивости (как функции от w1)
# -----------------------------------------------------------------------------
def stability_classic(z):
    """Φ(z) = 1 + z (классический Эйлер)."""
    return 1.0 + z


def stability_complex2(z, w1: complex = W1, w2: complex | None = None):
    """Φ(z) = (1 + w1·z)(1 + w2·z). Зависит от выбранных весов!

    При w1+w2=1, w1·w2=1/2 совпадает с 1 + z + z²/2.
    При w1+w2=1, но другим произведении — другая 2-степенная функция.
    """
    if w2 is None:
        w2 = 1.0 - w1
    return (1.0 + w1 * z) * (1.0 + w2 * z)


def infinity_norm_error(y_num: np.ndarray, y_exact: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(y_num) - np.asarray(y_exact))))


__all__ = [
    "classic_euler", "complex_two_step_euler",
    "classic_euler_reverse", "complex_two_step_euler_reverse",
    "stability_classic", "stability_complex2",
    "infinity_norm_error",
    "W1", "W2",
]


def stability_complex3(z):
    """Φ(z) = 1 + z + z²/2 + z³/6 — для справки, 3-шаговый комплексный Эйлер."""
    return 1.0 + z + 0.5 * z ** 2 + (1.0 / 6.0) * z ** 3
