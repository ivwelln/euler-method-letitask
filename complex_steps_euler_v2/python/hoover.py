import numpy as np  
import matplotlib.pyplot as plt  
from scipy.integrate import solve_ivp  
  
# =========================================================  
# 1. ПРАВЫЕ ЧАСТИ (система Нозе-Гувера для гармонического осциллятора)  
# =========================================================  
  
def nose_hoover_rhs(t, state, omega_sq=1.0, T0=1.0, tau=1.0):  
    """  
    Правые части для термостата Нозе-Гувера.  
    dx/dt = v  
    dv/dt = -omega_sq * x - zeta * v  
    dzeta/dt = (1/tau^2) * (v^2 - T0)  
    """  
    x, v, zeta = state  
      
    dxdt = v  
    dvdt = -omega_sq * x - zeta * v  
    dzeta_dt = (1.0 / tau**2) * (v**2 - T0)  
      
    return np.array([dxdt, dvdt, dzeta_dt])  
  
  
# =========================================================  
# 2. КЛАССИЧЕСКИЙ МЕТОД ЭЙЛЕРА  
# =========================================================  
  
def euler_classic(f, y0, t_span, dt):  
    """  
    Явный метод Эйлера 1-го порядка.  
    """  
    t0, t_end = t_span  
    n_steps = int((t_end - t0) / dt)  
      
    t = np.linspace(t0, t_end, n_steps + 1)  
    y = np.zeros((n_steps + 1, len(y0)))  
    y[0] = y0  
      
    for i in range(n_steps):  
        y[i+1] = y[i] + dt * f(t[i], y[i])  
      
    return t, y  
  
  
# =========================================================  
# 3. ВАШ КОМПЛЕКСНЫЙ 2-ШАГОВЫЙ МЕТОД ЭЙЛЕРА  
# =========================================================  
  
def euler_complex_2step(f, y0, t_span, dt, w1=0.5+0.5j):  
    """  
    Комплексный 2-шаговый метод.  
    w1 = 0.5 ± 0.5i  -> 2-й порядок  
    w2 = 1 - w1  
    """  
    t0, t_end = t_span  
    n_macro = int((t_end - t0) / dt)          # число макрошагов  
    dim = len(y0)                             # размерность системы  
      
    # Веса (w2 определяется из условия 1-го порядка)  
    w2 = 1.0 - w1  
      
    # Массивы для времени и решения (только вещественные части)  
    t = np.linspace(t0, t_end, n_macro + 1)  
    y = np.zeros((n_macro + 1, dim), dtype=float)  
    y[0] = y0  
      
    for k in range(n_macro):  
        # Текущее время (вещественное)  
        tk = t[k]  
        yk = y[k]  
          
        # ПОДШАГ 1: комплексный  
        y_star = yk + w1 * dt * f(tk, yk)           # комплексное  
        t_star = tk + w1 * dt  
          
        # ПОДШАГ 2: комплексный  
        # Нужно привести t_star и y_star к вещественному виду для f?  
        # В общем случае f ожидает вещественные аргументы.  
        # Для уравнений Нозе-Гувера f зависит от x, v, zeta (все вещественные).  
        # Поэтому берём вещественные части.  
        t_star_real = t_star.real  
        y_star_real = y_star.real  
          
        y_new = y_star + w2 * dt * f(t_star_real, y_star_real)  
          
        # Возврат на вещественную ось  
        y[k+1] = y_new.real  
      
    return t, y  
  
  
# =========================================================  
# 4. ПАРАМЕТРЫ МОДЕЛИ  
# =========================================================  
  
omega_sq = 1.0      # квадрат частоты (ω² = k/m)  
T0 = 1.0            # целевая температура  
tau = 2.0           # время релаксации термостата  
  
# Начальные условия: x(0)=1, v(0)=0, zeta(0)=0  
y0 = np.array([1.0, 0.0, 0.0])  
  
dt = 0.01           # шаг для классического метода (2 шага по dt)  
t_span = (0.0, 50.0)  
  
  
# =========================================================  
# 5. ЗАПУСК РАСЧЁТОВ  
# =========================================================  
  
print("=" * 60)  
print("Сравнение методов для термостата Нозе-Гувера")  
print("=" * 60)  
  
# А) Классический Эйлер  
t_classic, y_classic = euler_classic(  
    lambda t, y: nose_hoover_rhs(t, y, omega_sq, T0, tau),  
    y0, t_span, dt  
)  
  
# Б) Ваш комплексный метод (канонический: 2-й порядок)  
t_complex, y_complex = euler_complex_2step(  
    lambda t, y: nose_hoover_rhs(t, y, omega_sq, T0, tau),  
    y0, t_span, dt, w1=0.5+0.5j  
)  
  
# В) Для справки: "точное" решение (Рунге-Кутта 4-5 с мелким шагом)  
sol_ref = solve_ivp(  
    lambda t, y: nose_hoover_rhs(t, y, omega_sq, T0, tau),  
    t_span, y0, method='RK45', rtol=1e-8, atol=1e-10  
)  
t_ref = sol_ref.t  
y_ref = sol_ref.y.T  
  
  
# =========================================================  
# 6. ВИЗУАЛИЗАЦИЯ  
# =========================================================  
  
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  
  
# --- Панель 1: Положение x(t) ---  
ax = axes[0, 0]  
ax.plot(t_ref, y_ref[:, 0], 'k-', linewidth=1, alpha=0.7, label='Reference (RK45)')  
ax.plot(t_classic, y_classic[:, 0], 'b--', linewidth=0.8, label='Euler classic')  
ax.plot(t_complex, y_complex[:, 0], 'r-', linewidth=0.8, label='Complex Euler 2-step')  
ax.set_xlabel('t')  
ax.set_ylabel('x (position)')  
ax.set_title('Position vs Time')  
ax.legend(fontsize=8)  
ax.grid(True, alpha=0.3)  
  
# --- Панель 2: Температура (v²) ---  
ax = axes[0, 1]  
T_ref = y_ref[:, 1]**2  
T_classic = y_classic[:, 1]**2  
T_complex = y_complex[:, 1]**2  
ax.plot(t_ref, T_ref, 'k-', linewidth=1, alpha=0.7, label='Reference')  
ax.plot(t_classic, T_classic, 'b--', linewidth=0.8, label='Euler classic')  
ax.plot(t_complex, T_complex, 'r-', linewidth=0.8, label='Complex Euler')  
ax.axhline(y=T0, color='gray', linestyle=':', linewidth=1, label=f'T_target = {T0}')  
ax.set_xlabel('t')  
ax.set_ylabel('v² (temperature)')  
ax.set_title('Kinetic Temperature')  
ax.legend(fontsize=8)  
ax.grid(True, alpha=0.3)  
  
# --- Панель 3: Фазовая плоскость (x, v) ---  
ax = axes[1, 0]  
ax.plot(y_ref[:, 0], y_ref[:, 1], 'k-', linewidth=0.5, alpha=0.5, label='Reference')  
ax.plot(y_classic[:, 0], y_classic[:, 1], 'b--', linewidth=0.5, alpha=0.7, label='Euler classic')  
ax.plot(y_complex[:, 0], y_complex[:, 1], 'r-', linewidth=0.5, alpha=0.7, label='Complex Euler')  
ax.set_xlabel('x')  
ax.set_ylabel('v')  
ax.set_title('Phase Portrait')  
ax.legend(fontsize=8)  
ax.grid(True, alpha=0.3)  
  
# --- Панель 4: Переменная термостата ζ(t) ---  
ax = axes[1, 1]  
ax.plot(t_ref, y_ref[:, 2], 'k-', linewidth=1, alpha=0.7, label='Reference')  
ax.plot(t_classic, y_classic[:, 2], 'b--', linewidth=0.8, label='Euler classic')  
ax.plot(t_complex, y_complex[:, 2], 'r-', linewidth=0.8, label='Complex Euler')  
ax.set_xlabel('t')  
ax.set_ylabel('ζ')  
ax.set_title('Thermostat Variable')  
ax.legend(fontsize=8)  
ax.grid(True, alpha=0.3)  
  
plt.tight_layout()  
  
# =========================================================  
# 7. ДИАГНОСТИКА ФАЗОВОГО ОБЪЁМА  
# =========================================================  
  
def compute_phase_space_divergence(traj, dt):  
    """  
    Оценка сжатия/расширения фазового объёма.  
    В трёхмерном случае: sum_i (∂ẋ_i/∂x_i)  
    Для Нозе-Гувера: div = -ζ (теоретически)  
    """  
    # Второй столбец = v, третий = ζ  
    # Траектория: [x, v, ζ]  
    zeta = traj[:, 2]  
    divergence = -zeta  # дивергенция = -ζ  
    return divergence  
  
div_classic = compute_phase_space_divergence(y_classic, dt)  
div_complex = compute_phase_space_divergence(y_complex, dt)  
  
fig2, ax = plt.subplots(figsize=(10, 4))  
ax.plot(t_classic[1:], div_classic[1:], 'b--', linewidth=0.8, label='Euler classic (div = -ζ)')  
ax.plot(t_complex[1:], div_complex[1:], 'r-', linewidth=0.8, label='Complex Euler (div = -ζ)')  
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  
ax.set_xlabel('t')  
ax.set_ylabel('divergence')  
ax.set_title('Phase Space Divergence (compression if negative)')  
ax.legend()  
ax.grid(True, alpha=0.3)  
plt.tight_layout()  
  
plt.show()  
  
  
# =========================================================  
# 8. ВЫВОД ДИАГНОСТИКИ В КОНСОЛЬ  
# =========================================================  
  
print("\n" + "=" * 60)  
print("ДИАГНОСТИКА")  
print("=" * 60)  
  
# Средняя температура (v²) на последнем участке  
t_start_steady = 40.0  
mask = (t_complex >= t_start_steady)  
T_avg_complex = np.mean(y_complex[mask, 1]**2)  
T_avg_classic = np.mean(y_classic[mask, 1]**2)  
  
print(f"Целевая температура T0 = {T0}")  
print(f"Средняя T (комплексный метод): {T_avg_complex:.4f}")  
print(f"Средняя T (классический метод): {T_avg_classic:.4f}")  
  
# Дивергенция (сжатие фазового объёма)  
print(f"\nСредняя дивергенция (комплексный): {np.mean(div_complex[mask]):.4f} (отрицательная → сжатие)")  
print(f"Средняя дивергенция (классический): {np.mean(div_classic[mask]):.4f} (отрицательная → сжатие)")  
  
# Ошибка относительно референса  
y_ref_interp = np.interp(t_complex, t_ref, y_ref[:, 1])  
error_complex = np.linalg.norm(y_complex[:, 1] - y_ref_interp) / np.sqrt(len(t_complex))  
error_classic = np.linalg.norm(y_classic[:, 1] - y_ref_interp) / np.sqrt(len(t_classic))  
  
print(f"\nRMS ошибка скорости (complex): {error_complex:.6e}")  
print(f"RMS ошибка скорости (classic):  {error_classic:.6e}")  
print(f"Улучшение: {error_classic/error_complex:.1f} раз")  
