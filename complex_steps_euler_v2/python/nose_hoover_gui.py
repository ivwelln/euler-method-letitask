import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.gridspec import GridSpec
from datetime import datetime

# ── цвета ──────────────────────────────────────────────────
C_BG      = '#FFFFFF'
C_PANEL   = '#F0F2F5'
C_BORDER  = '#CCCCCC'
C_GRID    = '#E0E0E0'
C_TEXT    = '#1A1A2E'
C_CLASSIC = '#2B4EE6'   # синий  — классический Эйлер
C_COMPLEX = '#0F9E5E'   # зелёный — комплексный Эйлер
C_POINT   = '#CC2222'   # красный — точка z = i·Δt
C_ATTRACTOR = '#1A6FAF' # синий   — аттрактор
C_BOX     = '#E84040'   # красный — bounding box

# ── параметры по умолчанию ─────────────────────────────────
params = {
    're': 0.5,    # Re(w₁)
    'im': 0.5,    # Im(w₁)
    'dt': 0.05,   # Δt
    'T':  200.0,  # время интегрирования
}
ranges = {
    're': (-1.0, 1.5),
    'im': (-1.5, 1.5),
    'dt': (0.001, 0.5),
    'T':  (50.0, 1000.0),
}
labels = {
    're': 'Re(w₁)',
    'im': 'Im(w₁)',
    'dt': 'Δt',
    'T':  'T (время)',
}


_updating = False  # защита от рекурсии

#  СИСТЕМА НОЗЕ-ГУВЕРА

def nose_hoover_rhs(state):
    """
    Правая часть системы Нозе-Гувера:
        ẋ = y
        ẏ = −x − ζ·y
        ζ̇ = y² − 1
    """
    x, y, z = state
    return np.array([y,
                     -x - z * y,
                     y * y - 1.0])


def integrate_complex_euler(rhs, state0, T, dt, w1):
    """
    Интегрирование системы комплексным двухшаговым методом Эйлера.

    Параметры:
        rhs    — правая часть f(state)
        state0 — начальное условие
        T      — время интегрирования
        dt     — макрошаг
        w1     — комплексный вес w₁ (w₂ = 1 − w₁)

    Возвращает:
        trajectory — массив (N+1, 3) с траекторией
    """
    # w2 = 0.66
    w2 = 1.0 - w1
    h1 = w1 * dt   # комплексный подшаг 1
    h2 = w2 * dt   # комплексный подшаг 2

    N = int(T / dt)
    trajectory = np.zeros((N + 1, 3))
    trajectory[0] = state0

    state = state0.astype(complex)

    for k in range(N):
        f1 = rhs(state)                       # f в текущей комплексной точке
        y_star = state + h1 * f1              # подшаг 1 (комплексный)
        f2 = rhs(y_star)                      # f в комплексной промежуточной точке
        y_new = y_star + h2 * f2              # подшаг 2
        state = y_new.real.astype(complex)    # берём Re, возврат на ось
        trajectory[k + 1] = state.real

    return trajectory



#  ОБЛАСТИ УСТОЙЧИВОСТИ (вычисляем один раз)

def phi_classic(z):
    return 1.0 + z


def phi_complex(z, w1):
    w2 = 1.0 - w1
    return (1.0 + w1 * z) * (1.0 + w2 * z)


def make_stability_grid(N=400, re_range=(-4, 2), im_range=(-3, 3)):
    re = np.linspace(*re_range, N)
    im = np.linspace(*im_range, N)
    RE, IM = np.meshgrid(re, im)
    Z = RE + 1j * IM
    return RE, IM, Z


RE_GRID, IM_GRID, Z_GRID = make_stability_grid()
ABS_CLASSIC = np.abs(phi_classic(Z_GRID))


#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def bounding_box(traj):
    """Возвращает (min, max) по каждой оси."""
    mn = traj.min(axis=0)
    mx = traj.max(axis=0)
    return mn, mx


def box_volume(mn, mx):
    """Объём параллелепипеда."""
    sides = mx - mn
    return sides[0] * sides[1] * sides[2]


def box_area_xy(mn, mx):
    """Площадь проекции на плоскость XY."""
    return (mx[0] - mn[0]) * (mx[1] - mn[1])


def draw_bounding_box(ax, mn, mx):
    """Рисует wireframe параллелепипед вокруг аттрактора."""
    x0, y0, z0 = mn
    x1, y1, z1 = mx

    # 12 рёбер параллелепипеда
    edges = [
        # нижнее основание
        [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
        # верхнее основание
        [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
        # вертикальные рёбра
        [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
    ]
    for e in edges:
        xs = [e[0][0], e[1][0]]
        ys = [e[0][1], e[1][1]]
        zs = [e[0][2], e[1][2]]
        ax.plot(xs, ys, zs, color=C_BOX, lw=1.2, alpha=0.7)



#  ФИГУРА И КОМПОНОВКА

plt.rcParams.update({
    'figure.facecolor':  C_BG,
    'axes.facecolor':    C_BG,
    'axes.edgecolor':    C_BORDER,
    'axes.labelcolor':   C_TEXT,
    'xtick.color':       C_TEXT,
    'ytick.color':       C_TEXT,
    'text.color':        C_TEXT,
    'grid.color':        C_GRID,
    'grid.alpha':        0.6,
})
plt.style.use('default')

fig = plt.figure(figsize=(17, 10), facecolor=C_BG)
fig.canvas.manager.set_window_title(
    'Управление фазовым объёмом системы Нозе-Гувера')

gs = GridSpec(2, 2, figure=fig,
              left=0.05, right=0.97,
              top=0.86,  bottom=0.34,
              wspace=0.35, hspace=0.45)

ax_stab = fig.add_subplot(gs[:, 0])       # область устойчивости
ax_3d   = fig.add_subplot(gs[:, 1],
                           projection='3d')  # 3D аттрактор
ax_info = None  # информационная строка — текст на fig

ROW_H = 0.028
SL_W = 0.25
TB_W = 0.07
CONTROL_ROWS = {
    're': (0.07, 0.205),
    'im': (0.07, 0.135),
    'dt': (0.55, 0.205),
    'T':  (0.55, 0.135),
}
TB_GAP = 0.02
colors_sl = {
    're': C_CLASSIC,
    'im': C_COMPLEX,
    'dt': C_POINT,
    'T':  '#888888',
}

sliders   = {}
textboxes = {}

for key, (xpos, ypos) in CONTROL_ROWS.items():
    ax_sl = fig.add_axes([xpos, ypos, SL_W, ROW_H],
                         facecolor=C_PANEL)
    sl = Slider(ax_sl, labels[key],
                ranges[key][0], ranges[key][1],
                valinit=params[key],
                color=colors_sl[key])
    sl.label.set_color(C_TEXT)
    sl.valtext.set_visible(False)
    sliders[key] = sl

    ax_tb = fig.add_axes([xpos + SL_W + TB_GAP, ypos, TB_W, ROW_H],
                         facecolor=C_PANEL)
    tb = TextBox(ax_tb, '',
                 initial=f'{params[key]:.3f}',
                 color=C_PANEL,
                 hovercolor='#E0E4EA',
                 label_pad=0.0)
    tb.text_disp.set_color(C_TEXT)
    textboxes[key] = tb

fig.text(0.5, 0.295,
         'Enter для применения значений из полей ввода',
         color='#888', fontsize=8, ha='center')

# ── кнопка сохранения 
ax_btn = fig.add_axes([0.44, 0.055, 0.12, 0.035],
                      facecolor=C_PANEL)
btn_save = Button(ax_btn, 'Сохранить PNG',
                  color=C_PANEL, hovercolor='#D0D8E4')
btn_save.label.set_color(C_TEXT)

# ── заголовок (строка 1) ───────────────────────────────────
fig.text(0.5, 0.988,
         'Управление фазовым объёмом системы Нозе-Гувера',
         ha='center', va='top',
         fontsize=13, fontweight='bold', color=C_TEXT)

fig.text(0.5, 0.968,
         'через изменение комплексного коэффициента w₁ и области устойчивости',
         ha='center', va='top',
         fontsize=10, color=C_TEXT)

# ── информационная строка (строка 2): w1 и порядок 
info_line1 = fig.text(0.5, 0.945, '',
                      ha='center', va='center',
                      fontsize=10, color=C_TEXT,
                      fontfamily='monospace')

# ── информационная строка (строка 3): устойчивость 
info_line2 = fig.text(0.5, 0.922, '',
                      ha='center', va='center',
                      fontsize=10, color=C_TEXT,
                      fontfamily='monospace')



#  ФУНКЦИЯ ОТРИСОВКИ

# кэш траектории
_cache = {'traj': None, 'w1': None, 'dt': None, 'T': None}


def get_trajectory(w1, dt, T):
    """Возвращает траекторию (с кэшированием)."""
    if (_cache['traj'] is not None and
            _cache['w1'] == w1 and
            _cache['dt'] == dt and
            _cache['T'] == T):
        return _cache['traj']

    state0 = np.array([0.0, 5.0, 0.0])  # начальные условия
    # 
    # прогрев — отбрасываем первые 20% для выхода на аттрактор
    warmup_T = T * 0.2
    traj_warmup = integrate_complex_euler(
        nose_hoover_rhs, state0, warmup_T, dt, w1)
    state1 = traj_warmup[-1]

    traj = integrate_complex_euler(
        nose_hoover_rhs, state1, T, dt, w1)

    _cache['traj'] = traj
    _cache['w1']   = w1
    _cache['dt']   = dt
    _cache['T']    = T
    return traj


def draw(val=None):
    re_w1 = params['re']
    im_w1 = params['im']
    dt    = max(params['dt'], 0.001)
    T     = params['T']
    w1    = complex(re_w1, im_w1)
    w2    = 1.0 - w1

    # ── вычисляем траекторию ───────────────────────────────
    try:
        traj = get_trajectory(w1, dt, T)
        traj_ok = True
    except Exception:
        traj_ok = False

    # ── левая панель: область устойчивости ────────────────
    ax_stab.clear()
    ax_stab.set_facecolor(C_BG)

    # классический Эйлер (постоянная сетка)
    ax_stab.contourf(RE_GRID, IM_GRID, ABS_CLASSIC,
                     levels=[0, 1], colors=[C_CLASSIC], alpha=0.20)
    ax_stab.contour(RE_GRID, IM_GRID, ABS_CLASSIC,
                    levels=[1], colors=[C_CLASSIC], linewidths=2)

    # комплексный Эйлер (зависит от w₁)
    ABS_COMP = np.abs(phi_complex(Z_GRID, w1))
    ax_stab.contourf(RE_GRID, IM_GRID, ABS_COMP,
                     levels=[0, 1], colors=[C_COMPLEX], alpha=0.18)
    ax_stab.contour(RE_GRID, IM_GRID, ABS_COMP,
                    levels=[1], colors=[C_COMPLEX], linewidths=2)

    # точка z = i·Δt (собственное значение λ=i при масштабе Δt)
    z_point = 1j * dt
    ax_stab.plot(z_point.real, z_point.imag, 'o',
                 color=C_POINT, ms=12, zorder=10,
                 markeredgecolor='#660000', markeredgewidth=1.5,
                 label=f'z = i·Δt = {z_point.imag:.3f}i')

    # сопряжённая точка z = −i·Δt
    z_conj = -1j * dt
    ax_stab.plot(z_conj.real, z_conj.imag, 'o',
                 color=C_POINT, ms=8, zorder=10,
                 markeredgecolor='#660000', markeredgewidth=1.2,
                 alpha=0.6)

    ax_stab.annotate(f'z=i·{dt:.3f}',
                     xy=(z_point.real, z_point.imag),
                     xytext=(z_point.real + 0.25,
                             z_point.imag + 0.15),
                     color=C_POINT, fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->',
                                     color=C_POINT, lw=1.2))

    # оси
    ax_stab.axhline(0, color=C_BORDER, lw=0.8)
    ax_stab.axvline(0, color=C_BORDER, lw=0.8)
    ax_stab.grid(True, color=C_GRID, alpha=0.6, lw=0.5)

    # легенда
    p1 = mpatches.Patch(color=C_CLASSIC, alpha=0.6,
                        label='Классический: |1+z|≤1')
    p2 = mpatches.Patch(color=C_COMPLEX,  alpha=0.6,
                        label=f'Комплексный w₁={re_w1:.2f}'
                              f'{im_w1:+.2f}i: |Φ(z)|≤1')
    leg = ax_stab.legend(handles=[p1, p2], fontsize=8,
                         facecolor=C_PANEL, edgecolor=C_BORDER,
                         loc='upper right')
    for t in leg.get_texts():
        t.set_color(C_TEXT)

    ax_stab.set_xlim(-4.5, 2.5)
    ax_stab.set_ylim(-3.2, 3.2)
    ax_stab.set_xlabel('Re(z)', fontsize=10)
    ax_stab.set_ylabel('Im(z)', fontsize=10)
    ax_stab.set_title('Области абсолютной устойчивости\n'
                      '(точка z = i·Δt — собств. знач. Нозе-Гувера)',
                      fontsize=11, color=C_TEXT)

    # проверяем попадание точки
    in_classic = abs(phi_classic(z_point)) <= 1
    in_complex = abs(phi_complex(z_point, w1)) <= 1
    stab_info = (
        f"Классический: {'✓ устойчив' if in_classic else '✗ неустойчив'} "
        f"| Комплексный: {'✓ устойчив' if in_complex else '✗ неустойчив'}"
    )

    # 3D аттрактор 
    ax_3d.clear()
    ax_3d.set_facecolor(C_BG)

    vol_str = ''
    area_str = ''

    if traj_ok and len(traj) > 10:
        x, y, z_nh = traj[:, 0], traj[:, 1], traj[:, 2]

        #  для скорости отрисовки
        step = max(1, len(traj) // 5000)
        xs, ys, zs = x[::step], y[::step], z_nh[::step]

        #  аттрактор
        ax_3d.plot(xs, ys, zs,
                   color=C_ATTRACTOR, lw=0.4, alpha=0.6)

        mn, mx = bounding_box(traj)
        draw_bounding_box(ax_3d, mn, mx)

        # подписи сторон
        cx = (mn[0] + mx[0]) / 2
        cy = (mn[1] + mx[1]) / 2
        cz = (mn[2] + mx[2]) / 2
        dx = mx[0] - mn[0]
        dy = mx[1] - mn[1]
        dz = mx[2] - mn[2]
        vol  = dx * dy * dz
        area = dx * dy

        vol_str  = f'V = {vol:.2f}'
        area_str = f'(ΔX={dx:.2f}) × (ΔY={dy:.2f}) × (ΔZ={dz:.2f})'

        ax_3d.set_xlabel('x', fontsize=9, labelpad=2)
        ax_3d.set_ylabel('y', fontsize=9, labelpad=2)
        ax_3d.set_zlabel('ζ', fontsize=9, labelpad=2)

        ax_3d.set_title(
            'Аттрактор Нозе-Гувера',
            fontsize=11, color=C_TEXT)
        # размеры и объём под заголовком
        ax_3d.text2D(0.5, 1.07,
                     area_str,
                     transform=ax_3d.transAxes,
                     ha='center', va='bottom',
                     fontsize=8, color=C_TEXT,
                     fontfamily='monospace')
        ax_3d.text2D(0.5, 0,
                     f'Фазовый объём  V = {vol:.3f}',
                     transform=ax_3d.transAxes,
                     ha='center', va='bottom',
                     fontsize=9, color=C_BOX,
                     fontweight='bold')

        # легенда box
        box_patch = mpatches.Patch(
            color=C_BOX, alpha=0.7,
            label=f'Bounding box  V={vol:.2f}')
        attr_patch = mpatches.Patch(
            color=C_ATTRACTOR, alpha=0.8,
            label='Аттрактор')
        leg2 = ax_3d.legend(handles=[attr_patch, box_patch],
                            fontsize=8, facecolor=C_PANEL,
                            edgecolor=C_BORDER, loc='upper left')
        for t in leg2.get_texts():
            t.set_color(C_TEXT)

    else:
        ax_3d.text(0.5, 0.5, 0.5,
                   'Вычисление...', transform=ax_3d.transAxes,
                   ha='center', color=C_TEXT, fontsize=12)
        ax_3d.set_title('Аттрактор Нозе-Гувера', color=C_TEXT)

    ax_3d.tick_params(labelsize=8)

    # ── информационная строка 
    w1w2 = w1 * (1 - w1)
    order_str = '2-й порядок' if abs(w1w2 - 0.5) < 1e-10 else '1-й порядок'
    info_line1.set_text(
        f'w₁ = {re_w1:.3f}{im_w1:+.3f}i  |  '
        f'w₁·w₂ = {w1w2.real:.3f}{w1w2.imag:+.3f}i  |  '
        f'{order_str}  |  '
        f'Δt = {dt:.3f}'
    )
    info_line2.set_text(stab_info)

    fig.canvas.draw_idle()



#  ОБРАБОТЧИКИ СОБЫТИЙ


def clamp(val, key):
    lo, hi = ranges[key]
    return max(lo, min(hi, val))


def on_slider(key):
    def handler(val):
        global _updating
        if _updating:
            return
        _updating = True
        params[key] = val
        textboxes[key].set_val(f'{val:.3f}')
        _updating = False
        _cache['traj'] = None  # сбрасываем кэш
        draw()
    return handler


def on_textbox(key):
    def handler(text):
        global _updating
        if _updating:
            return
        try:
            val = float(text)
            val = clamp(val, key)
            _updating = True
            params[key] = val
            sliders[key].set_val(val)
            _updating = False
            _cache['traj'] = None
            draw()
        except ValueError:
            pass
    return handler


def on_save(event):
    fname = (f'nose_hoover_'
             f're{params["re"]:.2f}_'
             f'im{params["im"]:.2f}_'
             f'dt{params["dt"]:.3f}_'
             f'{datetime.now().strftime("%H%M%S")}.png')
    fig.savefig(fname, dpi=150,
                bbox_inches='tight', facecolor=C_BG)
    print(f'Сохранено: {fname}')


# подключаем обработчики
for key in params:
    sliders[key].on_changed(on_slider(key))
    textboxes[key].on_submit(on_textbox(key))

btn_save.on_clicked(on_save)

fig.text(0.07, 0.245,
         'Канонический: Re=0.5, Im=0.5  |  '
         'Вещественный: Re=1.0, Im=0.0  |  '
         'Произвольный: Re=0.3, Im=0.7',
         color='#666', fontsize=8)

draw()
plt.show()
