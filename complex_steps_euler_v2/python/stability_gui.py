"""
=============================================================
  ИНТЕРАКТИВНЫЙ GUI: Устойчивость методов Эйлера
=============================================================
Запуск:  python3 stability_gui.py
Нужно:   pip install matplotlib numpy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, TextBox
from matplotlib.gridspec import GridSpec

# ── цвета (светлая тема) ────────────────────────────────────
C_CLASSIC = '#2B4EE6'   # синий
C_COMPLEX  = '#0F9E5E'  # зелёный
C_EXACT    = '#C47A00'  # оранжевый
C_POINT    = '#CC2222'  # красный
C_BG       = '#FFFFFF'  # белый фон
C_PANEL    = '#F0F2F5'  # светло-серая панель
C_TEXT     = '#1A1A2E'  # тёмный текст
C_GRID     = '#CCCCCC'  # сетка
C_BORDER   = '#AAAAAA'  # рамки

# ── параметры ───────────────────────────────────────────────
params    = {'re': -1.0, 'im': 0.0, 'dt': 0.5, 'T': 10.0}
ranges    = {'re': (-5.0, 2.0), 'im': (-3.0, 3.0),
             'dt': (0.05, 3.0),  'T':  (2.0, 30.0)}
labels    = {'re': 'Re(λ)', 'im': 'Im(λ)', 'dt': 'Δt', 'T': 'T (конец)'}
colors_sl = {'re': C_CLASSIC, 'im': C_COMPLEX, 'dt': C_POINT, 'T': '#888888'}

_updating = False

# ── функции устойчивости ────────────────────────────────────
def phi_classic(z): return 1.0 + z
def phi_complex(z): return 1.0 + z + 0.5 * z**2

# ── области устойчивости (один раз) ─────────────────────────
def make_region(phi_func, re_range=(-4,2), im_range=(-3,3), N=400):
    re = np.linspace(*re_range, N)
    im = np.linspace(*im_range, N)
    RE, IM = np.meshgrid(re, im)
    return RE, IM, np.abs(phi_func(RE + 1j*IM))

RE_c, IM_c, ABS_c = make_region(phi_classic)
RE_x, IM_x, ABS_x = make_region(phi_complex)

# ── методы ──────────────────────────────────────────────────
def euler_classic(lam, y0, T, N):
    dt = T / N
    t  = np.linspace(0, T, N+1)
    y  = np.zeros(N+1, dtype=complex)
    y[0] = y0
    for k in range(N):
        y[k+1] = y[k] + dt * lam * y[k]
    return t, y.real

def euler_complex(lam, y0, T, N_macro):
    dt = T / N_macro
    t  = np.linspace(0, T, N_macro+1)
    y  = np.zeros(N_macro+1, dtype=complex)
    y[0] = y0
    w1, w2 = 0.5+0.5j, 0.5-0.5j
    for k in range(N_macro):
        yk     = y[k]
        y_star = yk + w1*dt*lam*yk
        y[k+1] = (y_star + w2*dt*lam*y_star).real
    return t, y.real

def exact_sol(lam, y0, T, N=500):
    t = np.linspace(0, T, N)
    return t, (y0 * np.exp(lam*t)).real

# ══════════════════════════════════════════════════════════
#  ФИГУРА
# ══════════════════════════════════════════════════════════
plt.style.use('default')
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
    'legend.facecolor':  C_PANEL,
    'legend.edgecolor':  C_BORDER,
})

fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
fig.canvas.manager.set_window_title('Устойчивость методов Эйлера')

gs = GridSpec(3, 2, figure=fig,
              left=0.06, right=0.97,
              top=0.93,  bottom=0.28,
              wspace=0.35, hspace=0.5)

ax_stab = fig.add_subplot(gs[:, 0])
ax_sol  = fig.add_subplot(gs[0:2, 1])
ax_info = fig.add_subplot(gs[2, 1])
ax_info.axis('off')

# ── ползунки и поля ввода ───────────────────────────────────
SL_LEFT = 0.08;  SL_W = 0.28
TB_LEFT = 0.38;  TB_W = 0.07
ROW_H   = 0.022
rows = {'re': 0.19, 'im': 0.14, 'dt': 0.09, 'T': 0.04}

sliders   = {}
textboxes = {}

for key, ypos in rows.items():
    ax_sl = fig.add_axes([SL_LEFT, ypos, SL_W, ROW_H], facecolor=C_PANEL)
    sl = Slider(ax_sl, labels[key],
                ranges[key][0], ranges[key][1],
                valinit=params[key], color=colors_sl[key])
    sl.label.set_color(C_TEXT)
    sl.valtext.set_color(C_TEXT)
    sliders[key] = sl

    ax_tb = fig.add_axes([TB_LEFT, ypos, TB_W, ROW_H], facecolor=C_PANEL)
    tb = TextBox(ax_tb, '', initial=f'{params[key]:.3f}',
                 color=C_PANEL, hovercolor='#E0E4EA', label_pad=0.0)
    tb.text_disp.set_color(C_TEXT)
    textboxes[key] = tb

fig.text(TB_LEFT + TB_W/2, 0.245,
         'Enter для применения', color='#888', fontsize=8, ha='center')

# ══════════════════════════════════════════════════════════
#  ОТРИСОВКА
# ══════════════════════════════════════════════════════════
def draw():
    re_lam = params['re']
    im_lam = params['im']
    dt     = max(params['dt'], 0.01)
    T      = params['T']
    lam    = complex(re_lam, im_lam)
    z      = lam * dt
    N_macro   = max(int(T / dt), 2)
    N_classic = 2 * N_macro

    # ── левая: области устойчивости ───────────────────────
    ax_stab.clear()
    ax_stab.set_facecolor(C_BG)

    ax_stab.contourf(RE_c, IM_c, ABS_c, levels=[0,1],
                     colors=[C_CLASSIC], alpha=0.15)
    ax_stab.contourf(RE_x, IM_x, ABS_x, levels=[0,1],
                     colors=[C_COMPLEX],  alpha=0.12)
    ax_stab.contour(RE_c, IM_c, ABS_c, levels=[1],
                    colors=[C_CLASSIC], linewidths=2.5)
    ax_stab.contour(RE_x, IM_x, ABS_x, levels=[1],
                    colors=[C_COMPLEX],  linewidths=2.5)

    ax_stab.axhline(0, color=C_BORDER, lw=1.0)
    ax_stab.axvline(0, color=C_BORDER, lw=1.0)
    ax_stab.grid(True, color=C_GRID, alpha=0.6, lw=0.5)

    # точка z
    ax_stab.plot(z.real, z.imag, 'o', color=C_POINT,
                 ms=14, zorder=10,
                 markeredgecolor='#660000', markeredgewidth=1.5)
    ox = 0.2 if z.real < 1.5 else -0.9
    oy = 0.2 if z.imag < 2.5 else -0.35
    ax_stab.annotate(f'z = {z.real:.2f}{z.imag:+.2f}i',
                     xy=(z.real, z.imag),
                     xytext=(z.real+ox, z.imag+oy),
                     color=C_POINT, fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=C_POINT, lw=1.5))

    p1 = mpatches.Patch(color=C_CLASSIC, alpha=0.7,
                        label='Классический: |1+z| ≤ 1')
    p2 = mpatches.Patch(color=C_COMPLEX,  alpha=0.7,
                        label='Комплексный:  |1+z+z²/2| ≤ 1')
    leg = ax_stab.legend(handles=[p1,p2], fontsize=9,
                         facecolor=C_PANEL, edgecolor=C_BORDER,
                         loc='upper right')
    for text in leg.get_texts():
        text.set_color(C_TEXT)

    ax_stab.set_xlim(-4.5, 2.5)
    ax_stab.set_ylim(-3.2, 3.2)
    ax_stab.set_xlabel('Re(z)', fontsize=11)
    ax_stab.set_ylabel('Im(z)', fontsize=11)
    ax_stab.set_title('Области абсолютной устойчивости\n'
                      '(точка z = λ·Δt должна быть внутри)',
                      fontsize=12, pad=10, color=C_TEXT)

    # ── правая: решение ────────────────────────────────────
    ax_sol.clear()
    ax_sol.set_facecolor(C_BG)
    try:
        CLIP = 50
        t_ex, y_ex = exact_sol(lam, 1.0, T)
        t_cl, y_cl = euler_classic(lam, 1.0, T, N_classic)
        t_cx, y_cx = euler_complex(lam, 1.0, T, N_macro)

        ax_sol.plot(t_ex, np.clip(y_ex,-CLIP,CLIP), '-',
                    color=C_EXACT, lw=2.5,
                    label='Точное решение', zorder=3)
        ax_sol.plot(t_cl, np.clip(y_cl,-CLIP,CLIP), '--o',
                    color=C_CLASSIC, lw=1.5, ms=3, alpha=0.85,
                    label=f'Классический ({N_classic} шагов)', zorder=2)
        ax_sol.plot(t_cx, np.clip(y_cx,-CLIP,CLIP), '--s',
                    color=C_COMPLEX, lw=1.5, ms=3, alpha=0.85,
                    label=f'Комплексный ({N_macro} макрошагов)', zorder=2)

        leg2 = ax_sol.legend(fontsize=9, facecolor=C_PANEL,
                             edgecolor=C_BORDER)
        for text in leg2.get_texts():
            text.set_color(C_TEXT)
    except Exception as e:
        ax_sol.text(0.5, 0.5, f'Ошибка: {e}',
                    transform=ax_sol.transAxes,
                    color='red', ha='center')

    ax_sol.axhline(0, color=C_BORDER, lw=0.8)
    ax_sol.grid(True, color=C_GRID, alpha=0.6, lw=0.5)
    ax_sol.set_xlabel('t', fontsize=11)
    ax_sol.set_ylabel('y(t)', fontsize=11)
    ax_sol.set_title('Численное решение  ẏ = λy,  y(0) = 1',
                     fontsize=12, color=C_TEXT)

    # ── инфо панель ────────────────────────────────────────
    ax_info.clear()
    ax_info.axis('off')
    ax_info.set_facecolor(C_PANEL)

    abs_c = abs(phi_classic(z))
    abs_x = abs(phi_complex(z))
    v_c   = '[OK] УСТОЙЧИВ'   if abs_c <= 1 else '[!!] НЕУСТОЙЧИВ'
    v_x   = '[OK] УСТОЙЧИВ'   if abs_x <= 1 else '[!!] НЕУСТОЙЧИВ'
    col_c = C_COMPLEX if abs_c <= 1 else C_POINT
    col_x = C_COMPLEX if abs_x <= 1 else C_POINT

    # фон инфо-панели
    ax_info.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96,
        boxstyle='round,pad=0.01',
        facecolor=C_PANEL, edgecolor=C_BORDER,
        transform=ax_info.transAxes, zorder=0))

    ax_info.text(0.5, 0.80,
                 f'λ = {re_lam:.3f}{im_lam:+.3f}i     '
                 f'Δt = {dt:.3f}     '
                 f'z = {z.real:.3f}{z.imag:+.3f}i',
                 transform=ax_info.transAxes, color=C_TEXT,
                 fontsize=11, ha='center', va='center',
                 fontfamily='monospace')

    ax_info.text(0.25, 0.44,
                 f'Классический\n|Φ(z)| = {abs_c:.4f}\n{v_c}',
                 transform=ax_info.transAxes, color=col_c,
                 fontsize=11, ha='center', va='center',
                 fontfamily='monospace', fontweight='bold')

    ax_info.text(0.75, 0.44,
                 f'Комплексный\n|Φ(z)| = {abs_x:.4f}\n{v_x}',
                 transform=ax_info.transAxes, color=col_x,
                 fontsize=11, ha='center', va='center',
                 fontfamily='monospace', fontweight='bold')

    # разделитель
    ax_info.axvline(0.5, color=C_BORDER, lw=1, alpha=0.5)

    if abs_c <= 1 and abs_x <= 1:
        sc, sc_col = 'Сценарий A1: оба устойчивы', C_COMPLEX
    elif abs_c > 1 and abs_x <= 1:
        sc = 'Сценарий A2: классический взрывается, комплексный устойчив'
        sc_col = C_CLASSIC
    else:
        sc, sc_col = 'Сценарий A3: оба неустойчивы', C_POINT

    ax_info.text(0.5, 0.10, sc,
                 transform=ax_info.transAxes, color=sc_col,
                 fontsize=11, ha='center', va='center',
                 fontweight='bold')

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════════════════
#  ОБРАБОТЧИКИ
# ══════════════════════════════════════════════════════════
def on_slider(key):
    def handler(val):
        global _updating
        if _updating: return
        _updating = True
        params[key] = val
        textboxes[key].set_val(f'{val:.3f}')
        _updating = False
        draw()
    return handler

def on_textbox(key):
    def handler(text):
        global _updating
        if _updating: return
        try:
            val = float(text)
            lo, hi = ranges[key]
            val = max(lo, min(hi, val))
            _updating = True
            params[key] = val
            sliders[key].set_val(val)
            _updating = False
            draw()
        except ValueError:
            pass
    return handler

for key in params:
    sliders[key].on_changed(on_slider(key))
    textboxes[key].on_submit(on_textbox(key))

# ── заголовок ──────────────────────────────────────────────
fig.text(0.5, 0.97,
         'Устойчивость методов Эйлера  |  ползунок или введи значение + Enter',
         color=C_TEXT, fontsize=13,
         ha='center', va='top', fontweight='bold')

draw()
plt.show()
