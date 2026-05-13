% ============================================================================
% complex_euler_mathscript.m
% ----------------------------------------------------------------------------
%   КОД ДЛЯ MathScript Node В LabVIEW
%
%   Назначение: сравнение классического и КОМПЛЕКСНОГО 2-шагового метода
%   Эйлера (раздел 2 статьи George, Jung, Mangan — arXiv:2110.04402).
%
%   Этот скрипт целиком помещается в один MathScript Node:
%       Functions Palette → Mathematics → Scripts & Formulas → MathScript Node
%
%   Входные элементы управления (controls):
%       eq_idx  — выбор уравнения (целое число 1..5)
%                 1 :  dy/dt = y,                y(0) = 1,   t∈[0, 3]
%                 2 :  dy/dt = y^2,              y(0) = 1,   t∈[0, 0.5]
%                 3 :  dy/dt = 4 y sin(t)^3 cos(t),  y(0)=1, t∈[0, 5]
%                 4 :  dy/dt = cos(t),           y(0) = 0,   t∈[0, 2*pi]
%                 5 :  d²y/dt² = -y,             y(0)=1, y'(0)=0, t∈[0, 4π]
%
%       n_macro — число «макрошагов» комплексного метода (slider 2..200)
%
%   Выходные индикаторы (indicators) — массивы для XY-графиков:
%       t_real, y_real   — классический Эйлер
%       t_cmpl, y_cmpl   — комплексный 2-шаговый Эйлер
%       t_exact, y_exact — точное решение (или эталон)
%       err_real, err_cmpl — норма погрешности ||y_num - y_exact||_inf
% ============================================================================

% --- 0. Параметры из контролов (значения по умолчанию для отладки) ----------
% В реальном LabVIEW eq_idx и n_macro приходят с фронт-панели; для запуска
% MathScript «вне LabVIEW» можно раскомментировать строки ниже:
% eq_idx  = 1;
% n_macro = 10;

% --- 1. Параметры выбранного уравнения --------------------------------------
switch eq_idx
    case 1   % dy/dt = y
        f_rhs   = @(t,y) y;
        y0      = 1;
        t0      = 0;
        tN      = 3;
        is_2nd  = false;             % уравнение 1-го порядка
        exact_fn= @(t) exp(t);
        eq_name = 'dy/dt = y';

    case 2   % dy/dt = y^2
        f_rhs   = @(t,y) y.^2;
        y0      = 1;
        t0      = 0;
        tN      = 0.5;
        is_2nd  = false;
        exact_fn= @(t) 1./(1 - t);
        eq_name = 'dy/dt = y^2';

    case 3   % dy/dt = 4 y sin(t)^3 cos(t)
        f_rhs   = @(t,y) 4 .* y .* sin(t).^3 .* cos(t);
        y0      = 1;
        t0      = 0;
        tN      = 5;
        is_2nd  = false;
        exact_fn= @(t) exp(sin(t).^4);
        eq_name = 'dy/dt = 4 y sin^3(t) cos(t)';

    case 4   % dy/dt = cos(t)
        f_rhs   = @(t,y) cos(t);
        y0      = 0;
        t0      = 0;
        tN      = 2*pi;
        is_2nd  = false;
        exact_fn= @(t) sin(t);
        eq_name = 'dy/dt = cos(t)';

    case 5   % d²y/dt² = -y  (сводим к системе Y = [y; y'])
        f_rhs   = @(t,Y) [Y(2); -Y(1)];
        y0      = [1; 0];
        t0      = 0;
        tN      = 4*pi;
        is_2nd  = true;
        exact_fn= @(t) cos(t);       % сравнение по первой компоненте
        eq_name = 'd²y/dt² = -y';

    otherwise
        error('eq_idx must be in 1..5');
end

% --- 2. Коэффициенты комплексных шагов (раздел 2 статьи) --------------------
w1 = 0.5 + 0.5i;    % первый комплексный шаг:  Δt/2 + i Δt/2
w2 = 0.5 - 0.5i;    % второй (сопряжённый):    Δt/2 − i Δt/2
% Условие 2-го порядка:  w1 + w2 = 1  и  w1*w2 = 1/2  (проверяется ниже)

% --- 3. Сетки -----------------------------------------------------------------
% Комплексный 2-шаговый: n_macro макрошагов, каждый из двух подшагов = 2 f
% Классический: 2*n_macro шагов — то же число обращений к f, честное сравнение
dt  = (tN - t0) / n_macro;
N_r = 2 * n_macro;            % число шагов для классического
dt_r = (tN - t0) / N_r;

t_cmpl = linspace(t0, tN, n_macro + 1);
t_real = linspace(t0, tN, N_r + 1);

% --- 4. Решение классическим Эйлером -----------------------------------------
if is_2nd
    y_real_full = zeros(2, N_r + 1);
    y_real_full(:,1) = y0(:);
    for k = 1:N_r
        y_real_full(:,k+1) = y_real_full(:,k) + dt_r * f_rhs(t_real(k), y_real_full(:,k));
    end
    y_real = y_real_full(1, :);   % для графика — только сама y
else
    y_real = zeros(1, N_r + 1);
    y_real(1) = y0;
    for k = 1:N_r
        y_real(k+1) = y_real(k) + dt_r * f_rhs(t_real(k), y_real(k));
    end
end

% --- 5. Решение комплексным 2-шаговым Эйлером --------------------------------
if is_2nd
    y_c_full = zeros(2, n_macro + 1);
    y_c_full(:,1) = y0(:);
    yc = complex(y0(:));
    for k = 1:n_macro
        tk     = t_cmpl(k);
        y_star = yc + w1 * dt * f_rhs(tk, yc);
        t_star = tk + w1 * dt;
        yc_new = y_star + w2 * dt * f_rhs(t_star, y_star);
        yc     = real(yc_new);                 % возврат на действительную ось
        y_c_full(:,k+1) = yc;
    end
    y_cmpl = y_c_full(1, :);
else
    y_cmpl = zeros(1, n_macro + 1);
    y_cmpl(1) = y0;
    yc = complex(y0);
    for k = 1:n_macro
        tk     = t_cmpl(k);
        y_star = yc + w1 * dt * f_rhs(tk, yc);
        t_star = tk + w1 * dt;
        yc_new = y_star + w2 * dt * f_rhs(t_star, y_star);
        yc     = real(yc_new);
        y_cmpl(k+1) = yc;
    end
end

% --- 6. Точное решение (густая сетка для линии «exact») ----------------------
t_exact = linspace(t0, tN, 400);
y_exact = exact_fn(t_exact);

% --- 7. Норма погрешности (на сетках численных методов) ----------------------
y_exact_on_real = exact_fn(t_real);
y_exact_on_cmpl = exact_fn(t_cmpl);
err_real = max(abs(y_real - y_exact_on_real));
err_cmpl = max(abs(y_cmpl - y_exact_on_cmpl));

% --- 8. Печать сводки (видна в окне Output блок-диаграммы) -------------------
fprintf('Уравнение: %s\n', eq_name);
fprintf('n_macro = %d (комплексный),  N_real = %d (классический)\n', n_macro, N_r);
fprintf('dt (макро) = %.4g\n', dt);
fprintf('||err||_inf  classic   = %.3e\n', err_real);
fprintf('||err||_inf  complex2  = %.3e\n', err_cmpl);
if err_cmpl > 0
    fprintf('Улучшение точности: x %.2f\n', err_real / err_cmpl);
end
