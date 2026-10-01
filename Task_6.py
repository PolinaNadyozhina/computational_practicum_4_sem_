# -*- coding: utf-8 -*-
"""
Задание 6. Построение квадратурной формулы наивысшей алгебраической степени точности
(КФНАСТ, гауссова типа) с N узлами для веса rho(x) на (a, b) и сравнение с ИКФ.

Алгоритм:
 1) моменты mu_k = int_a^b x^k rho(x) dx, k = 0..2N-1  (scipy.integrate.quad);
 2) ортогональный многочлен omega_N(x) = x^N + a_{N-1} x^{N-1} + ... + a_0:
    СЛАУ (6):  sum_{j=0}^{N-1} a_j mu_{j+k} = -mu_{N+k},  k = 0..N-1  (матрица Ганкеля);
    решается numpy.linalg.solve;
 3) узлы КФНАСТ - корни omega_N на (a, b): отделение корней табулированием +
    уточнение методом бисекции (Задание 1);
 4) коэффициенты A_k из условия точности на 1, x, ..., x^(N-1)  (СЛАУ (2) с матрицей Вандермонда);
 5) контроль: КФНАСТ точна на многочленах степени <= 2N-1;
 6) приближённое значение интеграла sum A_k f(x_k); сравнение с "точным" (quad)
    и с ИКФ с N равноотстоящими узлами (Задание 4.1).

Ввод с клавиатуры (Enter - значение по умолчанию). Тип чисел - float64.
Источник данных: вариант из списка (вес 1-15) или свои данные: вес rho(x) (выражение),
a, b, число узлов N, функция f(x) (пункт 6 списка функций, доступен и для вариантов).
Эталон интеграла - quad или точное значение, введённое пользователем.
"""
import math
import warnings
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")      # подавление предупреждений quad

# ---------------------------------------------------------------------------
# Варианты весов (как в Задании 4.1): номер -> (запись, rho, область определения, особые точки)
# ---------------------------------------------------------------------------
VARIANTS = {
    1: ("sqrt(x)", lambda x: math.sqrt(x), (0, math.inf), []),
    2: ("x^(1/4)", lambda x: x ** 0.25, (0, math.inf), []),
    3: ("1/sqrt(x)", lambda x: x ** -0.5, (0, math.inf), []),
    4: ("x^(-1/4)", lambda x: x ** -0.25, (0, math.inf), []),
    5: ("-ln(x)", lambda x: -math.log(x), (0, math.inf), []),
    6: ("-x ln(x)", lambda x: -x * math.log(x), (0, math.inf), []),
    7: ("|x-0.5|", lambda x: abs(x - 0.5), (-math.inf, math.inf), [0.5]),
    8: ("exp(x)", lambda x: math.exp(x), (-math.inf, math.inf), []),
    9: ("1/(x+0.1)", lambda x: 1 / (x + 0.1), (-0.1, math.inf), []),
    10: ("sqrt(1-x)", lambda x: math.sqrt(1 - x), (-math.inf, 1), []),
    11: ("cos(x)", lambda x: math.cos(x), (-math.inf, math.inf), []),
    12: ("sin(2x)", lambda x: math.sin(2 * x), (-math.inf, math.inf), []),
    13: ("exp(-x)", lambda x: math.exp(-x), (-math.inf, math.inf), []),
    14: ("cos^2(x)", lambda x: math.cos(x) ** 2, (-math.inf, math.inf), []),
    15: ("sqrt(x/(1-x))", lambda x: math.sqrt(x / (1 - x)), (0, 1), []),
}


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, использовано значение по умолчанию.")
        return default


# ---------------------------------------------------------------------------
# Пользовательские функции: ввод выражением (как в Заданиях 2, 3)
# ---------------------------------------------------------------------------
ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
ENV["abs"] = abs


def make_function(expr):
    code = compile(expr, "<f>", "eval")

    def f(x):
        v = eval(code, {"__builtins__": {}}, dict(ENV, x=x))
        if isinstance(v, complex):
            raise ValueError("комплексное значение (аргумент вне области определения)")
        return float(v)
    return f


def read_function(label):
    """Ввод выражения от x (степень ** или ^; имена из math: sin, cos, exp, log, sqrt, pi, ...)."""
    print(f"Введите {label}, например: sqrt(x)*exp(-x)  или  1 + x^2  (умножение - знак *)")
    while True:
        expr = input(f"{label} = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr)
            f(0.5)
        except (ValueError, ZeroDivisionError, OverflowError):
            return expr, f               # пробная точка 0.5 вне области определения
        except Exception as e:           # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторите.")
            continue
        return expr, f


def read_points(a_hint=""):
    s = input("Особые точки веса внутри (a, b) через пробел, где он негладкий, напр. 0.5 для |x-0.5| [нет]: ")
    try:
        return [float(t) for t in s.replace(",", ".").split()]
    except ValueError:
        print("  Некорректный ввод, особых точек нет.")
        return []


def check_weight(rho, a, b, samples=400):
    """Проверка rho > 0 в точках внутри (a, b). Возвращает текст ошибки или None."""
    h = (b - a) / samples
    for i in range(samples):
        x = a + (i + 0.5) * h
        try:
            v = rho(x)
        except Exception as e:
            return f"вес не вычисляется в x = {x:.6g} ({e})"
        if not math.isfinite(v):
            return f"вес не является конечным числом в x = {x:.6g}"
        if v <= 0:
            return f"вес должен быть положительным на (a, b), а rho({x:.6g}) = {v:.6g}"
    return None


def integrate(g, a, b, pts):
    """'Точное' значение интеграла матпакетом (scipy.quad)."""
    inner = [p for p in pts if a < p < b]
    val, err = quad(g, a, b, points=inner or None, epsabs=1e-14, epsrel=1e-14, limit=500)
    return val, err


def moments(rho, a, b, count, pts, check=False):
    """mu_k = int x^k rho(x) dx, k = 0..count-1. При check=True RuntimeError, если интеграл не вычислен надёжно."""
    out = []
    for j in range(count):
        val, err = integrate(lambda x, j=j: x ** j * rho(x), a, b, pts)
        if check:
            if not math.isfinite(val):
                raise RuntimeError(f"момент mu_{j} не вычислен (расходящийся интеграл или особенность веса)")
            if not err <= 1e-6 * max(1.0, abs(val)):
                raise RuntimeError(f"момент mu_{j} вычислен ненадёжно (оценка погрешности quad {err:.1e})")
        out.append(val)
    return np.array(out)


# ---------------------------------------------------------------------------
# Ортогональный многочлен и его корни (методы Задания 1)
# ---------------------------------------------------------------------------
def poly_val(coef, x):
    """omega_N(x) по схеме Горнера; coef = [a_0, ..., a_{N-1}, 1] (по возрастанию степеней)."""
    s = 0.0
    for c in reversed(coef):
        s = s * x + c
    return s


def separate_roots(w, a, b, parts):
    """Табулирование: отрезки перемены знака w на [a, b] (как в Задании 1)."""
    h = (b - a) / parts
    segs = []
    x1, y1 = a, w(a)
    for i in range(1, parts + 1):
        x2 = a + i * h
        y2 = w(x2)
        if y1 * y2 < 0 or y2 == 0:
            segs.append((x1, x2))
        x1, y1 = x2, y2
    return segs


def bisection(w, a, b, eps):
    """Метод половинного деления (блок-схема Задания 1). Число итераций ограничено."""
    for _ in range(200):
        if b - a <= 2 * eps:
            break
        c = (a + b) / 2
        if c == a or c == b:               # предел точности чисел с плавающей точкой
            break
        if w(a) * w(c) <= 0:
            b = c
        else:
            a = c
    return (a + b) / 2


def find_roots(coef, a, b, N, eps):
    """N корней omega_N на (a, b): число частей табулирования удваивается до получения N отрезков."""
    w = lambda x: poly_val(coef, x)
    parts = max(1000, 200 * N)
    while parts <= 4_000_000:
        segs = separate_roots(w, a, b, parts)
        if len(segs) == N:
            return np.array([bisection(w, s[0], s[1], eps) for s in segs]), parts
        parts *= 2
    raise RuntimeError(f"не удалось отделить {N} корней ортогонального многочлена")


# ---------------------------------------------------------------------------
# Коэффициенты по заданным узлам (ИКФ, как в Задании 4.1)
# ---------------------------------------------------------------------------
def coeffs_for_nodes(nodes, mu):
    N = len(nodes)
    V = np.vander(nodes, N, increasing=True).T      # V[j, k] = x_k ** j
    return np.linalg.solve(V, mu[:N]), V


def make_functions(N, own=False):
    """f1 - многочлен степени 2N-1; f2 - sin(x) (из варианта); f3..f5 - фиксированные функции; f6 - пользовательская."""
    deg = 2 * N - 1
    coef = [(-1) ** k * (k + 1) for k in range(deg + 1)]
    return {
        1: (f"f1 = многочлен степени 2N-1 = {deg}  (коэф. 1,-2,3,...)",
            lambda x: sum(c * x ** k for k, c in enumerate(coef))),
        2: ("f2 = sin(x)" + ("" if own else "  (из варианта)"), lambda x: math.sin(x)),
        3: ("f3 = exp(x)", lambda x: math.exp(x)),
        4: ("f4 = 1/(1+x^2)", lambda x: 1 / (1 + x * x)),
        5: (f"f5 = x^{2*N}  (степень 2N: КФНАСТ уже не точна)", lambda x: x ** (2 * N)),
        6: ("f6 = своя функция (вводится выражением)", None),
    }


def choose_ikf_nodes(a, b, N):
    if N == 1:
        return np.array([(a + b) / 2])
    return np.linspace(a, b, N)




def run(own):
    """Сеанс работы с выбранным источником. Возвращает 'q' (выход) или 'm' (в меню)."""
    if own:
        print("Свои данные: вес rho(x) > 0 на (a, b) вводится выражением от x.")
        wexpr, rho = read_function("rho(x)")
        pts = read_points()
        name, lo, hi = wexpr, -math.inf, math.inf
        print(f"Вес: rho(x) = {wexpr}")
    else:
        v = ask("Номер варианта (1-15)", 1, int)
        while v not in VARIANTS:
            v = ask("Нет такого варианта. Номер (1-15)", 1, int)
        name, rho, (lo, hi), pts = VARIANTS[v]
        print(f"Вариант {v}: rho(x) = {name};  f2(x) = sin(x)")
    user_f = None                       # (выражение, функция) - пользовательская f(x), вводится при первом выборе пункта 6

    while True:
        a = ask("a", 0.0, float)
        b = ask("b", 1.0, float)
        while not (a < b and math.isfinite(a) and lo <= a and b <= hi and (not own or math.isfinite(b))):
            print(f"Требуется конечное a < b" + ("" if own else f" и [a,b] внутри области определения веса [{lo}; {hi}]") + ".")
            a = ask("a", 0.0, float)
            b = ask("b", 1.0, float)
        if own:
            err = check_weight(rho, a, b)
            if err:
                print("Ошибка:", err)
                if input("Другой вес (w) / другие a, b (Enter): ").strip().lower() == "w":
                    wexpr, rho = read_function("rho(x)")
                    pts = read_points()
                    name = wexpr
                continue
        N = ask("Число узлов N (>=1)", 3, int)
        while N < 1:
            N = ask("Требуется N >= 1", 3, int)
        if N > 40:
            print("N > 40 не допускается: матрица Ганкеля вырождается в double.")
            continue
        eps = ask("Точность eps для корней (метод бисекции)", 1e-14, float)
        if not eps > 0:
            print("  eps должно быть > 0, использовано 1e-14.")
            eps = 1e-14

        # --- 1. моменты mu_0 .. mu_{2N-1} (и mu_{2N} для проверки)
        try:
            mu_ext = moments(rho, a, b, 2 * N + 1, pts, check=own)
        except RuntimeError as e:
            print("Ошибка:", e)
            continue
        except Exception as e:
            print(f"Ошибка при вычислении моментов: {e}.")
            continue
        mu = mu_ext[:2 * N]
        if not np.all(np.isfinite(mu_ext)):
            print("Ошибка: моменты веса бесконечны или не определены (вес не интегрируем на (a, b)).")
            continue
        print("\nМоменты mu_k = int x^k rho(x) dx, k = 0..2N-1:")
        for k in range(2 * N):
            print(f"  mu_{k} = {mu[k]:.15f}")

        # --- 2. ортогональный многочлен
        H = np.array([[mu[j + k] for j in range(N)] for k in range(N)])
        rhs = -np.array([mu[N + k] for k in range(N)])
        print(f"\nСЛАУ (6) для коэффициентов omega_N: матрица Ганкеля {N}x{N}, cond = {np.linalg.cond(H):.3e}")
        try:
            acoef = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            print("Ошибка: матрица Ганкеля вырождена.")
            continue
        coef = list(acoef) + [1.0]
        print("Ортогональный многочлен omega_N(x) = x^N + a_{N-1} x^{N-1} + ... + a_0:")
        for j in range(N):
            print(f"  a_{j} = {acoef[j]:.15e}")
        resid = float(np.max(np.abs(H @ acoef - rhs)))
        print(f"  невязка СЛАУ: {resid:.3e}")

        # --- 3. узлы
        try:
            nodes, parts = find_roots(coef, a, b, N, eps)
        except RuntimeError as e:
            print("Ошибка:", e)
            print("(большое N: матрица Ганкеля плохо обусловлена, в double ортогональный многочлен"
                  " теряет точность)")
            continue
        print(f"\nУзлы КФНАСТ (корни omega_N на ({a}; {b}); отделение: {parts} частей, бисекция eps = {eps:g}):")
        for k in range(N):
            print(f"  x_{k+1} = {nodes[k]:.15f}   |omega_N(x_k)| = {abs(poly_val(coef, nodes[k])):.2e}")
        if N > 1 and np.min(np.diff(nodes)) <= 0:
            print("  !!! узлы не попарно различны")

        # --- 4. коэффициенты
        try:
            A, V = coeffs_for_nodes(nodes, mu)
        except np.linalg.LinAlgError:
            print("Ошибка: матрица Вандермонда вырождена (узлы совпали).")
            continue
        print(f"\nМатрица Вандермонда {N}x{N}, cond = {np.linalg.cond(V):.3e}")
        print(f"{'k':>3}{'x_k':>22}{'A_k':>24}")
        for k in range(N):
            print(f"{k+1:>3}{nodes[k]:>22.15f}{A[k]:>24.15f}")
        print(f"Контроль: sum A_k = {A.sum():.15f},  mu_0 = {mu[0]:.15f};  все A_k > 0: {bool(np.all(A > 0))}")

        # --- 5. контроль точности на многочленах степени <= 2N-1
        print("\nКонтроль точности КФНАСТ на одночленах x^j, j = 0..2N-1 (и для сравнения j = 2N):")
        print(f"{'j':>3}{'int rho x^j':>22}{'sum A_k x_k^j':>22}{'|погрешность|':>16}")
        worst = 0.0
        for j in range(2 * N + 1):
            appr = float(np.sum(A * nodes ** j))
            err = abs(mu_ext[j] - appr)
            if j <= 2 * N - 1:
                worst = max(worst, err)
            print(f"{j:>3}{mu_ext[j]:>22.15f}{appr:>22.15f}{err:>16.3e}" + ("   <- степень 2N, не точна" if j == 2 * N else ""))
        print(f"max погрешность при j <= 2N-1: {worst:.3e}")

        # --- ИКФ с N равноотстоящими узлами
        xi = choose_ikf_nodes(a, b, N)
        try:
            Ai, _ = coeffs_for_nodes(xi, mu)
        except np.linalg.LinAlgError:
            print("Ошибка: матрица Вандермонда ИКФ вырождена.")
            continue
        print(f"\nИКФ с N = {N} равноотстоящими узлами (из Задания 4.1):")
        print(f"{'k':>3}{'x_k':>22}{'A_k':>24}")
        for k in range(N):
            print(f"{k+1:>3}{xi[k]:>22.15f}{Ai[k]:>24.15f}")

        funcs = make_functions(N, own)
        while True:
            print("\nФункции:")
            for k, (nm, _) in funcs.items():
                print(f"  {k}) {nm}" + (f":  {user_f[0]}" if k == 6 and user_f else ""))
            n = ask("Номер функции", 2, int)
            while n not in funcs:
                n = ask("Нет такой функции. Номер", 2, int)
            exact_user = None
            if n == 6:
                if user_f is None or input("Ввести f(x) заново (y) / оставить прежнюю (Enter): ").strip().lower() == "y":
                    user_f = read_function("f(x)")
                f, label = user_f[1], f"f6 = {user_f[0]}"
                s = input("Точное значение интеграла (число или выражение, напр. pi/4; Enter - вычислить quad): ").strip()
                if s:
                    try:
                        exact_user = float(eval(s.replace("^", "**").replace(",", "."), {"__builtins__": {}}, dict(ENV)))
                        if not math.isfinite(exact_user):
                            raise ValueError("не конечное число")
                    except Exception as e:
                        print(f"  Не удалось разобрать значение ({e}); эталон - quad.")
                        exact_user = None
            else:
                f, label = funcs[n][1], funcs[n][0]
            try:
                if exact_user is None:
                    exact, qerr = integrate(lambda x: rho(x) * f(x), a, b, pts)
                    if not math.isfinite(exact):
                        raise ValueError("интеграл rho*f не вычислен (расходится или f не определена на (a, b))")
                g = float(sum(A[k] * f(nodes[k]) for k in range(N)))
                i = float(sum(Ai[k] * f(xi[k]) for k in range(N)))
            except Exception as e:
                print(f"Ошибка при вычислении интеграла: {e}.")
                continue
            if not (math.isfinite(g) and math.isfinite(i)):
                print("Ошибка: значение f в узлах не является конечным числом.")
                continue
            print(f"\nИнтеграл int_a^b rho(x) f(x) dx,  {label}")
            if exact_user is not None:
                exact = exact_user
                print(f"  заданное пользователем = {exact:.15f}")
            elif own:
                print(f"  эталон (quad)          = {exact:.15f}   (оценка quad: {qerr:.1e})")
            else:
                print(f"  'точное' (quad)        = {exact:.15f}   (оценка quad: {qerr:.1e})")
            print(f"{'':<12}{'приближённое':>24}{'абс. погрешность':>20}{'отн. погрешность':>20}")
            for lab, val in (("КФНАСТ", g), ("ИКФ", i)):
                ae = abs(exact - val)
                re = ae / abs(exact) if exact != 0 else float("nan")
                print(f"{lab:<12}{val:>24.15f}{ae:>20.3e}{re:>20.3e}")
            ae_g, ae_i = abs(exact - g), abs(exact - i)
            if ae_g > 0 and ae_i > 0:
                print(f"  ИКФ хуже КФНАСТ в {ae_i / ae_g:.3g} раз(а)")
            c = input("\nДругая функция (Enter) / новые параметры (p) / в меню (m) / выход (q): ").strip().lower()
            if c in ("p", "q", "m"):
                break
        if c in ("q", "m"):
            return c


def main():
    print("=" * 78)
    print("ЗАДАНИЕ 6. КВАДРАТУРНАЯ ФОРМУЛА НАИВЫСШЕЙ АСТ (КФНАСТ) И СРАВНЕНИЕ С ИКФ")
    print("=" * 78)
    while True:
        print("\nИСТОЧНИК ДАННЫХ")
        print("  1 - вариант из списка (вес 1-15, f2 = sin x)")
        print("  2 - свои данные: вес rho(x), a, b, число узлов N, своя f(x)")
        print("  q - выход")
        c = input("Выбор [1]: ").strip().lower() or "1"
        if c == "q":
            break
        if c not in ("1", "2"):
            print("Допустимо: 1, 2, q.")
            continue
        if run(c == "2") == "q":
            break


if __name__ == "__main__":
    main()
