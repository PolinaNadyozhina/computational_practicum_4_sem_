# -*- coding: utf-8 -*-
"""
Задание №1. Численные методы решения нелинейных уравнений f(x) = 0.
Этап 1: отделение корней табулированием [A, B] с шагом h = (B-A)/N.
Этап 2: уточнение корня на отрезке перемены знака методами бисекции,
        Ньютона, модифицированным Ньютона, секущих.
Дополнительно: задача о глубине погружения шара (ball_table).

Ввод: клавиатура (Enter - значение по умолчанию). Главное меню:
  1 - тестовая задача из таблицы (1-30);
  2 - функция f(x) вводится вручную;
  3 - задача о погружении шара (список материалов или свой материал).
Тип вычислений: float (64 бита).
"""
import math

# ---------------------------------------------------------------------------
# 1. ТЕСТОВЫЕ ЗАДАЧИ: номер -> (выражение на Python, A, B, eps)
# ---------------------------------------------------------------------------
TASKS = {
    1: ("x - 10*sin(x)", -5, 3, 1e-6),
    2: ("2**(-x) - sin(x)", -5, 10, 1e-6),
    3: ("2**x - 2*cos(x)", -8, 10, 1e-6),
    4: ("sqrt(4*x+7) - 3*cos(x)", -1.5, 2, 1e-8),
    5: ("x*sin(x) - 1", -10, 2, 1e-5),
    6: ("8*cos(x) - x - 6", -9, 1, 1e-7),
    7: ("10*cos(x) - 0.1*x**2", -8, 2, 1e-5),
    8: ("4*cos(x) + 0.3*x", -15, 5, 1e-5),
    9: ("5*sin(2*x) - sqrt(1-x)", -15, -10, 1e-6),
    10: ("1.2*x**4 + 2*x**3 - 13*x**2 - 14.2*x - 24.1", -5, 5, 1e-6),
    11: ("2*x**2 - 2*x - 5", -3, 7, 1e-9),
    12: ("2**(-x) + 0.5*x**2 - 10", -3, 5, 1e-8),
    13: ("sin(x) + x**3 - 9*x + 3", -5, 4, 1e-8),
    14: ("x - cos(pi*x)**2", -1, 2, 1e-8),
    15: ("(x-1)**2 - exp(-x)", -1, 3, 1e-8),
    16: ("sin(5*x) + x**2 - 1", -3, 3, 1e-8),
    17: ("cos(3*x) - x**3", -2, 1, 1e-8),
    18: ("x**2 - sin(5*x)", -2, 1, 1e-8),
    19: ("1.8*x**2 - sin(10*x)", -1, 1, 1e-6),
    20: ("sqrt(x) - 2*cos(pi*x/2)", 0, 4.5, 1e-8),
    21: ("x - 3*cos(1.04*x)**2", 0, 3.5, 1e-8),
    22: ("(x-3)*cos(x) - 1", -6.5, 6.5, 1e-8),
    23: ("exp(-x) + x**2 - 2", -10, 10, 1e-8),
    24: ("8*x**4 + 4*x**3 - 14*x**2 - x + 2", -5, 5, 1e-10),
    25: ("log(x) + (x-1)**3", 0.01, 5, 1e-10),
    26: ("x**2 - 20*sin(x)", -4.5, 5, 1e-6),
    27: ("0.001*x**5 + x**2 - 1", -15, 10, 1e-10),
    28: ("0.5**x - (x-1)**2 + 1", 0, 12.5, 1e-7),
    29: ("8*x**5 + 8*x**3 - x**2 - 9", -10, 5.5, 1e-10),
    30: ("24*x**5 + 8*x**3 - 3*x**2 - 9", -10, 8.5, 1e-9),
}

MAX_ITER = 200  # предельное число итераций


# ---------------------------------------------------------------------------
# 2. ФУНКЦИЯ И ПРОИЗВОДНАЯ
# ---------------------------------------------------------------------------
def make_function(expr):
    """Строит f(x) по строке; вне области определения f возвращает nan."""
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    code = compile(expr, "<f>", "eval")

    def f(x):
        try:
            return float(eval(code, {"__builtins__": {}}, dict(env, x=x)))
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")
    return f


def make_derivative(expr, f):
    """Производная: аналитическая (sympy), при его отсутствии численная."""
    try:
        import sympy as sp
        X = sp.Symbol("x")
        d = sp.diff(sp.sympify(expr.replace("log", "ln")), X)
        fd = sp.lambdify(X, d, "math")
        return (lambda x: float(fd(x))), "аналитическая (sympy)"
    except Exception:
        def fd(x):
            h = 1e-6 * max(1.0, abs(x))
            return (f(x + h) - f(x - h)) / (2 * h)
        return fd, "численная (центральная разность)"


# ---------------------------------------------------------------------------
# 3. ОТДЕЛЕНИЕ КОРНЕЙ
# ---------------------------------------------------------------------------
def separate_roots(f, A, B, N, verbose=True):
    """Табулирование на [A, B]; возвращает список отрезков [a_i, b_i] перемены знака.
    Узлы: A + i*h (без накопления суммы). Нулевое значение в узле учитывается один раз."""
    h = (B - A) / N
    segs = []
    x1, y1 = A, f(A)
    if y1 == 0:                      # корень в точке A
        segs.append((A, A + h))
    for i in range(1, N + 1):
        x2 = A + i * h
        y2 = f(x2)
        if y1 * y2 < 0 or y2 == 0:   # перемена знака или корень в правом конце
            segs.append((x1, x2))
        x1, y1 = x2, y2
    if verbose:
        print(f"Шаг h = {h:.3e}. Найдено отрезков перемены знака: {len(segs)}")
        for k, (a, b) in enumerate(segs, 1):
            print(f"  {k:3d}) [{a:.10f}; {b:.10f}]")
    return segs


# ---------------------------------------------------------------------------
# 4. МЕТОДЫ УТОЧНЕНИЯ. Результат - dict:
#    name, start, m, x, dx, res  (dx = |x_m - x_{m-1}| или длина посл. отрезка)
# ---------------------------------------------------------------------------
def bisection(f, a, b, eps):
    start = f"[{a:.8f}; {b:.8f}]"
    m = 0
    while b - a > 2 * eps:
        c = (a + b) / 2
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
        m += 1
    x = (a + b) / 2
    return dict(name="Бисекция", start=start, m=m, x=x, dx=b - a, res=abs(f(x)))


def newton(f, df, x0, eps):
    x_prev, x, m = x0, x0, 0
    while m < MAX_ITER:
        d = df(x_prev)
        if d == 0 or d != d:
            return dict(name="Ньютон", start=f"x0={x0:.8f}", m=m, x=x_prev,
                        dx=float("nan"), res=abs(f(x_prev)), fail="f'(x)=0")
        x = x_prev - f(x_prev) / d
        m += 1
        if abs(x - x_prev) < eps:
            return dict(name="Ньютон", start=f"x0={x0:.8f}", m=m, x=x,
                        dx=abs(x - x_prev), res=abs(f(x)))
        x_prev = x
    return dict(name="Ньютон", start=f"x0={x0:.8f}", m=m, x=x,
                dx=abs(x - x_prev), res=abs(f(x)), fail="не сошёлся")


def mod_newton(f, df, x0, eps):
    d0 = df(x0)                      # производная вычисляется один раз, в x0
    x_prev, x, m = x0, x0, 0
    if d0 == 0 or d0 != d0:
        return dict(name="Мод. Ньютон", start=f"x0={x0:.8f}", m=0, x=x0,
                    dx=float("nan"), res=abs(f(x0)), fail="f'(x0)=0")
    while m < MAX_ITER:
        x = x_prev - f(x_prev) / d0
        m += 1
        if abs(x - x_prev) < eps:
            return dict(name="Мод. Ньютон", start=f"x0={x0:.8f}", m=m, x=x,
                        dx=abs(x - x_prev), res=abs(f(x)))
        x_prev = x
    return dict(name="Мод. Ньютон", start=f"x0={x0:.8f}", m=m, x=x,
                dx=abs(x - x_prev), res=abs(f(x)), fail="не сошёлся")


def secant(f, x0, x1, eps):
    m = 0
    start = f"x0={x0:.8f}, x1={x1:.8f}"
    while m < MAX_ITER:
        f0, f1 = f(x0), f(x1)
        if f1 == f0:                 # деление на ноль
            return dict(name="Секущих", start=start, m=m, x=x1,
                        dx=abs(x1 - x0), res=abs(f1), fail="f(x_k)=f(x_{k-1})")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        m += 1
        if abs(x2 - x1) < eps:
            return dict(name="Секущих", start=start, m=m, x=x2,
                        dx=abs(x2 - x1), res=abs(f(x2)))
        x0, x1 = x1, x2
    return dict(name="Секущих", start=start, m=m, x=x1,
                dx=abs(x1 - x0), res=abs(f(x1)), fail="не сошёлся")


def print_results(results):
    for r in results:
        print("-" * 70)
        print(f"Метод: {r['name']}")
        print(f"  начальное приближение: {r['start']}")
        print(f"  число шагов m        : {r['m']}")
        print(f"  x_m                  : {r['x']:.12f}")
        print(f"  |x_m - x_(m-1)|      : {r['dx']:.3e}"
              + ("  (длина последнего отрезка)" if r['name'] == "Бисекция" else ""))
        print(f"  невязка |f(x_m)|     : {r['res']:.3e}")
        if "fail" in r:
            print(f"  !!! ПРОБЛЕМА: {r['fail']}")
    print("-" * 70)


def refine_all(f, df, a, b, eps):
    x0 = (a + b) / 2                 # старт Ньютона и мод. Ньютона: середина отрезка
    res = [bisection(f, a, b, eps),
           newton(f, df, x0, eps),
           mod_newton(f, df, x0, eps),
           secant(f, a, b, eps)]     # секущие: x0 = a, x1 = b
    print_results(res)
    return res


# ---------------------------------------------------------------------------
# 5. ЗАДАЧА О ПОГРУЖЕНИИ ШАРА
#    Условие плавания: масса шара = масса вытесненной воды.
#    Объём шарового сегмента V = pi*d^2*(3r-d)/3, объём шара 4/3*pi*r^3,
#    rho_water = 1  =>  d^3 - 3 r d^2 + 4 rho r^3 = 0,  d in [0, 2r].
#    f(0) = 4 rho r^3 > 0, f(2r) = 4 r^3 (rho-1) < 0 при rho < 1: корень единственный.
# ---------------------------------------------------------------------------
MATERIALS = [("Пробка", 0.25), ("Бамбук", 0.4), ("Сосна (белая)", 0.5),
             ("Кедр", 0.55), ("Дуб", 0.7), ("Бук", 0.75),
             ("Красное дерево", 0.8), ("Тиковое дерево", 0.85),
             ("Парафин", 0.9), ("Лёд/Полиэтилен", 0.92),
             ("Пчелиный воск", 0.95)]


def ball_depth(r, rho, eps):
    """Глубина погружения d шара радиуса r, плотность rho (г/мл).
    Возвращает (d, число шагов, невязка, ok); ok=False - результат получен бисекцией."""
    f = lambda d: d**3 - 3*r*d**2 + 4*rho*r**3
    df = lambda d: 3*d**2 - 6*r*d
    # Ньютон со стартом d0 = r (f'(r) = -3r^2 != 0); сверка с бисекцией на [0, 2r]
    rn = newton(f, df, r, eps)
    rb = bisection(f, 0.0, 2*r, eps)
    ok = "fail" not in rn and abs(rn['x'] - rb['x']) < 1e-6 and 0 <= rn['x'] <= 2*r
    if ok:
        return rn['x'], rn['m'], rn['res'], True
    return rb['x'], rb['m'], rb['res'], False


def ball_table(r, eps=1e-8):
    print(f"\nЗАДАЧА О ШАРЕ: r = {r} м, eps = {eps}")
    print(f"{'Вещество':<18}{'rho, г/мл':>10}{'d, м':>16}{'d/(2r)':>10}"
          f"{'шагов':>7}{'|f(d)|':>12}")
    for name, rho in MATERIALS:
        d, m, res, ok = ball_depth(r, rho, eps)
        print(f"{name:<18}{rho:>10.2f}{d:>16.8f}{d/(2*r):>10.4f}{m:>7d}{res:>12.2e}"
              + ("" if ok else "  (Ньютон не подошёл -> бисекция)"))


def ball_custom(r, eps=1e-8):
    """Свой материал: название и плотность в г/мл (0 < rho < 1)."""
    name = input("Название материала [мой материал]: ").strip() or "мой материал"
    rho = ask("Плотность rho, г/мл (воды = 1)", 0.6, float)
    if rho <= 0:
        print("Требуется rho > 0.")
        return
    if rho >= 1:
        print(f"rho = {rho} >= 1: шар не плавает (тонет или нейтрально плавучий, "
              f"глубина погружения d = 2r = {2*r} м); корня на (0, 2r) нет.")
        return
    d, m, res, ok = ball_depth(r, rho, eps)
    print(f"\nМатериал: {name}, rho = {rho}, r = {r} м, eps = {eps}")
    print(f"  глубина погружения d = {d:.10f} м  (d/(2r) = {d/(2*r):.4f})")
    print(f"  число шагов = {m},  |f(d)| = {res:.2e}"
          + ("" if ok else "  (Ньютон не подошёл -> бисекция)"))


def ball_menu():
    print("\nЗАДАЧА О ПОГРУЖЕНИИ ШАРА")
    r = ask("Радиус шара r, м (r > 0)", 0.62, float)
    while r <= 0:
        r = ask("Требуется r > 0", 0.62, float)
    while True:
        c = input("\n1 - таблица для материалов из списка; 2 - ввести свой материал; "
                  "0 - в главное меню [1]: ").strip()
        if c in ("", "1"):
            ball_table(r)
        elif c == "2":
            ball_custom(r)
        elif c == "0":
            return
        else:
            print("Введите 1, 2 или 0.")


# ---------------------------------------------------------------------------
# 6. РЕШЕНИЕ УРАВНЕНИЯ (тестовая или введённая функция)
# ---------------------------------------------------------------------------
def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, использовано значение по умолчанию.")
        return default


def read_function():
    """Ввод f(x): имена из math (sin, cos, exp, log, sqrt, pi, ...), степень ** или ^.
    Выражение проверяется пробным вычислением в x = 0.5."""
    print("Введите f(x) в синтаксисе Python, например: x**3 - 2*x + 1  или  sin(x) - x/2")
    print("(умножение - только знаком *, степень - ** или ^, логарифм - log(x))")
    while True:
        expr = input("f(x) = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr)
            f(0.5)                        # пробное вычисление: проверка имён и синтаксиса
        except Exception as e:            # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторите.")
            continue
        return expr, f


def solve_equation(expr, f, A0, B0, eps0, manual=False):
    """Отделение корней и уточнение на выбранных отрезках."""
    print(f"\nf(x) = {expr}")
    A = ask("A", A0, float)
    B = ask("B", B0, float)
    while not A < B:
        print("Требуется A < B.")
        A = ask("A", A0, float)
        B = ask("B", B0, float)
    N = ask("N (число частей, N>=2)", 1000, int)
    while N < 2:
        N = ask("Требуется N >= 2", 1000, int)

    df, dtype = make_derivative(expr, f)
    if manual:
        s = input("Производная f'(x) (Enter - вычислить автоматически): ").strip().replace("^", "**")
        if s:
            try:
                dfm = make_function(s)
                dfm(0.5)
                df, dtype = dfm, "введена вручную"
            except Exception as e:
                print(f"Не удалось разобрать производную ({e}); используется автоматическая.")

    print(f"\nЗАДАЧА: найти все корни нечётной кратности f(x)=0 на [{A}; {B}]")
    print(f"f(x) = {expr}   (производная: {dtype})")

    segs = separate_roots(f, A, B, N)
    # контроль полноты: табулирование с шагом в 10 раз меньше
    cnt10 = len(separate_roots(f, A, B, 10 * N, verbose=False))
    print(f"Контроль: при N*10={10*N} отрезков найдено {cnt10} "
          + ("(совпадает - отделение завершено)" if cnt10 == len(segs)
             else "(не совпадает - отделение не завершено)"))
    if not segs:
        print("Отрезков перемены знака нет.")
        return

    while True:
        k = ask(f"Номер отрезка для уточнения (1..{len(segs)}, 0 - назад в меню)", 1, int)
        if k == 0:
            return
        if not 1 <= k <= len(segs):
            print("Нет такого отрезка.")
            continue
        a, b = segs[k - 1]
        eps = ask("Точность eps", eps0, float)
        while eps <= 0:
            eps = ask("Требуется eps > 0", eps0, float)
        print(f"\nУточнение на отрезке [{a:.10f}; {b:.10f}], eps = {eps}")
        refine_all(f, df, a, b, eps)


def main():
    print("=" * 70)
    print("ЗАДАНИЕ №1. ЧИСЛЕННЫЕ МЕТОДЫ РЕШЕНИЯ НЕЛИНЕЙНЫХ УРАВНЕНИЙ")
    print("=" * 70)
    while True:
        print("\nГЛАВНОЕ МЕНЮ")
        print("  1 - нелинейное уравнение: тестовая задача из таблицы (1-30)")
        print("  2 - нелинейное уравнение: ввести функцию вручную")
        print("  3 - задача о погружении шара")
        print("  q - выход")
        c = input("Выбор [1]: ").strip().lower() or "1"
        if c == "q":
            break
        if c == "1":
            n = ask("Номер тестовой задачи (1-30)", 1, int)
            while n not in TASKS:
                n = ask("Нет такой задачи. Номер (1-30)", 1, int)
            expr, A0, B0, eps0 = TASKS[n]
            solve_equation(expr, make_function(expr), A0, B0, eps0)
        elif c == "2":
            expr, f = read_function()
            solve_equation(expr, f, -5.0, 5.0, 1e-6, manual=True)
        elif c == "3":
            ball_menu()
        else:
            print("Введите 1, 2, 3 или q.")


if __name__ == "__main__":
    main()
