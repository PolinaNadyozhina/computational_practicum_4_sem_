# -*- coding: utf-8 -*-
"""
Задание №3. Численное дифференцирование таблично заданной функции.

3.1. Формулы для f''(x) по значениям f(x), f(x+h), f(x+2h), f(x+3h)  (12)
     и по f(x), f(x-h), f(x-2h), f(x-3h)                                (13)
     (метод неопределённых коэффициентов):
       (12): f''(x) = [2 f(x) - 5 f(x+h) + 4 f(x+2h) - f(x+3h)] / h^2
                      + 11/12 h^2 f''''(xi),  xi in (x, x+3h)
       (13): f''(x) = [2 f(x) - 5 f(x-h) + 4 f(x-2h) - f(x-3h)] / h^2
                      + 11/12 h^2 f''''(xi),  xi in (x-3h, x)

3.2. По таблице y_k = f(x_k), x_k = x0 + k h, k = 0..m, вычисляются:
     a) f'  с погрешностью O(h^2): (4) в x0, (3) во внутренних, (5) в x_m;
     b) f'  с погрешностью O(h^4): (7) x0, (8) x1, (9) x2..x_{m-2}, (10) x_{m-1}, (11) x_m;
     c) f'' с погрешностью O(h^2): (6) во внутренних, (12) в x0, (13) в x_m.
     Результат: таблица с "точными" значениями производных и абсолютными
     фактическими погрешностями; затем исследование оптимального шага.

Ввод с клавиатуры (Enter - значение по умолчанию). Числа - float (double).
Источник данных: функция из списка, своя функция (выражение; производные до
4-го порядка вычисляются рядами Тейлора) или своя таблица значений y_k.
Формулы используют только табличные значения y_k; аналитическая функция
нужна для построения таблицы и "точных" производных.
"""
import math

# ---------------------------------------------------------------------------
# Функции: (название, [f, f', f'', f''', f''''])
# ---------------------------------------------------------------------------
FUNCS = {
    1: ("f1 = 3x^2 - 2x + 1  (многочлен степени 2)",
        [lambda x: 3*x*x - 2*x + 1, lambda x: 6*x - 2, lambda x: 6.0,
         lambda x: 0.0, lambda x: 0.0]),
    2: ("f2 = x^3 - 2x^2 + x - 1  (многочлен степени 3)",
        [lambda x: x**3 - 2*x*x + x - 1, lambda x: 3*x*x - 4*x + 1,
         lambda x: 6*x - 4, lambda x: 6.0, lambda x: 0.0]),
    3: ("f3 = exp(4x)  (быстро растущая)",
        [lambda x: math.exp(4*x), lambda x: 4*math.exp(4*x),
         lambda x: 16*math.exp(4*x), lambda x: 64*math.exp(4*x),
         lambda x: 256*math.exp(4*x)]),
    4: ("f4 = sin(2x) - 1.25x^2 + 0.35  (гладкая)",
        [lambda x: math.sin(2*x) - 1.25*x*x + 0.35,
         lambda x: 2*math.cos(2*x) - 2.5*x,
         lambda x: -4*math.sin(2*x) - 2.5,
         lambda x: -8*math.cos(2*x),
         lambda x: 16*math.sin(2*x)]),
    5: ("f5 = exp(2x)  (исследование оптимального шага)",
        [lambda x: math.exp(2*x), lambda x: 2*math.exp(2*x),
         lambda x: 4*math.exp(2*x), lambda x: 8*math.exp(2*x),
         lambda x: 16*math.exp(2*x)]),
}

# ---------------------------------------------------------------------------
# Своя функция: выражение задаётся текстом; "точные" производные f', f'', f''', f''''
# вычисляются арифметикой усечённых рядов Тейлора (с машинной точностью).
# ---------------------------------------------------------------------------
ORD = 4                                           # старший порядок производной


class Jet:
    """Усечённый ряд Тейлора c0 + c1 t + ... + c4 t^4 в точке (t = x - x0)."""

    def __init__(self, c):
        self.c = list(c) + [0.0] * (ORD + 1 - len(c))

    @staticmethod
    def lift(v):
        return v if isinstance(v, Jet) else Jet([float(v)])

    def __add__(self, o):
        o = Jet.lift(o)
        return Jet([a + b for a, b in zip(self.c, o.c)])
    __radd__ = __add__

    def __neg__(self):
        return Jet([-a for a in self.c])

    def __pos__(self):
        return self

    def __sub__(self, o):
        return self + (-Jet.lift(o))

    def __rsub__(self, o):
        return Jet.lift(o) + (-self)

    def __mul__(self, o):
        o = Jet.lift(o)
        return Jet([sum(self.c[j] * o.c[k - j] for j in range(k + 1)) for k in range(ORD + 1)])
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Jet.lift(o)
        q = []
        for k in range(ORD + 1):
            q.append((self.c[k] - sum(q[j] * o.c[k - j] for j in range(k))) / o.c[0])
        return Jet(q)

    def __rtruediv__(self, o):
        return Jet.lift(o) / self

    def __pow__(self, p):
        if isinstance(p, float) and p == int(p) and abs(p) <= 50:
            p = int(p)
        if isinstance(p, int):
            if p >= 0:
                r = Jet([1.0])
                for _ in range(p):
                    r = r * self
                return r
            return Jet([1.0]) / (self ** (-p))
        return jexp(Jet.lift(p) * jlog(self))

    def __rpow__(self, b):
        return jexp(self * math.log(b))

    def deriv(self, k):
        return math.factorial(k) * self.c[k]


def jexp(u):
    e = [math.exp(u.c[0])]
    for k in range(1, ORD + 1):
        e.append(sum(j * u.c[j] * e[k - j] for j in range(1, k + 1)) / k)
    return Jet(e)


def jlog(u):
    if u.c[0] <= 0:
        raise ValueError("log от неположительного числа")
    l = [math.log(u.c[0])]
    for k in range(1, ORD + 1):
        l.append((u.c[k] - sum(j * l[j] * u.c[k - j] for j in range(1, k)) / k) / u.c[0])
    return Jet(l)


def jsincos(u):
    s, c = [math.sin(u.c[0])], [math.cos(u.c[0])]
    for k in range(1, ORD + 1):
        s.append(sum(j * u.c[j] * c[k - j] for j in range(1, k + 1)) / k)
        c.append(-sum(j * u.c[j] * s[k - j] for j in range(1, k + 1)) / k)
    return Jet(s), Jet(c)


def jintegrate(d, a0):
    """Первообразная ряда d с постоянной a0."""
    return Jet([a0] + [d.c[k - 1] / k for k in range(1, ORD + 1)])


def jderiv(u):
    return Jet([(k + 1) * u.c[k + 1] for k in range(ORD)])


def jatan(u):
    return jintegrate(jderiv(u) / (1 + u * u), math.atan(u.c[0]))


def jasin(u):
    return jintegrate(jderiv(u) / (1 - u * u) ** 0.5, math.asin(u.c[0]))


def _wrap(jf, mf):
    return lambda u: jf(u) if isinstance(u, Jet) else mf(u)


def _env():
    """Имена, доступные в выражении (для чисел и для рядов Тейлора)."""
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    env.update(
        exp=_wrap(jexp, math.exp), log=_wrap(jlog, math.log),
        sin=_wrap(lambda u: jsincos(u)[0], math.sin), cos=_wrap(lambda u: jsincos(u)[1], math.cos),
        tan=_wrap(lambda u: jsincos(u)[0] / jsincos(u)[1], math.tan),
        sqrt=_wrap(lambda u: u ** 0.5, math.sqrt),
        sinh=_wrap(lambda u: (jexp(u) - jexp(-u)) / 2, math.sinh),
        cosh=_wrap(lambda u: (jexp(u) + jexp(-u)) / 2, math.cosh),
        tanh=_wrap(lambda u: (jexp(u) - jexp(-u)) / (jexp(u) + jexp(-u)), math.tanh),
        atan=_wrap(jatan, math.atan), asin=_wrap(jasin, math.asin),
        acos=_wrap(lambda u: -jasin(u) + math.pi / 2, math.acos),
        log10=_wrap(lambda u: jlog(u) / math.log(10), math.log10),
    )
    return env


ENV = _env()


def make_function(expr):
    """[f, f', f'', f''', f''''] для выражения expr (переменная x)."""
    code = compile(expr, "<f>", "eval")

    def val(x):
        return eval(code, {"__builtins__": {}}, dict(ENV, x=x))

    def der(k):
        return lambda x: Jet.lift(val(Jet([x, 1.0]))).deriv(k)

    return [lambda x: float(val(float(x)))] + [der(k) for k in range(1, ORD + 1)]


def read_function():
    """Ввод f(x): синтаксис Python, степень ** или ^, умножение только *;
    имена: sin, cos, tan, exp, log, sqrt, atan, asin, sinh, cosh, tanh, pi, e."""
    print("Введите f(x), например: x**3 - 2*x + 1  или  sin(2*x) - 1.25*x^2")
    while True:
        expr = input("f(x) = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            fs = make_function(expr)
            fs[4](0.5)                      # пробное вычисление
        except (ValueError, ZeroDivisionError, OverflowError):
            return expr, fs                 # синтаксис верен, 0.5 вне области определения
        except Exception as e:              # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторите.")
            continue
        return expr, fs


def read_table(count):
    """Ввод count значений y_k (через пробел, в несколько строк)."""
    print(f"Введите {count} значений y_0 ... y_{count - 1} через пробел (допустимо в несколько строк):")
    ys = []
    while len(ys) < count:
        try:
            ys += [float(t) for t in input("  y: ").replace(",", ".").split()]
        except ValueError:
            print("  Нужны числа, строка пропущена.")
    return ys[:count]


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, использовано значение по умолчанию.")
        return default


# ---------------------------------------------------------------------------
# ФОРМУЛЫ (входные данные: таблица y и шаг h)
# ---------------------------------------------------------------------------
def d1_order2(y, h):
    """f' с O(h^2): (4) в начале, (3) внутри, (5) в конце."""
    m = len(y) - 1
    d = [0.0] * (m + 1)
    d[0] = (-3*y[0] + 4*y[1] - y[2]) / (2*h)                       # (4)
    for k in range(1, m):
        d[k] = (y[k+1] - y[k-1]) / (2*h)                           # (3)
    d[m] = (3*y[m] - 4*y[m-1] + y[m-2]) / (2*h)                    # (5)
    return d


def d1_order4(y, h):
    """f' с O(h^4): (7),(8),(9),(10),(11). Требуется m >= 4."""
    m = len(y) - 1
    d = [0.0] * (m + 1)
    d[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*h)          # (7)
    d[1] = (-3*y[0] - 10*y[1] + 18*y[2] - 6*y[3] + y[4]) / (12*h)              # (8)
    for k in range(2, m - 1):
        d[k] = (y[k-2] - 8*y[k-1] + 8*y[k+1] - y[k+2]) / (12*h)                # (9)
    d[m-1] = (3*y[m] + 10*y[m-1] - 18*y[m-2] + 6*y[m-3] - y[m-4]) / (12*h)     # (10)
    d[m] = (25*y[m] - 48*y[m-1] + 36*y[m-2] - 16*y[m-3] + 3*y[m-4]) / (12*h)  # (11)
    return d


def d2_order2(y, h):
    """f'' с O(h^2): (6) внутри, (12) в начале, (13) в конце. Требуется m >= 3."""
    m = len(y) - 1
    d = [0.0] * (m + 1)
    d[0] = (2*y[0] - 5*y[1] + 4*y[2] - y[3]) / (h*h)                           # (12)
    for k in range(1, m):
        d[k] = (y[k+1] - 2*y[k] + y[k-1]) / (h*h)                              # (6)
    d[m] = (2*y[m] - 5*y[m-1] + 4*y[m-2] - y[m-3]) / (h*h)                     # (13)
    return d


# ---------------------------------------------------------------------------
# Таблица результатов
# ---------------------------------------------------------------------------
def solve_table(x0, h, y, fs=None):
    """y - таблица значений; fs - [f, f', f'', ...] или None (без 'точных' значений
    печатаются только приближения)."""
    count = len(y)
    x = [x0 + k*h for k in range(count)]
    a1 = d1_order2(y, h)
    a2 = d1_order4(y, h)
    a3 = d2_order2(y, h)

    print(f"\nТаблица значений (m+1 = {count}, x0 = {x0}, h = {h}):")
    if fs is None:
        print(f"{'k':>3}{'x_k':>10}{'y_k':>16}{'f~ O(h2)':>16}{'f~ O(h4)':>16}{'f2~ O(h2)':>16}")
        for k in range(count):
            print(f"{k:>3}{x[k]:>10.5f}{y[k]:>16.8f}{a1[k]:>16.8f}{a2[k]:>16.8f}{a3[k]:>16.8f}")
        print("\n'Точные' производные не заданы - погрешности не вычисляются.")
        return
    d1t = [fs[1](t) for t in x]
    d2t = [fs[2](t) for t in x]
    w = "{:>3}{:>10.5f}{:>14.8f}{:>14.8f}{:>14.8f}{:>10.2e}{:>14.8f}{:>10.2e}{:>14.8f}{:>14.8f}{:>10.2e}"
    print(f"{'k':>3}{'x_k':>10}{'y_k':>14}{'f_T':>14}{'f~ O(h2)':>14}{'погр.':>10}"
          f"{'f~ O(h4)':>14}{'погр.':>10}{'f2_T':>14}{'f2~ O(h2)':>14}{'погр.':>10}")
    for k in range(count):
        print(w.format(k, x[k], y[k], d1t[k], a1[k], abs(d1t[k]-a1[k]),
                       a2[k], abs(d1t[k]-a2[k]), d2t[k], a3[k], abs(d2t[k]-a3[k])))
    e1 = max(abs(d1t[k]-a1[k]) for k in range(count))
    e2 = max(abs(d1t[k]-a2[k]) for k in range(count))
    e3 = max(abs(d2t[k]-a3[k]) for k in range(count))
    print(f"\nМаксимальные погрешности: f' O(h^2): {e1:.3e};  f' O(h^4): {e2:.3e};  f'' O(h^2): {e3:.3e}")


# ---------------------------------------------------------------------------
# Оптимальный шаг
# ---------------------------------------------------------------------------
def optimal_step(fs):
    f, df, ddf, d3, d4 = fs
    print("\n--- ИССЛЕДОВАНИЕ ОПТИМАЛЬНОГО ШАГА ---")
    which = ask("Производная: 1 - f' по формуле (4) O(h^2); 2 - f'' по формуле (6)", 1, int)
    x = ask("Точка x", 1.0, float)
    h = ask("Начальный шаг h", 0.1, float)
    s = ask("Значения функции округлять до s знаков после запятой (s)", 5, int)
    steps = ask("Сколько раз уменьшать шаг вдвое", 10, int)
    eps = 0.5 * 10**(-s)                      # погрешность округления
    r = lambda t: round(t, s)                 # округлённые значения функции

    exact = df(x) if which == 1 else ddf(x)
    print(f"\n'Точное' значение производной в x={x}: {exact:.10f};  eps = {eps:.1e}")
    print(f"{'h':>12}{'приближённое':>18}{'погрешность':>16}")
    rows = []
    for _ in range(steps + 1):
        if which == 1:
            val = (-3*r(f(x)) + 4*r(f(x + h)) - r(f(x + 2*h))) / (2*h)
        else:
            val = (r(f(x + h)) - 2*r(f(x)) + r(f(x - h))) / (h*h)
        err = abs(exact - val)
        rows.append((h, val, err))
        print(f"{h:>12.6f}{val:>18.8f}{err:>16.6e}")
        h /= 2
    best = min(rows, key=lambda t: t[2])
    print(f"\nЭкспериментально оптимальный шаг (минимум погрешности): h = {best[0]:.6f}, "
          f"погрешность {best[2]:.3e}")
    # теоретическая оценка
    h0 = rows[0][0]
    if which == 1:
        M = max(abs(d3(x + 2*h0*t/200)) for t in range(201))
        # |R| <= 8eps/(2h) + h^2/3 * M3  =>  h_opt = (6 eps / M3)^(1/3)
        if M > 0:
            ho = (6*eps/M) ** (1/3)
            print(f"Теоретически: |R| <= 4eps/h + h^2 M3/3, M3 ~ {M:.3f} => h_opt ~ {ho:.5f}")
        else:
            print("Теоретически: M3 = 0 (многочлен степени <= 2), формула точна, h_opt не определён.")
    else:
        M = max(abs(d4(x + h0*(t/100 - 1))) for t in range(201))
        # |R| <= 4eps/h^2 + h^2/12 * M4 => h_opt = (48 eps / M4)^(1/4)
        if M > 0:
            ho = (48*eps/M) ** 0.25
            print(f"Теоретически: |R| <= 4eps/h^2 + h^2 M4/12, M4 ~ {M:.3f} => h_opt ~ {ho:.5f}")
        else:
            print("Теоретически: M4 = 0 (многочлен степени <= 3), формула точна, h_opt не определён.")


# ---------------------------------------------------------------------------
def read_grid(default_x0=0.0, default_h=0.1):
    count = ask("Количество точек m+1 (>= 5)", 11, int)
    while count < 5:
        count = ask("Нужно m+1 >= 5 (для формул порядка O(h^4)). Повторите", 11, int)
    x0 = ask("Начальная точка x0", default_x0, float)
    h = ask("Шаг h (>0)", default_h, float)
    while h <= 0:
        h = ask("Нужно h > 0. Повторите", default_h, float)
    return count, x0, h


def main():
    print("=" * 74)
    print("ЗАДАНИЕ №3. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ ТАБЛИЧНО ЗАДАННОЙ ФУНКЦИИ")
    print("=" * 74)
    while True:
        print("\nИсточник данных:")
        print("  1) функция из списка")
        print("  2) своя функция (вводится выражением)")
        print("  3) своя таблица значений y_k (равноотстоящие узлы)")
        c = input("Выбор [1], q - выход: ").strip().lower()
        if c == "q":
            break
        fs = None
        if c == "3":
            count, x0, h = read_grid()
            y = read_table(count)
        else:
            if c == "2":
                expr, fs = read_function()
                print(f"Выбрано: f(x) = {expr}")
            else:
                print("\nФункции:")
                for k, (name, _) in FUNCS.items():
                    print(f"  {k}) {name}")
                n = ask("Номер функции", 4, int)
                while n not in FUNCS:
                    n = ask("Нет такой функции, повторите", 4, int)
                fs = FUNCS[n][1]
                print(f"Выбрано: {FUNCS[n][0]}")
            count, x0, h = read_grid()
            try:
                y = [fs[0](x0 + k*h) for k in range(count)]
                for t in (x0, x0 + (count - 1)*h):
                    fs[4](t)                      # проверка существования f и производных
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                print(f"Функция не определена на отрезке [{x0}, {x0 + (count-1)*h}] ({e}). "
                      "Выберите другие данные.")
                continue

        try:
            solve_table(x0, h, y, fs)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            print(f"Ошибка вычисления производных функции ({e}).")
            continue

        if fs is not None and input("\nИсследовать оптимальный шаг? (y/n) [n]: ").strip().lower() == "y":
            try:
                optimal_step(fs)
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                print(f"Ошибка: функция вне области определения при таких x, h ({e}).")

        if input("\nНовые данные (Enter) или выход (q): ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
