# -*- coding: utf-8 -*-
"""
Задание №2. Задача алгебраического интерполирования.
Интерполяционный многочлен в форме Лагранжа и в форме Ньютона.

Алгоритм:
 1) таблица (z_k, f(z_k)), k = 0..m, узлы попарно различны
    (равноотстоящие, случайные или заданные вручную);
 2) ввод x и n (n <= m);
 3) сортировка таблицы по |z_k - x| по возрастанию, выбор первых n+1 узлов;
 4) P_n^L(x) по Лагранжу (контроль: сумма l_k = 1);
 5) P_n^N(x) по Ньютону через таблицу разделённых разностей;
 6) погрешности |f(x) - P(x)|; повтор с новыми x, n или выход.
Источник данных (главное меню):
  1 - вариант из списка (1-18; 0 - тестовый многочлен);
  2 - функция f(x), заданная выражением; a, b, число значений и узлы
      (равноотстоящие, случайные или введённые вручную);
  3 - таблица, введённая вручную: погрешность вычисляется только при
      введённом значении f(x).
Ввод: клавиатура (Enter - значение по умолчанию). Тип чисел: float (64 бита).
"""
import math
import random

# вариант: номер -> (выражение, a, b, m+1, n)
VARIANTS = {
    0: ("x**3 - 2*x + 1", 0, 1, 10, 5),           # тестовый многочлен (q = 3)
    1: ("sin(x) - x**2/2", 0, 1, 21, 9),
    2: ("log(1+x) - 2*x", 0, 1.5, 16, 8),
    3: ("exp(x) - x", 0, 2, 41, 12),
    4: ("sqrt(1+x**2)", 0, 0.7, 15, 10),
    5: ("1 - exp(-2*x)", -0.5, 1, 51, 8),
    6: ("x**2/(1+x**2)", 0.4, 1, 13, 11),
    7: ("exp(-x) - x**2/2", 0, 5, 26, 10),
    8: ("2*sin(x) - x/2", 0.2, 0.7, 20, 13),
    9: ("1 - exp(-x) + x**2", 0, 1.5, 16, 9),
    10: ("cos(x) + 2*x", 0.5, 1.8, 14, 7),
    11: ("sin(x) + x**2/2", 0.4, 1.9, 31, 11),
    12: ("exp(-x) - x**2/2", 0, 1, 16, 8),
    13: ("log(1+x) - exp(x)", 1, 10, 31, 7),
    14: ("sqrt(1+x) + x", 0, 1, 26, 7),
    15: ("exp(-x) - x**2/2", 0, 1, 16, 8),
    16: ("x**2 - x - exp(x)", -3, 5, 41, 10),
    17: ("sqrt(1+x) + x**2 - 1", 0, 1, 26, 8),
    18: ("cos(x) + sin(2*x)", 0, 3, 16, 10),
}


def make_function(expr):
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    code = compile(expr, "<f>", "eval")
    return lambda x: float(eval(code, {"__builtins__": {}}, dict(env, x=x)))


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, использовано значение по умолчанию.")
        return default


# ---------------------------------------------------------------------------
# Подготовка таблицы
# ---------------------------------------------------------------------------
def make_nodes(a, b, count, kind):
    """Попарно различные узлы из [a, b]."""
    m = count - 1
    if kind == 1:                              # равноотстоящие z_k = a + k*h
        h = (b - a) / m
        return [a + k * h for k in range(count)]
    nodes = set()
    while len(nodes) < count:                  # случайные; set обеспечивает попарную различность
        nodes.add(random.uniform(a, b))
    return sorted(nodes)


def print_table(z, fz, title):
    print(title)
    print(f"{'k':>4}{'z_k':>20}{'f(z_k)':>22}")
    for k, (zk, fk) in enumerate(zip(z, fz)):
        print(f"{k:>4}{zk:>20.12f}{fk:>22.12f}")


# ---------------------------------------------------------------------------
# Лагранж
# ---------------------------------------------------------------------------
def lagrange(xs, ys, x):
    """Возвращает (P(x), сумма коэффициентов l_k(x)).
    l_k(x) = prod_{j!=k} (x-x_j)/(x_k-x_j); деления на (x-x_k) нет,
    поэтому случай x = узел обрабатывается без деления на ноль."""
    total, s = 0.0, 0.0
    for k in range(len(xs)):
        l = 1.0
        for j in range(len(xs)):
            if j != k:
                l *= (x - xs[j]) / (xs[k] - xs[j])
        s += l
        total += ys[k] * l
    return total, s


# ---------------------------------------------------------------------------
# Ньютон
# ---------------------------------------------------------------------------
def divided_differences(xs, ys):
    """Таблица разделённых разностей: dd[i][k] = f(x_i,...,x_{i+k})."""
    n1 = len(xs)
    dd = [[0.0] * n1 for _ in range(n1)]
    for i in range(n1):
        dd[i][0] = ys[i]
    for k in range(1, n1):
        for i in range(n1 - k):
            dd[i][k] = (dd[i + 1][k - 1] - dd[i][k - 1]) / (xs[i + k] - xs[i])
    return dd


def newton(xs, dd, x):
    """P(x) = f(x0) + f(x0,x1)(x-x0) + ...; вычисление по схеме Горнера."""
    n = len(xs) - 1
    p = dd[0][n]
    for k in range(n - 1, -1, -1):
        p = p * (x - xs[k]) + dd[0][k]
    return p


def print_dd(xs, dd):
    n1 = len(xs)
    print("Таблица разделённых разностей (строка i: f(x_i), f(x_i,x_i+1), ...):")
    head = f"{'i':>3}{'x_i':>14}" + "".join(f"{'ПР'+str(k):>15}" for k in range(n1))
    print(head)
    for i in range(n1):
        row = f"{i:>3}{xs[i]:>14.8f}"
        for k in range(n1 - i):
            row += f"{dd[i][k]:>15.6e}"
        print(row)


# ---------------------------------------------------------------------------
# Ввод данных
# ---------------------------------------------------------------------------
def read_function():
    """Ввод f(x): синтаксис Python, степень ** или ^, имена из math
    (sin, cos, exp, log, sqrt, pi, ...). Проверка пробным вычислением в x = 0.5."""
    print("Введите f(x), например: x**3 - 2*x + 1  или  sin(x) - x/2")
    print("(умножение - только знаком *, степень - ** или ^, логарифм - log(x))")
    while True:
        expr = input("f(x) = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr)
            f(0.5)
        except (ValueError, ZeroDivisionError, OverflowError):
            return expr, f                 # выражение корректно; x = 0.5 вне области определения
        except Exception as e:             # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторите.")
            continue
        return expr, f


def read_nodes(count):
    """Ввод count попарно различных узлов через пробел."""
    while True:
        raw = input(f"Введите {count} различных узлов через пробел: ").replace(",", ".").split()
        try:
            z = [float(t) for t in raw]
        except ValueError:
            print("Некорректный ввод.")
            continue
        if len(z) != count:
            print(f"Нужно ровно {count} узлов, введено {len(z)}.")
        elif len(set(z)) != count:
            print("Узлы должны быть попарно различны.")
        else:
            return sorted(z)


def read_table():
    """Ввод таблицы: число значений m+1, затем строки 'z_k  f(z_k)'."""
    count = ask("Число значений в таблице (m+1 >= 2)", 6, int)
    while count < 2:
        count = ask("Требуется m+1 >= 2", 6, int)
    print("Вводите по одной строке: узел и значение функции через пробел (например: 0.5 1.25)")
    z, fz = [], []
    while len(z) < count:
        raw = input(f"  строка {len(z)}: ").replace(",", ".").split()
        try:
            zk, fk = float(raw[0]), float(raw[1])
        except (ValueError, IndexError):
            print("  Нужны два числа: узел и значение.")
            continue
        if zk in z:
            print("  Такой узел уже введён: узлы должны быть попарно различны.")
            continue
        z.append(zk)
        fz.append(fk)
    order = sorted(range(count), key=lambda i: z[i])
    return [z[i] for i in order], [fz[i] for i in order]


def build_from_function(f, a0, b0, cnt0):
    """Запрос count, a, b и вида узлов; таблица по f. Возвращает (z, fz, a, b)."""
    while True:
        count = ask("Число значений в таблице (m+1)", cnt0, int)
        while count < 2:
            count = ask("Требуется m+1 >= 2", cnt0, int)
        a = ask("a", a0, float)
        b = ask("b", b0, float)
        while a >= b:
            print("Требуется a < b")
            a = ask("a", a0, float)
            b = ask("b", b0, float)
        kind = ask("Узлы: 1 - равноотстоящие, 2 - случайные, 3 - ввести вручную", 2, int)
        z = read_nodes(count) if kind == 3 else make_nodes(a, b, count, kind)
        try:
            fz = [f(t) for t in z]
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            print(f"Функция не определена в одном из узлов ({e}). Измените a, b или узлы.")
            continue
        return z, fz, min(a, min(z)), max(b, max(z))


# ---------------------------------------------------------------------------
def interpolate(z, fz, f, a, b, n0):
    """Цикл: ввод x и n, сортировка узлов, P_n^L, P_n^N, погрешности.
    f - функция или None (задана только таблица)."""
    count = len(z)
    m = count - 1
    while True:
        x = ask("\nТочка интерполирования x", (a + b) / 2, float)
        n = ask(f"Степень n (n <= m = {m})", min(n0, m), int)
        while n > m or n < 0:
            print("Введено недопустимое значение n")
            n = ask(f"Введите n <= {m}", min(n0, m), int)
        print(f"x = {x}, n = {n}")

        # значение f(x): по функции, для таблицы - из ввода (необязательно)
        fx = None
        if f is not None:
            try:
                fx = f(x)
            except (ValueError, ZeroDivisionError, OverflowError):
                print("f(x) в этой точке не определена - погрешность не считается.")
        else:
            s = input("Истинное значение f(x), если известно (Enter - нет): ").strip().replace(",", ".")
            try:
                fx = float(s) if s else None
            except ValueError:
                print("Некорректное число - погрешность не считается.")

        # устойчивая сортировка по |z_k - x|
        order = sorted(range(count), key=lambda i: abs(z[i] - x))
        sx = [z[i] for i in order]
        sy = [fz[i] for i in order]
        print("\nОтсортированная таблица (по |z_k - x|):")
        print(f"{'k':>4}{'x_k':>18}{'f(x_k)':>20}{'|x_k - x|':>16}")
        for k in range(count):
            mark = "  <- узел для P_n" if k <= n else ""
            print(f"{k:>4}{sx[k]:>18.12f}{sy[k]:>20.12f}{abs(sx[k]-x):>16.6e}{mark}")

        xs, ys = sx[:n + 1], sy[:n + 1]
        print(f"\nP_{n} строится по {n+1} ближайшим к x узлам x_0..x_{n}.")

        pl, ssum = lagrange(xs, ys, x)
        if fx is not None:
            print(f"\nf(x)                 = {fx:.14f}")
        print(f"P_n^L(x) (Лагранж)   = {pl:.14f}")
        if fx is not None:
            print(f"|f(x) - P_n^L(x)|    = {abs(fx - pl):.3e}")
        print(f"Контроль: сумма l_k(x) = {ssum:.15f}  (|сумма-1| = {abs(ssum-1):.2e})")

        dd = divided_differences(xs, ys)
        print()
        if n <= 8:
            print_dd(xs, dd)
        else:
            print("(таблица разделённых разностей большая - выводятся только f(x0,..,xk))")
        print("Верхняя диагональ f(x_0,...,x_k):",
              ", ".join(f"{dd[0][k]:.4e}" for k in range(n + 1)))
        pn = newton(xs, dd, x)
        print(f"\nP_n^N(x) (Ньютон)    = {pn:.14f}")
        if fx is not None:
            print(f"|f(x) - P_n^N(x)|    = {abs(fx - pn):.3e}")
        print(f"|P_n^L(x) - P_n^N(x)| = {abs(pl - pn):.3e}")

        again = input("\nНовые x и n (Enter), другие данные (m) или выход (q): ").strip().lower()
        if again in ("m", "q"):
            return again


def main():
    print("=" * 70)
    print("ЗАДАЧА АЛГЕБРАИЧЕСКОГО ИНТЕРПОЛИРОВАНИЯ")
    print("(формы Лагранжа и Ньютона)")
    print("=" * 70)
    while True:
        print("\nИСТОЧНИК ДАННЫХ")
        print("  1 - вариант из списка (1-18; 0 - тестовый многочлен x^3-2x+1)")
        print("  2 - своя функция f(x), свои a, b, число значений и узлы")
        print("  3 - своя таблица (узлы и значения вводятся вручную)")
        print("  q - выход")
        c = input("Выбор [1]: ").strip().lower() or "1"
        if c == "q":
            break
        if c == "1":
            v = ask("Номер варианта (1-18; 0 - тест-многочлен)", 1, int)
            while v not in VARIANTS:
                v = ask("Нет такого варианта. Номер (1-18; 0 - тест)", 1, int)
            expr, a0, b0, cnt0, n0 = VARIANTS[v]
            f = make_function(expr)
            print(f"Вариант {v}: f(x) = {expr}")
            z, fz, a, b = build_from_function(f, a0, b0, cnt0)
        elif c == "2":
            expr, f = read_function()
            n0 = ask("Желаемая степень n по умолчанию", 5, int)
            z, fz, a, b = build_from_function(f, 0.0, 1.0, 11)
        elif c == "3":
            f, n0 = None, 5
            z, fz = read_table()
            a, b = z[0], z[-1]
        else:
            print("Введите 1, 2, 3 или q.")
            continue
        print_table(z, fz, f"\nИсходная таблица ({len(z)} значений, m = {len(z) - 1}):")
        if interpolate(z, fz, f, a, b, n0) == "q":
            break


if __name__ == "__main__":
    main()
