# -*- coding: utf-8 -*-
"""
Задание 4.3. Составные квадратурные формулы (СКФ) и уточнение по Рунге-Ромбергу.
Вес rho(x) = 1. Интеграл J = int_A^B f(x) dx; h = (B-A)/m.

СКФ:                                                          порядок r (для Рунге)
  левых прямоугольников : h * sum_{j=0}^{m-1} f(A + j h)                1
  правых прямоугольников: h * sum_{j=0}^{m-1} f(A + (j+1) h)            1
  средних прямоугольников: h * sum f(A + (j+1/2) h)                     2
  трапеций              : h/2 (f(A) + 2 sum_{j=1}^{m-1} f(y_j) + f(B))  2
  Симпсона              : h/6 (f(A) + 2 sum_{j=1}^{m-1} f(y_j)
                               + 4 sum f(y_j + h/2) + f(B))             4

Уточнение по Рунге: J_R = (l^r J(h/l) - J(h)) / (l^r - 1)  (l = 2: (2^r J(h/2) - J(h)) / (2^r - 1)).
Точное значение J: первообразная (формула Ньютона-Лейбница).
Источник данных: функция из списка либо функция, заданная выражением (вычисляется
на массивах numpy). Для функции-выражения первообразной нет: значение J вводится вручную
либо вычисляется эталон scipy.integrate.quad (при сбое - формула Симпсона, m = 10^6).
Ввод: клавиатура (Enter = значение по умолчанию). Тип чисел: float64 (numpy).
"""
import math
import numpy as np

# номер -> (описание, f (векторная), первообразная F, степень многочлена или None)
FUNCS = {
    0: ("f0 = 4  (степень 0)", lambda x: 4.0 + 0 * x, lambda x: 4 * x, 0),
    1: ("f1 = 3x - 2  (степень 1)", lambda x: 3 * x - 2, lambda x: 1.5 * x**2 - 2 * x, 1),
    2: ("f2 = x^2 + x + 1  (степень 2)", lambda x: x**2 + x + 1,
        lambda x: x**3 / 3 + x**2 / 2 + x, 2),
    3: ("f3 = 2x^3 - x^2 + 5  (степень 3)", lambda x: 2 * x**3 - x**2 + 5,
        lambda x: x**4 / 2 - x**3 / 3 + 5 * x, 3),
    4: ("f4 = exp(x) * sin(x)  (не многочлен)", lambda x: np.exp(x) * np.sin(x),
        lambda x: np.exp(x) * (np.sin(x) - np.cos(x)) / 2, None),
    5: ("f5 = 1.27 x^5 + 2.04 x  (степень 5)",
        lambda x: 1.27 * x**5 + 2.04 * x, lambda x: 1.27 * x**6 / 6 + 1.02 * x**2, 5),
    6: ("f6 = exp(x)  (быстрый рост)",
        lambda x: np.exp(x), lambda x: np.exp(x), None),
}


# ---------------------------------------------------------------------------
# Составные квадратурные формулы
# ---------------------------------------------------------------------------
def left_rect(f, A, B, m):
    h = (B - A) / m
    return h * np.sum(f(A + h * np.arange(m)))


def right_rect(f, A, B, m):
    h = (B - A) / m
    return h * np.sum(f(A + h * np.arange(1, m + 1)))


def mid_rect(f, A, B, m):
    h = (B - A) / m
    return h * np.sum(f(A + h * (np.arange(m) + 0.5)))


def trapezoid(f, A, B, m):
    h = (B - A) / m
    inner = np.sum(f(A + h * np.arange(1, m)))
    return h / 2 * (f(np.float64(A)) + 2 * inner + f(np.float64(B)))


def simpson(f, A, B, m):
    h = (B - A) / m
    inner = np.sum(f(A + h * np.arange(1, m)))
    mids = np.sum(f(A + h * (np.arange(m) + 0.5)))
    return h / 6 * (f(np.float64(A)) + 2 * inner + 4 * mids + f(np.float64(B)))


# название, функция, порядок r (для Рунге), число значений f через m
FORMULAS = [
    ("левых прямоугольников", left_rect, 1, "m"),
    ("правых прямоугольников", right_rect, 1, "m"),
    ("средних прямоугольников", mid_rect, 2, "m"),
    ("трапеций", trapezoid, 2, "m+1"),
    ("Симпсона", simpson, 4, "2m+1"),
]


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, принято значение по умолчанию.")
        return default


# ---------------------------------------------------------------------------
# Функция, заданная выражением; вычисляется на массивах numpy
# ---------------------------------------------------------------------------
M_MAX = 10**7                                     # максимальное число промежутков


def make_function(expr):
    """Векторная f(x) из выражения (имена numpy: sin, cos, exp, log, sqrt, ...);
    результат приводится к форме x (константа даёт массив той же формы)."""
    env = {k: getattr(np, k) for k in
           ("sin", "cos", "tan", "exp", "log", "log2", "log10", "sqrt", "sinh", "cosh",
            "tanh", "arctan", "arcsin", "arccos", "fabs", "floor", "ceil", "pi", "e")}
    env.update(atan=np.arctan, asin=np.arcsin, acos=np.arccos, abs=np.abs, ln=np.log)
    code = compile(expr, "<f>", "eval")

    def f(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            y = eval(code, {"__builtins__": {}}, dict(env, x=x))
            return np.array(np.broadcast_to(np.asarray(y, dtype=float), x.shape))
    return f


def read_function():
    """Ввод f(x) (синтаксис Python; степень ** или ^; умножение только *).
    Имена: sin, cos, tan, exp, log, sqrt, atan, asin, acos, sinh, cosh, tanh, abs, pi, e."""
    print("Ввод f(x), например: x**3 - 2*x + 1  или  exp(x)*sin(x)")
    print("(умножение - *, степень - ** или ^, логарифм - log(x))")
    while True:
        expr = input("f(x) = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr)
            f(np.array([0.5, 1.0]))
        except Exception as e:             # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторный ввод.")
            continue
        return expr, f


def nodes_ok(f, A, B, m):
    """Проверка конечности f во всех узлах и серединах (2m+1 точек); при отказе - сообщение."""
    x = A + (B - A) * np.arange(2 * m + 1) / (2 * m)
    y = f(x)
    bad = ~np.isfinite(y)
    if bad.any():
        print(f"  Функция не определена или бесконечна в узле x = {x[bad][0]:.6g} "
              f"(всего таких узлов: {int(bad.sum())}).")
        return False
    return True


def reference(f, A, B):
    """Эталон интеграла для функции-выражения: quad, иначе Симпсон при m = 10^6."""
    try:
        from scipy.integrate import quad
        val = quad(lambda t: float(f(np.float64(t))), A, B, limit=500)[0]
        if math.isfinite(val):
            return val, "эталон (quad)"
    except Exception:
        pass
    return float(simpson(f, A, B, 10**6)), "эталон (Симпсон, m = 10^6)"


def exact_value(f, F, A, B):
    """Возвращает (J, подпись): Ньютон-Лейбниц для функции из списка; для функции-выражения -
    введённое значение или эталон."""
    if F is not None:
        return float(F(np.float64(B)) - F(np.float64(A))), "Точное значение J (Ньютон-Лейбниц)"
    s = input("Точное значение интеграла (Enter - вычислить эталон): ").strip().replace(",", ".")
    if s:
        try:
            J = float(s)
            if math.isfinite(J):
                return J, "Точное значение J (введено)"
        except ValueError:
            pass
        print("  Некорректное число, вычисляется эталон.")
    return reference(f, A, B)


def rel(err, J):
    return err / abs(J) if J != 0 else float("nan")


def show_table(f, A, B, m, J, label="Точное значение J (Ньютон-Лейбниц)"):
    h = (B - A) / m
    print(f"\nA = {A}, B = {B}, m = {m}, h = (B-A)/m = {h:.6e}")
    print(f"{label} = {J:.14e}")
    print(f"{'СКФ':<26}{'J(h)':>24}{'|J-J(h)|':>14}{'отн. погр.':>14}{'зн. f':>7}")
    res = {}
    for nm, fn, r, nv in FORMULAS:
        Jh = float(fn(f, A, B, m))
        err = abs(J - Jh)
        res[nm] = Jh
        rs = f"{rel(err, J):.3e}" if J != 0 else "  (J=0)"
        print(f"{nm:<26}{Jh:>24.14e}{err:>14.3e}{rs:>14}{nv:>7}")
    return res


def runge_stage(f, A, B, m, J, res_h):
    l = ask("Множитель l числа промежутков m (натуральное l >= 2)", 2, int)
    while l < 2:
        l = ask("Требуется l >= 2", 2, int)
    m2 = m * l
    if m2 > M_MAX:
        print(f"  m*l = {m2} превышает {M_MAX}; уточнение не выполнено.")
        return
    if not nodes_ok(f, A, B, m2):
        return
    print(f"\nНовое число промежутков m*l = {m2}, шаг h/l = {(B - A) / m2:.6e}")
    print(f"{'СКФ':<26}{'J(h)':>22}{'|J-J(h)|':>11}{'J(h/l)':>22}{'|J-J(h/l)|':>12}"
          f"{'J_R':>22}{'|J-J_R|':>11}{'r':>3}")
    for nm, fn, r, _ in FORMULAS:
        Jh = res_h[nm]
        Jl = float(fn(f, A, B, m2))
        JR = (l**r * Jl - Jh) / (l**r - 1)
        print(f"{nm:<26}{Jh:>22.13e}{abs(J-Jh):>11.2e}{Jl:>22.13e}{abs(J-Jl):>12.2e}"
              f"{JR:>22.13e}{abs(J-JR):>11.2e}{r:>3}")
    print("Относительные погрешности: ")
    for nm, fn, r, _ in FORMULAS:
        Jh = res_h[nm]
        Jl = float(fn(f, A, B, m2))
        JR = (l**r * Jl - Jh) / (l**r - 1)
        print(f"  {nm:<26} J(h): {rel(abs(J-Jh), J):.3e}  J(h/l): {rel(abs(J-Jl), J):.3e}"
              f"  J_R: {rel(abs(J-JR), J):.3e}")


def poly_test():
    """Тест СКФ на многочленах: погрешность ~ машинный нуль при степени <= АСТ (0, 0, 1, 1, 3)."""
    print("\nТЕСТ НА МНОГОЧЛЕНАХ  (A=-1, B=2, m=7)")
    A, B, m = -1.0, 2.0, 7
    print(f"{'СКФ':<26}" + "".join(f"{'f'+str(k):>12}" for k in range(4)))
    for nm, fn, r, _ in FORMULAS:
        row = ""
        for k in range(4):
            _, f, F, _ = FUNCS[k]
            J = float(F(B) - F(A))
            row += f"{abs(J - float(fn(f, A, B, m))):>12.2e}"
        print(f"{nm:<26}{row}")
    print("АСТ: левых 0, правых 0, средних 1, трапеций 1, Симпсона 3")


def read_params(A, B, m):
    """Ввод A, B, m: конечные A < B, 1 <= m <= M_MAX."""
    A = ask("A", A, float)
    B = ask("B", B, float)
    while not (math.isfinite(A) and math.isfinite(B) and A < B):
        print("Допустимы конечные A < B.")
        A = ask("A", 0.0, float)
        B = ask("B", 1.0, float)
    m = ask("Число промежутков m (натуральное)", m, int)
    while not 1 <= m <= M_MAX:
        m = ask(f"Допустимо 1 <= m <= {M_MAX}", 10, int)
    return A, B, m


def compute(f, F, A, B, m):
    """Точное значение и таблица СКФ; None, если f не определена в узлах."""
    if not nodes_ok(f, A, B, m):
        return None
    J, label = exact_value(f, F, A, B)
    return J, show_table(f, A, B, m, J, label)


def main():
    print("=" * 78)
    print("ЗАДАНИЕ 4.3. СОСТАВНЫЕ КВАДРАТУРНЫЕ ФОРМУЛЫ. УТОЧНЕНИЕ ПО РУНГЕ-РОМБЕРГУ")
    print("=" * 78)
    n = 4
    A, B, m = 0.0, 1.0, 10
    while True:
        print("\nИСТОЧНИК ДАННЫХ")
        print("  1 - функция из списка (0-6)")
        print("  2 - функция-выражение f(x) (свои A, B, m; точное значение необязательно)")
        print("  t - тест всех СКФ на многочленах")
        print("  q - выход")
        s = input("Выбор [1]: ").strip().lower() or "1"
        if s == "q":
            break
        if s == "t":
            poly_test()
            continue
        if s == "1":
            print("Функции:")
            for k, v in FUNCS.items():
                print(f"  {k}) {v[0]}")
            n = ask("Номер функции", n, int)
            while n not in FUNCS:
                n = ask("Нет функции с таким номером. Номер (0-6)", 4, int)
            name, f, F, deg = FUNCS[n]
            print(f"Выбрано: {name}")
        elif s == "2":
            expr, f = read_function()
            F = None
            print(f"Выбрано: f(x) = {expr}")
        else:
            print("Допустимый ввод: 1, 2, t, q.")
            continue

        while True:
            A, B, m = read_params(A, B, m)
            out = compute(f, F, A, B, m)
            if out is not None:
                break
        J, res = out

        while True:
            c = input("\n1 - уточнение по Рунге; 2 - другие A, B, m; 3 - другая функция; q - выход [1]: ").strip().lower()
            if c in ("", "1"):
                runge_stage(f, A, B, m, J, res)
            elif c == "2":
                A, B, m = read_params(A, B, m)
                out = compute(f, F, A, B, m)
                if out is not None:
                    J, res = out
            else:
                break
        if c == "q":
            break


if __name__ == "__main__":
    main()
