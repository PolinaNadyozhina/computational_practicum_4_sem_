# -*- coding: utf-8 -*-
"""
Задание 4.2. Приближённое вычисление интеграла int_a^b f(x) dx (вес rho = 1)
по квадратурным формулам:
  1) левых прямоугольников : (b-a) f(a)                              АСТ = 0
  2) правых прямоугольников: (b-a) f(b)                              АСТ = 0
  3) средних прямоугольников: (b-a) f((a+b)/2)                       АСТ = 1
  4) трапеций              : (b-a)/2 (f(a)+f(b))                     АСТ = 1
  5) Симпсона              : (b-a)/6 (f(a)+4f((a+b)/2)+f(b))         АСТ = 3
Точное значение: формула Ньютона-Лейбница (первообразные заданы в FUNCS).
Ввод: клавиатура (Enter = значение по умолчанию). Тип чисел: float (double).
Источник данных: функция из списка (с первообразной) либо функция, заданная выражением,
с произвольными a, b. Для функции, заданной выражением, первообразной нет: значение
интеграла вводится вручную либо вычисляется эталон scipy.integrate.quad (при отсутствии
scipy - составная формула Симпсона). Эталон в выводе помечается как "эталон".
"""
import math

try:
    from scipy.integrate import quad
except ImportError:                      # без scipy: составная формула Симпсона (см. reference)
    quad = None

# номер -> (описание, f, первообразная F)
FUNCS = {
    0: ("f0 = 4  (многочлен степени 0)", lambda x: 4.0, lambda x: 4 * x),
    1: ("f1 = 3x - 2  (степень 1)", lambda x: 3 * x - 2, lambda x: 1.5 * x * x - 2 * x),
    2: ("f2 = x^2 + x + 1  (степень 2)", lambda x: x * x + x + 1,
        lambda x: x ** 3 / 3 + x * x / 2 + x),
    3: ("f3 = 2x^3 - x^2 + 5  (степень 3)", lambda x: 2 * x ** 3 - x * x + 5,
        lambda x: x ** 4 / 2 - x ** 3 / 3 + 5 * x),
    4: ("f4 = exp(x) * sin(x)  (не многочлен)", lambda x: math.exp(x) * math.sin(x),
        lambda x: math.exp(x) * (math.sin(x) - math.cos(x)) / 2),
    5: ("f5 = x^4  (степень 4)", lambda x: x ** 4,
        lambda x: x ** 5 / 5),
}

# формулы: (название, функция, АСТ)
FORMULAS = [
    ("левых прямоугольников", lambda f, a, b: (b - a) * f(a), 0),
    ("правых прямоугольников", lambda f, a, b: (b - a) * f(b), 0),
    ("средних прямоугольников", lambda f, a, b: (b - a) * f((a + b) / 2), 1),
    ("трапеций", lambda f, a, b: (b - a) / 2 * (f(a) + f(b)), 1),
    ("Симпсона", lambda f, a, b: (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b)), 3),
]


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, принято значение по умолчанию.")
        return default


# ---------------------------------------------------------------------------
# Функция, заданная выражением (синтаксис Python; степень ** или ^)
# ---------------------------------------------------------------------------
ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def make_function(expr):
    """f(x) по выражению expr (переменная x); значение приводится к float, должно быть конечным."""
    code = compile(expr, "<f>", "eval")

    def f(x):
        v = float(eval(code, {"__builtins__": {}}, dict(ENV, x=x)))
        if not math.isfinite(v):
            raise ValueError(f"значение не конечно при x = {x}")
        return v
    return f


def read_function():
    """Ввод f(x). Доступны имена модуля math: sin, cos, tan, exp, log, sqrt, atan, asin, sinh, cosh, tanh, pi, e и др."""
    print("Ввод f(x), например: exp(x)*sin(x)  или  x^3 - 2*x + 1  (умножение только *)")
    while True:
        expr = input("f(x) = ").strip().replace("^", "**")
        if not expr:
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr)
            f(0.5)                          # пробное вычисление
        except (ValueError, ZeroDivisionError, OverflowError):
            return expr, f                  # выражение корректно, 0.5 вне области определения
        except Exception as e:              # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторный ввод.")
            continue
        return expr, f


def reference(f, a, b):
    """Эталон интеграла для функции без первообразной: (значение, метка метода)."""
    if quad is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = quad(f, a, b, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
            if math.isfinite(v):
                return v, "эталон (quad)"
        except Exception:
            pass
    n = 2000                                # составная формула Симпсона, 2n отрезков
    h = (b - a) / (2 * n)
    s = f(a) + f(b)
    for k in range(1, 2 * n):
        s += (4 if k % 2 else 2) * f(a + k * h)
    return s * h / 3, "эталон (составная Симпсона, 4000 отрезков)"


def run(name, f, a, b, exact, label):
    print(f"\n{name};  [a, b] = [{a}, {b}]")
    print(f"{label}: {exact:.14f}")
    print(f"\n{'КФ':<26}{'приближённое':>20}{'абс. погрешность':>20}{'АСТ':>5}")
    for nm, formula, ast in FORMULAS:
        val = formula(f, a, b)
        err = abs(exact - val)
        mark = ""
        if err < 1e-12 * max(1.0, abs(exact)):
            mark = "  <- точна"
        print(f"{nm:<26}{val:>20.12f}{err:>20.3e}{ast:>5}{mark}")


def read_ab():
    a = ask("a", 0.0, float)
    b = ask("b", 1.0, float)
    while not (math.isfinite(a) and math.isfinite(b) and a < b):
        print("Допустим конечный промежуток a < b.")
        a = ask("a", 0.0, float)
        b = ask("b", 1.0, float)
    return a, b


def main():
    print("=" * 74)
    print("ЗАДАНИЕ 4.2. ПРОСТЕЙШИЕ КВАДРАТУРНЫЕ ФОРМУЛЫ (вес rho(x) = 1)")
    print("=" * 74)
    while True:
        print("\nИсточник данных:")
        print("  1) функция из списка (точное значение по Ньютону-Лейбницу)")
        print("  2) функция-выражение и свои a, b")
        c = input("Выбор [1], q - выход: ").strip().lower()
        if c == "q":
            break
        if c == "2":
            expr, f = read_function()
            name = f"f(x) = {expr}  (своя функция)"
            a, b = read_ab()
            try:
                for t in (a, (a + b) / 2, b):
                    f(t)                    # узлы КФ: a, (a+b)/2, b
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                print(f"Функция не определена в точке a, (a+b)/2 или b ({e}).")
                continue
            s = input("Точное значение интеграла (Enter - вычислить эталон): "
                      ).strip().replace(",", ".")
            exact = None
            if s:
                try:
                    exact = float(s)
                    if not math.isfinite(exact):
                        raise ValueError
                    label = "Точное значение (введено пользователем)"
                except ValueError:
                    print("  Некорректное число, вычисляется эталон.")
                    exact = None
            if exact is None:
                try:
                    exact, meth = reference(f, a, b)
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    print(f"Эталон не вычислен: функция не определена на отрезке ({e}).")
                    continue
                label = f"Значение интеграла - {meth}"
            run(name, f, a, b, exact, label)
        else:
            print("\nФункции:")
            for k, (nm, _, _) in FUNCS.items():
                print(f"  {k}) {nm}")
            n = ask("Номер функции", 4, int)
            while n not in FUNCS:
                n = ask("Нет функции с таким номером. Номер", 4, int)
            name, f, F = FUNCS[n]
            a, b = read_ab()
            run(name, f, a, b, F(b) - F(a), "Точное значение (Ньютон-Лейбниц)")

        c = input("\nДругие данные (Enter) или выход (q): ").strip().lower()
        if c == "q":
            break


if __name__ == "__main__":
    main()
