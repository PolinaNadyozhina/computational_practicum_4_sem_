# -*- coding: utf-8 -*-
"""
Задание 4.1. Интерполяционная квадратурная формула (ИКФ)
с N узлами для веса rho(x) на (a, b):

    I(f) = int_a^b rho(x) f(x) dx  ~  sum_{k=1}^N A_k f(x_k).

Коэффициенты A_k определяются из условия точности ИКФ на одночленах
1, x, ..., x^(N-1):  sum_k A_k x_k^j = mu_j,  j = 0..N-1,
где mu_j = int_a^b x^j rho(x) dx - моменты веса (scipy.integrate.quad).
СЛАУ (матрица Вандермонда) решается numpy.linalg.solve.

Ввод с клавиатуры (Enter - значение по умолчанию). Числа - float64.
Источник данных (главное меню):
  1 - вариант из списка (вес rho, номера 1-15; f5 = sin x);
  2 - свои данные: вес rho(x) (выражение), a, b, особые точки веса,
      число узлов N, опционально функция f5(x). Моменты вычисляются scipy.quad;
      при неинтегрируемой особенности или расходимости выдаётся сообщение.
"""
import math
import warnings
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")   # подавление предупреждений quad о точности

# ---------------------------------------------------------------------------
# Варианты весов: номер -> (запись, rho(x), область определения (lo, hi), особые точки)
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


class WeightError(Exception):
    """Вес не определён, не интегрируем или моменты не вычислены."""


# ---------------------------------------------------------------------------
# Ввод выражений: синтаксис Python, ^ -> **, имена из math
# ---------------------------------------------------------------------------
ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
ENV.update(abs=abs, ln=math.log)


def make_function(expr):
    """Функция x -> float по тексту выражения; вне области определения - nan."""
    code = compile(expr, "<expr>", "eval")

    def f(x):
        try:
            return float(eval(code, {"__builtins__": {}}, dict(ENV, x=x)))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return float("nan")     # вне области определения (в т.ч. комплексный результат)
    return f


def read_expr(label, default=None):
    """Ввод выражения от x с пробным вычислением. Пустой ввод - default (если задан)."""
    while True:
        expr = input(f"{label} = " + (f"[{default}] " if default else "")).strip().replace("^", "**")
        if not expr:
            if default:
                expr = default
            else:
                print("Выражение пустое.")
                continue
        try:
            f = make_function(expr)
            f(0.5)
        except Exception as e:                 # SyntaxError, NameError, ...
            print(f"Не удалось разобрать выражение ({e}). Повторите.")
            continue
        return expr, f


def read_special_points():
    raw = input("Особые точки веса внутри (a,b) (изломы, разрывы) через пробел, Enter - нет: ")
    try:
        return [float(t) for t in raw.replace(",", ".").split()]
    except ValueError:
        print("  Некорректный ввод, особые точки не заданы.")
        return []


def check_weight(rho, a, b, pts):
    """Проверка веса на (a,b): конечные вещественные значения во внутренних точках.
    Возвращает (min, max) по выборке; бросает WeightError."""
    xs = np.linspace(a, b, 200)[1:-1]
    xs = np.array([x for x in xs if all(abs(x - p) > 1e-12 for p in pts)])   # особые точки исключаются
    vals = np.array([rho(float(x)) for x in xs])
    bad = ~np.isfinite(vals)
    if bad.any():
        raise WeightError(f"вес не определён или бесконечен в точке x = {xs[int(np.argmax(bad))]:.6g} "
                          "(внутри (a, b)); интегрируемую особенность внутри промежутка следует задать как особую точку")
    return float(vals.min()), float(vals.max())


def read_interval(lo=-math.inf, hi=math.inf):
    """Ввод конечного промежутка a < b внутри области определения [lo; hi]."""
    while True:
        a = ask("a", 0.0, float)
        b = ask("b", 1.0, float)
        if not (math.isfinite(a) and math.isfinite(b)):
            print("a и b должны быть конечными числами.")
        elif not (a < b and lo <= a and b <= hi):
            print(f"Нужно a < b и [a,b] внутри области определения веса [{lo}; {hi}].")
        else:
            return a, b


def integrate(g, a, b, pts):
    """'Точное' значение интеграла (scipy.quad) с учётом особых точек.
    Бросает WeightError, если результат не конечен или quad не сошёлся (расходимость)."""
    inner = [p for p in pts if a < p < b]
    def checked(x):                 # nan/inf в подынтегральной функции приводят к зависанию quad
        y = g(x)
        if not math.isfinite(y):
            raise WeightError(f"подынтегральная функция не определена или бесконечна при x = {x:.6g} "
                              "(особенность внутри (a, b) задаётся как особая точка)")
        return y

    res = quad(checked, a, b, points=inner or None, epsabs=1e-14, epsrel=1e-14, limit=500,
               full_output=True)
    val, err, ier = res[0], res[1], (res[3] if len(res) > 3 else 0)
    if not math.isfinite(val) or not math.isfinite(err):
        raise WeightError("интеграл не удалось вычислить (нечисловое значение)")
    if ier and err > 1e-6 * max(1.0, abs(val)):
        raise WeightError("quad не сошёлся (оценка погрешности "
                          f"{err:.1e}): вероятно, неинтегрируемая особенность, интеграл расходится")
    return val, err


def moments(rho, a, b, N, pts):
    """mu_j = int_a^b x^j rho(x) dx, j = 0..N-1; возвращает (mu, оценки погрешности quad)."""
    mu, errs = np.empty(N), np.empty(N)
    for j in range(N):
        mu[j], errs[j] = integrate(lambda x, j=j: x ** j * rho(x), a, b, pts)
    return mu, errs


def build_ikf(rho, a, b, nodes, pts):
    """Возвращает (A, mu, V, errs): коэффициенты, моменты, матрица Вандермонда, оценки quad."""
    N = len(nodes)
    mu, errs = moments(rho, a, b, N, pts)
    V = np.vander(nodes, N, increasing=True).T     # V[j, k] = x_k ** j
    A = np.linalg.solve(V, mu)
    return A, mu, V, errs


def make_functions(N, f5name="f5 = sin(x)  (из варианта)", f5=math.sin):
    """f0..f5. f4 - многочлен степени N-1 (коэффициенты 1, -2, 3, -4, ...); f5 - из варианта или своя."""
    coef = [(-1) ** k * (k + 1) for k in range(N)]
    return {
        0: ("f0 = 3  (многочлен степени 0)", lambda x: 3.0),
        1: ("f1 = 2x + 1  (степень 1)", lambda x: 2 * x + 1),
        2: ("f2 = x^2 - 3x + 2  (степень 2)", lambda x: x * x - 3 * x + 2),
        3: ("f3 = x^3 - 2x^2 + x - 1  (степень 3)", lambda x: x ** 3 - 2 * x * x + x - 1),
        4: (f"f4 = многочлен степени N-1 = {N-1}  (коэф. 1,-2,3,...)",
            lambda x: sum(c * x ** k for k, c in enumerate(coef))),
        5: (f5name, f5),
    }


def choose_nodes(a, b, N):
    print("Узлы: 1 - равноотстоящие на [a,b]; 2 - чебышёвские на [a,b]; 3 - ввести вручную")
    kind = ask("Выбор", 1, int)
    if kind == 3:
        while True:
            raw = input(f"Введите {N} различных узлов через пробел: ").replace(",", ".").split()
            try:
                x = [float(s) for s in raw]
            except ValueError:
                print("Некорректный ввод.")
                continue
            if len(x) != N:
                print(f"Нужно ровно {N} узлов.")
            elif len(set(x)) != N:
                print("Узлы должны быть попарно различны.")
            else:
                return np.array(sorted(x))
    if kind == 2:
        k = np.arange(N)
        return np.sort((a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * N)))
    if N == 1:
        return np.array([(a + b) / 2])
    return np.linspace(a, b, N)


def read_nodes_count():
    N = ask("Число узлов N (1-40)", 5, int)
    while not 1 <= N <= 40:
        N = ask("N должно быть от 1 до 40", 5, int)
    return N


def again():
    """После ошибки: True - повторный ввод параметров, False - главное меню."""
    return input("Ввести параметры заново (Enter) / в главное меню (m): ").strip().lower() != "m"


def session(rho, lo, hi, pts, own, f5name, f5):
    """Один вес: параметры -> ИКФ -> интегралы. Возвращает 'm' (меню) или 'q'."""
    while True:
        a, b = read_interval(lo, hi)
        if own:
            try:
                vmin, vmax = check_weight(rho, a, b, pts)
            except WeightError as e:
                print(f"Ошибка веса: {e}.")
                if again():
                    continue
                return "m"
            if vmin < 0:
                print("  Предупреждение: вес принимает отрицательные значения "
                      f"(min по выборке {vmin:.3g}); формула строится, но вес не знакопостоянен.")
        N = read_nodes_count()
        nodes = choose_nodes(a, b, N)
        print("Узлы x_k:", np.array2string(nodes, precision=10, max_line_width=100))

        # --- коэффициенты ИКФ
        try:
            A, mu, V, qerrs = build_ikf(rho, a, b, nodes, pts)
        except WeightError as e:
            print(f"Ошибка при вычислении моментов: {e}.")
            if again():
                continue
            return "m"
        except np.linalg.LinAlgError:
            print("Матрица Вандермонда вырождена (узлы почти совпадают). Измените узлы.")
            if again():
                continue
            return "m"
        print("\nМоменты mu_j = int x^j rho(x) dx:")
        for j in range(N):
            print(f"  mu_{j} = {mu[j]:.15f}")
        rel = float(np.max(qerrs / np.maximum(1.0, np.abs(mu))))
        if rel > 1e-9:
            print(f"  Предупреждение: оценка погрешности quad для моментов (относительная) до {rel:.1e}: "
                  "моменты могут быть неточны (особенность веса или длинный промежуток).")
        cond = np.linalg.cond(V)
        print(f"\nМатрица Вандермонда {N}x{N}, cond = {cond:.3e}")
        if cond > 1e12:
            print("  Предупреждение: матрица очень плохо обусловлена, коэффициенты A_k "
                  "могут быть неточны.")
        print(f"{'k':>3}{'x_k':>20}{'A_k':>24}")
        for k in range(N):
            print(f"{k+1:>3}{nodes[k]:>20.12f}{A[k]:>24.15f}")
        print(f"Контроль: sum A_k = {A.sum():.15f},  mu_0 = {mu[0]:.15f}")

        # --- проверка точности на одночленах и на многочлене степени N-1
        print("\nПроверка на одночленах x^j, j = 0..N-1 (точное равенство):")
        worst = 0.0
        for j in range(N):
            exact = mu[j]
            appr = float(np.sum(A * nodes ** j))
            worst = max(worst, abs(exact - appr))
        print(f"  max_j |mu_j - sum A_k x_k^j| = {worst:.3e}")
        funcs = make_functions(N, f5name, f5)
        fN1 = funcs[4][1]
        try:
            ex, _ = integrate(lambda x: rho(x) * fN1(x), a, b, pts)
            ap = float(sum(A[k] * fN1(nodes[k]) for k in range(N)))
            print(f"  многочлен f4 степени N-1={N-1}: точное {ex:.12f}, ИКФ {ap:.12f}, "
                  f"|погр.| = {abs(ex-ap):.3e}")
        except (WeightError, OverflowError) as e:
            print(f"  Проверка на f4 не выполнена: {e}.")

        # --- интегралы
        while True:
            print("\nФункции:")
            for k, (nm, _) in funcs.items():
                print(f"  {k}) {nm}")
            n = ask("Номер функции", 5, int)
            while n not in funcs:
                n = ask("Нет такой функции. Номер", 5, int)
            f = funcs[n][1]
            try:
                appr = float(sum(A[k] * f(nodes[k]) for k in range(N)))
                if not math.isfinite(appr):
                    raise WeightError("f не определена (или бесконечна) в некотором узле x_k")
                exact, qerr = integrate(lambda x: rho(x) * f(x), a, b, pts)
            except (WeightError, OverflowError, ValueError, ZeroDivisionError) as e:
                print(f"Не удалось вычислить интеграл: {e}.")
            else:
                absE = abs(exact - appr)
                relE = absE / abs(exact) if exact != 0 else float("nan")
                print(f"\nИнтеграл int_a^b rho(x) f(x) dx,  {funcs[n][0]}")
                print(f"  приближённое (ИКФ)   = {appr:.14f}")
                print(f"  'точное' (quad)      = {exact:.14f}   (оценка quad: {qerr:.1e})")
                print(f"  абсолютная погрешность = {absE:.3e}")
                print(f"  относительная погрешность = {relE:.3e}")
                if n <= 4 and n <= N - 1:
                    print("  (степень f <= N-1: ИКФ точна, погрешность ~ машинный нуль)")
            c = input("\nДругая функция (Enter) / новые параметры (p) / главное меню (m) / выход (q): "
                      ).strip().lower()
            if c in ("p", "m", "q"):
                break
        if c != "p":
            return c


def main():
    print("=" * 74)
    print("ЗАДАНИЕ 4.1. ИНТЕРПОЛЯЦИОННАЯ КВАДРАТУРНАЯ ФОРМУЛА (ИКФ)")
    print("=" * 74)
    while True:
        print("\nИсточник данных: 1 - вариант из списка (1-15); 2 - свои данные "
              "(свой вес rho(x), a, b, N); q - выход")
        src = input("Выбор [1]: ").strip().lower() or "1"
        if src == "q":
            break
        if src == "1":
            v = ask("Номер варианта (1-15)", 1, int)
            while v not in VARIANTS:
                v = ask("Нет такого варианта. Номер (1-15)", 1, int)
            name, rho, (lo, hi), pts = VARIANTS[v]
            print(f"Вариант {v}: rho(x) = {name};  f5(x) = sin(x)")
            c = session(rho, lo, hi, pts, False, "f5 = sin(x)  (из варианта)", math.sin)
        elif src == "2":
            print("Введите вес rho(x) (синтаксис Python; степень ** или ^; умножение - только *;\n"
                  "имена: sin, cos, exp, log (натуральный), sqrt, abs, pi, e, ...). Пример: x^(-0.5)")
            expr, rho = read_expr("rho(x)")
            pts = read_special_points()
            print("Своя функция f5(x) для интегрирования (Enter - sin(x)).")
            e5, f5 = read_expr("f5(x)", "sin(x)")
            print(f"Свои данные: rho(x) = {expr};  f5(x) = {e5}")
            c = session(rho, -math.inf, math.inf, pts, True, f"f5 = {e5}  (своя)", f5)
        else:
            print("Допустимый ввод: 1, 2 или q.")
            continue
        if c == "q":
            break


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван, выход.")
