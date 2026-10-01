# -*- coding: utf-8 -*-
"""
Задание 5. Численное решение задачи Коши y' = f(x, y), y(x0) = y0 на [x0, X].

Методы:
  1) Эйлера                         y_{k+1} = y_k + h f(x_k, y_k)                       порядок 1
  2) Эйлера I (средней точки)       (4): y_{k+1/2} = y_k + h/2 f;  y_{k+1} = y_k + h f(x_k+h/2, y_{k+1/2})   порядок 2
  3) Эйлера II (трапеций)           (5): Y = y_k + h f;  y_{k+1} = y_k + h/2 [f(x_k,y_k)+f(x_{k+1},Y)]       порядок 2
  4) неявный Эйлера                 y_{k+1} = y_k + h f(x_{k+1}, y_{k+1})  (простая итерация)                порядок 1
  5) Рунге-Кутты 4-го порядка       k1..k4, y_{k+1} = y_k + (k1+2k2+2k3+k4)/6                               порядок 4
  6) экстраполяционный Адамса (ЭМА) 4-го порядка, разностная форма:
        y_{n+1} = y_n + eta_n + 1/2 D eta_{n-1} + 5/12 D^2 eta_{n-2} + 3/8 D^3 eta_{n-3} + 251/720 D^4 eta_{n-4}
  7) интерполяционный Адамса (ИМА) 4-го порядка, схема predictor-corrector:
        y_{n+1} = y_n + eta_{n+1} - 1/2 D eta_n - 1/12 D^2 eta_{n-1} - 1/24 D^3 eta_{n-2} - 19/720 D^4 eta_{n-3}
     (eta_j = h f(x_j, y_j); предиктор - ЭМА, корректор - простая итерация до |y^(s+1) - y^(s)| < eps).
Методы Адамса: y_1..y_4 вычисляются методом Рунге-Кутты 4-го порядка.

Ввод с клавиатуры (Enter - значение по умолчанию). Тип чисел - float (double).
Источник данных: задача из списка (с точным решением) или своя задача: f(x, y) вводится выражением,
задаются x0, y0, X, n; точное решение y(x) вводится выражением, при его отсутствии сравнение
выполняется с эталоном (scipy solve_ivp, rtol = atol = 1e-13; без scipy - РК4 с шагом h/64).
"""
import math

# номер -> (описание, f(x,y), x0, y0, X, точное решение y(x))
PROBLEMS = {
    1: ("y' = 1 + y^2, y(0) = 0  [точное: tg x]", lambda x, y: 1 + y * y, 0.0, 0.0, 1.0, math.tan),
    2: ("y' = y, y(0) = 1  [точное: e^x]", lambda x, y: y, 0.0, 1.0, 1.0, math.exp),
    3: ("y' = x + y, y(0) = 1  [точное: 2e^x - x - 1]", lambda x, y: x + y, 0.0, 1.0, 1.0,
        lambda x: 2 * math.exp(x) - x - 1),
    4: ("y' = -y + sin x, y(0) = 1  [точное: (sin x - cos x)/2 + 1.5 e^{-x}]",
        lambda x, y: -y + math.sin(x), 0.0, 1.0, 5.0,
        lambda x: (math.sin(x) - math.cos(x)) / 2 + 1.5 * math.exp(-x)),
}

EPS_ITER = 1e-12      # точность корректора и неявного метода (простая итерация)
MAX_ITER = 100        # предел числа итераций


def ask(prompt, default, cast):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    try:
        return cast(s) if s else default
    except ValueError:
        print("  Некорректный ввод, использовано значение по умолчанию.")
        return default


def ask_finite(prompt, default):
    """Число float; nan и inf не принимаются (повторный запрос)."""
    v = ask(prompt, default, float)
    while not math.isfinite(v):
        print("  Требуется конечное число.")
        v = ask(prompt, default, float)
    return v


# ---------------------------------------------------------------------------
# Своя задача: правая часть f(x, y) и (необязательно) точное решение y(x) вводятся выражением
# ---------------------------------------------------------------------------
MAX_N = 200000        # предел числа шагов


def _env():
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    env.update(abs=abs, min=min, max=max)
    return env


ENV = _env()


def make_function(expr, names=("x",)):
    """Функция от переменных names по выражению expr (синтаксис Python, имена из math)."""
    code = compile(expr, "<f>", "eval")
    return lambda *a: float(eval(code, {"__builtins__": {}}, dict(ENV, **dict(zip(names, a)))))


def read_expr(title, names, example, allow_empty=False):
    """Ввод выражения (степень ** или ^, умножение - только *); проверка пробным вычислением.
    Пустая строка допустима при allow_empty (возвращается None)."""
    print(f"Введите {title}, например: {example}")
    print("(умножение - знак *, степень - ** или ^; имена: sin, cos, tan, exp, log, sqrt, atan, pi, e, ...)")
    while True:
        expr = input(f"{title} = ").strip().replace("^", "**")
        if not expr:
            if allow_empty:
                return None, None
            print("Выражение пустое.")
            continue
        try:
            f = make_function(expr, names)
            f(*([0.5] * len(names)))
        except (ValueError, ZeroDivisionError, OverflowError):
            return expr, f                 # пробная точка 0.5 вне области определения
        except TypeError as e:
            if "complex" in str(e):
                return expr, f             # комплексное значение в пробной точке
            print(f"Не удалось разобрать выражение ({e}). Повторный ввод.")
            continue
        except Exception as e:             # SyntaxError, NameError, TypeError ...
            print(f"Не удалось разобрать выражение ({e}). Повторный ввод.")
            continue
        return expr, f


def build_reference(f, x0, y0, h, n):
    """Эталонное решение для задачи без точного: значения в узлах сетки шага h/4
    (сетки сводки h, h/2, h/4 - её подсетки). scipy solve_ivp (DOP853, rtol = atol = 1e-13),
    при отсутствии scipy - РК4 с шагом h/64. Возвращает (функция y_эт(x), описание)."""
    m = 4 * n
    hh = h / 4
    vals = None
    try:
        from scipy.integrate import solve_ivp
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calls = [0]

            def rhs(t, y):
                calls[0] += 1
                if calls[0] > 3000000:
                    raise RuntimeError("эталон не построен за допустимое число вычислений f")
                return [f(t, y[0])]
            sol = solve_ivp(rhs, (x0, x0 + m * hh), [y0], method="DOP853",
                            rtol=1e-13, atol=1e-13, t_eval=[x0 + j * hh for j in range(m + 1)])
        if not sol.success or len(sol.t) != m + 1:
            raise RuntimeError("решение уходит в бесконечность или не определено на [x0, X]")
        vals = [float(v) for v in sol.y[0]]
        how = "scipy solve_ivp (DOP853, rtol = atol = 1e-13)"
    except ImportError:
        pass
    except (OverflowError, ZeroDivisionError, ValueError, TypeError):
        raise RuntimeError("переполнение или f(x, y) не определена на траектории (решение уходит в бесконечность)")
    if vals is None:
        r = 16                                                 # подшагов РК4 на шаг h/4
        try:
            ys = run_onestep(step_rk4, f, x0, y0, hh / r, m * r)
        except (OverflowError, ZeroDivisionError, ValueError):
            raise RuntimeError("переполнение или f(x, y) не определена на траектории (решение уходит в бесконечность)")
        vals = ys[::r]
        how = "Рунге-Кутты 4 с шагом h/64"
    if not all(math.isfinite(v) for v in vals):
        raise RuntimeError("эталонное решение не конечно (решение уходит в бесконечность)")

    def ref(x):
        return vals[round((x - x0) / hh)]
    return ref, how


def safe_run(run, f, x0, y0, h, n):
    """Запуск метода; сбои вычислений и расходимость преобразуются в RuntimeError."""
    try:
        ys = run(f, x0, y0, h, n)
    except OverflowError:
        raise RuntimeError("переполнение (решение растёт слишком быстро / расходится)")
    except ZeroDivisionError:
        raise RuntimeError("деление на ноль в f(x, y)")
    except ValueError as e:
        raise RuntimeError(f"f(x, y) не определена на траектории ({e})")
    except TypeError as e:
        raise RuntimeError(f"f(x, y) дала недопустимое значение ({e})")
    if not all(math.isfinite(v) for v in ys):
        raise RuntimeError("решение расходится (nan/inf)")
    return ys


# ---------------------------------------------------------------------------
# Одношаговые методы: step(f, x, y, h) -> y_{k+1}
# ---------------------------------------------------------------------------
def step_euler(f, x, y, h):
    return y + h * f(x, y)


def step_euler1(f, x, y, h):
    ym = y + h / 2 * f(x, y)
    return y + h * f(x + h / 2, ym)


def step_euler2(f, x, y, h):
    Y = y + h * f(x, y)
    return y + h / 2 * (f(x, y) + f(x + h, Y))


def step_implicit(f, x, y, h):
    """y_{k+1} = y + h f(x+h, y_{k+1}); начальное приближение - явный Эйлер."""
    z = y + h * f(x, y)
    for _ in range(MAX_ITER):
        z_new = y + h * f(x + h, z)
        if abs(z_new - z) < EPS_ITER:
            return z_new
        z = z_new
    raise RuntimeError("неявный метод Эйлера: простая итерация не сошлась")


def step_rk4(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h / 2, y + k2 / 2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def run_onestep(step, f, x0, y0, h, n):
    ys = [y0]
    for k in range(n):
        ys.append(step(f, x0 + k * h, ys[-1], h))
    return ys


# ---------------------------------------------------------------------------
# Методы Адамса (разностная форма), 4-й порядок. Требуются y_0..y_4 (старт - РК4).
# ---------------------------------------------------------------------------
def backward_diffs(eta):
    """eta = [..., eta_{j-4}, ..., eta_j] (5 значений). Возвращает
    [eta_j, nabla eta_j, nabla^2 eta_j, nabla^3 eta_j, nabla^4 eta_j]
    (обратные разности, заканчивающиеся в eta_j)."""
    d = list(eta)
    res = [d[-1]]
    for _ in range(4):
        d = [d[i + 1] - d[i] for i in range(len(d) - 1)]
        res.append(d[-1])
    return res


def ema_step(eta5, yn):
    """ЭМА: eta5 = [eta_{n-4},...,eta_n]; обратные разности в точке n."""
    e0, d1, d2, d3, d4 = backward_diffs(eta5)
    return yn + e0 + d1 / 2 + 5 * d2 / 12 + 3 * d3 / 8 + 251 * d4 / 720


def ima_step(f, h, xs_next, eta4, yn, y_pred):
    """ИМА: eta4 = [eta_{n-3},...,eta_n]; xs_next = x_{n+1}; y_pred - предиктор (ЭМА).
    Корректор: простая итерация. Возвращает (y_{n+1}, число итераций)."""
    y = y_pred
    for s in range(1, MAX_ITER + 1):
        eta5 = eta4 + [h * f(xs_next, y)]           # [eta_{n-3},...,eta_n, eta_{n+1}]
        e1, d1, d2, d3, d4 = backward_diffs(eta5)   # разности, заканчивающиеся в n+1
        y_new = yn + e1 - d1 / 2 - d2 / 12 - d3 / 24 - 19 * d4 / 720
        if abs(y_new - y) < EPS_ITER:
            return y_new, s
        y = y_new
    raise RuntimeError("ИМА: итерации корректора не сошлись")


def run_adams(f, x0, y0, h, n, mode):
    """mode = 'ema' или 'ima'. Возвращает список y_0..y_n."""
    ys = run_onestep(step_rk4, f, x0, y0, h, min(4, n))      # y_0..y_4 методом РК4
    if n <= 4:
        return ys
    eta = [h * f(x0 + k * h, ys[k]) for k in range(5)]
    for k in range(4, n):                                    # вычисление y_{k+1}
        y_pred = ema_step(eta[-5:], ys[k])
        if mode == "ema":
            y_next = y_pred
        else:
            y_next, _ = ima_step(f, h, x0 + (k + 1) * h, eta[-4:], ys[k], y_pred)
        ys.append(y_next)
        eta.append(h * f(x0 + (k + 1) * h, y_next))
    return ys


METHODS = [
    ("Эйлера", lambda f, x0, y0, h, n: run_onestep(step_euler, f, x0, y0, h, n), 1),
    ("Эйлера I (ср. точки)", lambda f, x0, y0, h, n: run_onestep(step_euler1, f, x0, y0, h, n), 2),
    ("Эйлера II (трапеций)", lambda f, x0, y0, h, n: run_onestep(step_euler2, f, x0, y0, h, n), 2),
    ("неявный Эйлера", lambda f, x0, y0, h, n: run_onestep(step_implicit, f, x0, y0, h, n), 1),
    ("Рунге-Кутты 4", lambda f, x0, y0, h, n: run_onestep(step_rk4, f, x0, y0, h, n), 4),
    ("Адамса экстрапол. (ЭМА4)", lambda f, x0, y0, h, n: run_adams(f, x0, y0, h, n, "ema"), 5),
    ("Адамса интерполяц. (ИМА4)", lambda f, x0, y0, h, n: run_adams(f, x0, y0, h, n, "ima"), 5),
]


def max_err(ys, x0, h, exact):
    return max(abs(exact(x0 + k * h) - y) for k, y in enumerate(ys))


def read_variant():
    print("\nЗадачи:")
    for k, v in PROBLEMS.items():
        print(f"  {k}) {v[0]}")
    p = ask("Номер задачи", 1, int)
    while p not in PROBLEMS:
        p = ask("Нет такой задачи. Номер", 1, int)
    return PROBLEMS[p]


def read_own():
    """Своя задача: f(x, y) и, по желанию, точное решение. Возвращает (name, f, x0d, y0d, Xd, exact)."""
    expr, f = read_expr("f(x, y)", ("x", "y"), "y - x^2 + 1  или  sin(x)*y + x")
    exact = None
    print("Точное решение y(x) (Enter - неизвестно, сравнение с эталоном).")
    e_expr, exact = read_expr("y(x)", ("x",), "tan(x)  или  2*exp(x) - x - 1", allow_empty=True)
    name = f"y' = {expr}  [своя задача" + (f"; точное y(x) = {e_expr}]" if exact else "; точное неизвестно]")
    return name, f, 0.0, 1.0, 1.0, exact


def main():
    print("=" * 78)
    print("ЗАДАНИЕ 5. ЧИСЛЕННЫЕ МЕТОДЫ РЕШЕНИЯ ЗАДАЧИ КОШИ ДЛЯ ОДУ")
    print("=" * 78)
    while True:
        print("\nИсточник данных:")
        print("  1) задача из списка")
        print("  2) своя задача: f(x, y) вводится выражением, свои x0, y0, X, n")
        c = input("Выбор [1], q - выход: ").strip().lower()
        if c == "q":
            break
        if c not in ("", "1", "2"):
            print("  Допустимо: 1, 2, q.")
            continue
        own = c == "2"
        name, f, x0d, y0d, Xd, exact = read_own() if own else read_variant()
        x0 = ask_finite("x0", x0d)
        y0 = ask_finite("y0", y0d)
        X = ask_finite("Правый конец X", Xd)
        Xr = Xd if Xd > x0 else x0 + 1.0                 # значение по умолчанию при повторном вводе
        while X <= x0:
            X = ask_finite("Требуется X > x0", Xr)
        n = ask("Число шагов n (шаг h = (X-x0)/n), n >= 5", 10, int)
        while n < 5 or n > MAX_N:
            n = ask(f"Требуется 5 <= n <= {MAX_N}", 10, int)
        h = (X - x0) / n
        print(f"\n{name}\nx0 = {x0}, y0 = {y0}, X = {X}, n = {n}, h = {h:.6e}")

        try:
            f(x0, y0)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            print(f"f(x, y) не определена в начальной точке ({e}). Данные отклонены.")
            continue

        # --- точное решение или эталон
        label = "точное"
        if own and exact is None:
            label = "эталон"
            try:
                exact, how = build_reference(f, x0, y0, h, n)
            except Exception as e:
                print(f"Не удалось построить эталонное решение: {e}. Данные отклонены.")
                continue
            print(f"Точное решение неизвестно: сравнение с эталоном - {how}.")
        elif own:
            try:
                if abs(exact(x0) - y0) > 1e-9 * max(1.0, abs(y0)):
                    print(f"Введённое y(x) не удовлетворяет y(x0) = y0 (y({x0}) = {exact(x0)}).")
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        try:
            ex = [exact(x0 + k * h) for k in range(n + 1)]
            if not all(math.isfinite(v) for v in ex):
                raise ValueError("значение не конечно")
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            print(f"y(x) не определено на [x0, X] ({e}). Данные отклонены.")
            continue

        sols = []
        for nm, run, _ in METHODS:
            try:
                sols.append(safe_run(run, f, x0, y0, h, n))
            except RuntimeError as e:
                print(f"Ошибка, метод '{nm}': {e}")
                sols.append(None)
        if all(s is None for s in sols):
            print("Ни один метод не дал решения. Данные отклонены.")
            continue
        show = ask("Печатать таблицу решения? (1 - да, 0 - нет)", 1, int)
        if show:
            hdr = f"{'x_k':>9}{label:>15}" + "".join(f"{'y('+str(i+1)+')':>15}" for i in range(len(METHODS)))
            print("\nМетоды: " + "; ".join(f"({i+1}) {m[0]}" for i, m in enumerate(METHODS)))
            dash = f"{'-':>15}"
            print(hdr)
            for k in range(n + 1):
                xk = x0 + k * h
                print(f"{xk:>9.4f}{ex[k]:>15.9f}" + "".join(dash if s is None else f"{s[k]:>15.9f}" for s in sols))
            print("\nАбсолютная погрешность в точках" + (" (относительно эталона):" if label == "эталон" else ":"))
            print(f"{'x_k':>9}" + "".join(f"{'(' + str(i+1) + ')':>15}" for i in range(len(METHODS))))
            for k in range(n + 1):
                xk = x0 + k * h
                print(f"{xk:>9.4f}" + "".join(dash if s is None else f"{abs(ex[k]-s[k]):>15.3e}" for s in sols))

        # --- сводка по шагам h, h/2, h/4 и оценка порядка
        if label == "эталон":
            print("\nСВОДКА: max|y_эт(x_k) - y_k| при шагах h, h/2, h/4 (y_эт - эталон, не точное решение) и наблюдаемый порядок")
        else:
            print("\nСВОДКА: max|y(x_k) - y_k| при шагах h, h/2, h/4 и наблюдаемый порядок")
        print(f"{'метод':<28}{'h':>12}{'h/2':>12}{'h/4':>12}{'порядок':>10}{'теор.':>7}")
        for nm, run, order in METHODS:
            try:
                errs = []
                for q in (1, 2, 4):
                    hh = h / q
                    errs.append(max_err(safe_run(run, f, x0, y0, hh, n * q), x0, hh, exact))
                if errs[1] > 0 and errs[2] > 0:
                    obs = math.log2(errs[1] / errs[2])
                    obs = f"{obs:>10.2f}"
                else:
                    obs = f"{'-':>10}"
                print(f"{nm:<28}{errs[0]:>12.3e}{errs[1]:>12.3e}{errs[2]:>12.3e}{obs}{order:>7}")
            except RuntimeError as e:
                print(f"{nm:<28} {e}")
        if label == "эталон":
            print("Порядок относительно эталона не определяется при ошибке метода, сравнимой с точностью эталона (~1e-12).")

        if input("\nДругая задача/параметры (Enter) или выход (q): ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
