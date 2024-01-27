from numba import njit, cfunc, prange
import numpy as np
from orbits import crtbp_ode

jkw = dict(cache=True)


@njit(**jkw)
def rk7_step(f, t, s: np.ndarray, h, But: np.ndarray, b: np.ndarray, c: np.ndarray, mc: np.ndarray):
    '''
    Функция rk7_step выполняет один шаг размера h метода Рунге-Кутты 5-го порядка на основе таблицы Бутчера

    :param f: func
            функция, представляющая правую часть СОДУ (дифференциальное уравнение)
    :param t: float
            текущее время
    :param s: np.ndarray
            текущее состояние системы (вектор значений переменных)
    :param h: float
            шаг интегрирования
    :param But, b, c: np.ndarray, np.ndarray, np.ndarray
            таблица Бутчера, определяющая коэффициенты метода Рунге-Кутты
    :param mc: np.ndarray
            параметры модели

    :return: np.ndarray
            измененный вектор s после одного шага интегрирования
    '''
    sn = But.shape[0]

    k_arr = np.zeros((sn, len(s)), dtype=np.float64)
    k_arr[0] = f(t, s, mc)
    ki = np.empty_like(s)

    for i in range(1, sn):
        ki[...] = 0
        j = 0
        while j < i:
            ki += But[i,j] * k_arr[j] * h
            j += 1
        k_arr[i] = f(t + c[i]*h, s + ki, mc)

    b_end = np.zeros(len(s), dtype=np.float64)
    for i in range(sn):
        b_end += b[i]*k_arr[i]

    return s + h * b_end


@njit(**jkw)
def rk7_nsteps(f, t, s: np.ndarray, h, mc: np.ndarray, n, But: np.ndarray, b: np.ndarray, c: np.ndarray, pl: np.ndarray):
    '''
    Функция rk7_nsteps выполняет несколько шагов размера h метода Рунге-Кутты 5-го порядка

    :param f: func
            функция, представляющая правую часть системы дифференциальных уравнений
    :param t: float
            текущее время
    :param s: np.ndarray
            текущее состояние системы (вектор значений переменных)
    :param h: float
            шаг интегрированя
    :param mc: np.ndarray
            параметры модели
    :param n: int
            количество шагов
    :param But, b, c: np.ndarray, np.ndarray, np.ndarray
            таблица Бутчера, определяющая коэффициенты метода Рунге-Кутты
    :param pl: np.ndarray
            границы интегрирования

    :return: np.ndarray
            массив, содержащий состояние системы в каждый момент времени, пока происходило интегрирование
    '''
    arr = np.empty((n + 1, s.shape[0] + 1), dtype=np.float64)
    arr[:, 0] = t + h * np.arange(n + 1)
    arr[0, 1:] = s

    for i in range(n):
        arr[i + 1, 1:] = rk7_step(f,           # правая часть СОДУ
                                  arr[i, 0],   # t_0
                                  arr[i, 1:],  # s_0
                                  h,           # шаг dt
                                  But,         # таблица Бутчера
                                  b,           # вектор b
                                  c,           # вектор c
                                  mc)          # параметры модели
        x = arr[i + 1, 1]
        if x < pl[0] or x > pl[1]:
            break
    return arr[:i + 2]


@njit(**jkw)
def get_plane(vy, f, s, h, mc, n, But, b, c, pl):
    '''
    Определяет, вышел ли спутник за границы интегрирования

    :param vy: float
            вертикальная скорость спутника
    :param f: func
            функция, представляющая правую часть системы дифференциальных уравнений
    :param s: np.ndarray
            текущее состояние системы (вектор значений переменных)
    :param h: float
            шаг интегрированя
    :param mc: np.ndarray
            параметры модели
    :param n: int
            количество шагов
    :param But, b, c: np.ndarray, np.ndarray, np.ndarray
            таблица Бутчера, определяющая коэффициенты метода Рунге-Кутты
    :param pl: np.ndarray
            границы интегрирования

    :return: int
            -1, если спутник вне отрезка, 1 - если внутри
    '''
    s0 = s.copy()
    s0[4] = vy
    arr = rk7_nsteps(f, 0., s0, h, mc, n, But, b, c, pl)
    x = arr[-1, 1]
    xmean = np.mean(pl)
    return -1 if x < xmean else 1


@njit(**jkw)
def bisect_custom(func, a, b, *args):
    '''
    Реализация алгоритма бисекции

    :param func: func
            исследуемая функция
    :param a: float
            левая граница
    :param b: float
            правая граница
    :param args: *args
            аргументы для функции func

    :return:
            точка разрыва функции, либо np.nan, если ее нет
    '''
    fa = func(a, *args)
    fb = func(b, *args)
    tol = 1e-16
    max_iter = 200

    if fa * fb >= 0:
        return np.nan

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = func(c, *args)

        if abs(a - b) < tol:
            return c

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    return np.nan


@njit(**jkw)
def v0(f, s, h, mc, n, But, b, c, pl):
    '''
    Функция для расчета начальной скорости vy0 на основе метода бисекции для орбиты,
    заданной начальным положением (x0,0,z0) и условием ортогональности вектора скорости и плоскости 𝑋𝑂𝑍 в начальный момент времени

    :param f: func
            функция, представляющая правую часть системы дифференциальных уравнений
    :param s: np.ndarray
            текущее состояние системы (вектор значений переменных)
    :param h: float
            шаг интегрированя
    :param mc: np.ndarray
            параметры модели
    :param n: int
            количество шагов
    :param But, b, c: np.ndarray, np.ndarray, np.ndarray
            таблица Бутчера, определяющая коэффициенты метода Рунге-Кутты
    :param pl: np.ndarray
            границы интегрирования

    :return: float
            начальная скорость vy0
    '''
    return bisect_custom(get_plane, -1.0, 1.0, f, s, h, mc, n, But, b, c, pl)
    #return bisect(get_plane, -1.0, 1.0, args=(f, s, h, mc, n, But, b, c, pl))


@njit(**jkw)
def jac(x, y, z, v, mc):
    '''
    вычисление константы якоби

    :param x: float
            х-координата
    :param y: float
            у-координата
    :param z: float
            z-координата
    :param v: float
            скорость
    :param mc: np.ndarray
            параметры модели

    :return: float
            константа якоби
    '''
    mu2 = mc[0]
    mu1 = 1 - mu2

    r1 = ((x + mu2) ** 2 + y ** 2 + z ** 2) ** 0.5
    r2 = ((x - mu1) ** 2 + y ** 2 + z ** 2) ** 0.5
    return (x ** 2 + y ** 2) + 2 * (mu1 / r1) + 2 * (mu2 / r2) - v ** 2


@njit(parallel=True, **jkw)
def calculate(N, x_arr, z_arr, h, mc, n, But, b, c, pl):
    '''
    алгоритм вычисления начальных скоростей орбит, начальные состояния которых заданы на решетке

    :param N: int
            кол-во узлов на решетке по одной оси
    :param x_arr: np.ndarray
            границы решетки по оси х
    :param z_arr: np.ndarray
            границы решетки по оси z
    :param s: np.ndarray
            текущее состояние системы (вектор значений переменных)
    :param h: float
            шаг интегрированя
    :param mc: np.ndarray
            параметры модели
    :param n: int
            количество шагов
    :param But, b, c: np.ndarray, np.ndarray, np.ndarray
            таблица Бутчера, определяющая коэффициенты метода Рунге-Кутты
    :param pl: np.ndarray
            границы интегрирования
    :return:
    '''
    J_c = np.zeros((N, N), dtype=np.float64)
    vv = np.zeros((N, N), dtype=np.float64)
    lat_x = np.linspace(x_arr[0], x_arr[1], N)
    lat_z = np.linspace(z_arr[0], z_arr[1], N)
    for i in prange(N):
        for j in range(N):
            s_n = np.zeros(6)
            s_n[0] = lat_x[i]
            s_n[2] = lat_z[j]
            v1 = v0(crtbp_ode, s_n, h, mc, n, But, b, c, pl)
            vv[i, j] = v1
            if v1 != np.nan:
                jacobian = jac(lat_x[i], 0., lat_z[j], v1, mc)
                J_c[i, j] = jacobian
    return J_c, vv