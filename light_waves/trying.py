import time
import numpy as np
import taichi as ti
import numba

ti.init(arch=ti.gpu)

nx, ny = 1280, 720
res = nx, ny
print(res)

past_u = ti.Vector.field(3, dtype=ti.f32, shape=res)
current_u = ti.Vector.field(3, dtype=ti.f32, shape=res)
future_u = ti.Vector.field(3, dtype=ti.f32, shape=res)

accum_u = ti.Vector.field(3, dtype=ti.f32, shape=res)
kappa = ti.Vector.field(3, dtype=ti.f32, shape=res)


# константы

# левая вогнутая парабола
# правая выпуклая сфера

# размер решетки
h = 1.0  # пространственный шаг решетки
c = 1.0  # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг

acc = 0.05         # вес кадра для аккумулятора

imp_freq = 400    # "частота" для генерации нескольких волн импульса
imp_sigma = np.array([0.01, 0.13]) # размеры источника света
s_pos = np.array([-0.7, 0.])  # положение источника
s_alpha = np.radians(20.)     # направление источника
s_rot = np.array([
    [np.cos(s_alpha), -np.sin(s_alpha)],
    [np.sin(s_alpha), np.cos(s_alpha)]
])
prism_s = 3.0  # масштаб призмы
prism_pos = np.array([-0.6, 0.])  # положение 1 призмы
prism_pos_2 = np.array([0.5, 0.])  # положение 2 призмы

n = np.array([  # коэффициент преломления
    1.30, # R
    1.35, # G
    1.40  # B
])

glass_col = np.array([201., 228., 235., 30.]) / 255.  # цвет стекла
black_col = np.zeros(3)  # черный цвет
white_col = np.ones(3)   # белый цвет
red_col = np.array([1., 0., 0.])
green_col = np.array([0., 1., 0.])
blue_col = np.array([0., 0., 1.])
transp_col = np.array([0., 0., 0., 0.]) # "прозрачный" цвет


# def mix(a: float, b: float, x: float) -> float:
#     return a * x + b * (1.0 - x)

def length(p):
    '''
    Возвращает длину вектора

    :param p: np.array
              Вектор
    :return: float
             Длина вектора
    '''
    return np.sqrt(p.dot(p))

def clamp(x, low, high):
    '''
    Ограничение значение так, чтобы оно лежало между двумя другими значениями.
    clamp возвращает значение x, ограниченное диапазоном от low до high.

    :param x: float
              Входное значение
    :param low: float
                Нижняя граница значения
    :param high: float
                 Верхняя граница значения

    :return: float
             Значение x, ограниченное диапазоном от low до high.
    '''
    return np.maximum(np.minimum(x, high), low)


def smoothstep(edge0, edge1, x):
    '''
    Плавный переход (размытие) между двумя значениями

    :param edge0: float
                  Нижняя граница размытия
    :param edge1: float
                  Верхняя граница размытия
    :param x: float
              Исходное значение

    :return: float
             Вычисленное значение
    '''
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

sqrt_3 = 3 ** 0.5
isqrt_3 = 1 / sqrt_3


# def sd_equilateral_triangle(p):
#     r = np.array([abs(p[0]) - 1.0, p[1] + isqrt_3])
#     if r[0] + sqrt_3 * r[1] > 0.0:
#         r = np.array([r[0] - sqrt_3 * r[1], -sqrt_3 * r[0] - r[1]]) * 0.5
#     r[0] -= clamp(r[0], -2.0, 0.0)
#     return -(r @ r)**0.5 * np.sign(r[1])
#
# def sd_rectangle(p):
#     r = np.array([abs(p[0]) - 0.2, abs(p[1]) - 1.]) # Расстояние от точки p до центра прямоугольника
#     d = max(r[0], r[1], 0.0)  # Расстояние внутри прямоугольника (0, если точка внутри)
#     d += min(max(r[0], r[1]), 0.0)  # Добавление отрицательного расстояния за пределами прямоугольника
#     return d

def sdCircle(p, r):
    '''
    Вычисляет знаковое расстояние от точки `p` до окружности радиусом `r`.

    :param p: numpy.ndarray
              Двумерные координаты точки
    :param r: float
              Радиус окружности

    :return: float
             Знаковое расстояние от точки `p` до окружности. Знак указывает, находится ли точка внутри
               (отрицательное расстояние) или снаружи (положительное расстояние) окружности.
    '''
    return np.linalg.norm(p) - r

def rotate(pos, angle):
    '''
    Поворачивает веткор на угол angle.

    :param pos: numpy.ndarray
                Координаты вектора, которые требуется повернуть
    :param angle: float
                  Угол поворота в радианах

    :return: numpy.ndarray
             Повернутый вектор
    '''
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, pos)


def sdParabola(pos, k, angle):
    '''
    Вычисляет знаковое расстояние от точки `p` до параболы `r`.

    :param p: numpy.ndarray
                  Двумерные координаты точки
    :param k: float
                  Коэффициент параболы
    :param angle: float
                  Угол поворота в радианах на который поворачиваем параболу

    :return: float
                 Знаковое расстояние от точки `p` до параболы. Знак указывает, находится ли точка внутри
                   (положительное расстояние) или снаружи (отрицательное расстояние) параболы.
    '''
    pos_rotated = rotate(pos, angle)  # поворот координат
    pos_rotated[0] = abs(pos_rotated[0])
    ik = 1.0 / k
    p = ik * (pos_rotated[1] - 0.5 * ik) / 3.0
    q = 0.25 * ik * ik * pos_rotated[0]
    h = q * q - p * p * p
    r = np.sqrt(abs(h))
    x = np.where(h > 0.0,
                 np.power(q + r, 1.0 / 3.0) - np.power(abs(q - r), 1.0 / 3.0) * np.sign(r - q),
                 2.0 * np.cos(np.arctan(r / q) / 3.0) * np.sqrt(np.abs(p)))
    return -np.linalg.norm(pos_rotated - np.array([x, k * x * x])) * np.sign(pos_rotated[0] - x)


def prism_mask(prism1, prism2, a: float = 0.01, b: float = 0.0):
    '''
    Создает маску, содержащую линзы, c плавным переходом от 0 к 1.

    :param prism1: numpy.ndarray
                   Координаты первой линзы.
    :param prism2: numpy.ndarray
                   Координаты второй линзы.
    :param a: float
              Нижний порог значений для интерполяции.
    :param b: float
              Верхний порог значений для интерполяции.

    :return: numpy.ndarray
             Маска, содержащая линзы.

    '''
    res = np.empty((nx, ny), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            if uv[0] > -0.3 and uv[0] < 0.4:
                if uv[0] > 0:
                    d = sdCircle((uv + prism1) * prism_s, 1.)
                    res[j, i] = smoothstep(a, b, d)
                else:
                    d = sdCircle((uv + prism2) * prism_s, 1.)
                    res[j, i] = smoothstep(a, b, d)
            elif uv[1] > -0.27 and uv[1] < 0.27:
                if uv[0] < 0.5 and uv[0] > 0.3 and uv[1] != 0:
                    d = sdParabola((uv + np.array([-0.4, 0.])) * prism_s, 0.4, np.radians(90.))
                    res[j, i] = smoothstep(a, b, d)
                if uv[0] < -0.2 and uv[1] != 0:
                    d = sdParabola((uv + np.array([0.3, 0.])) * prism_s, 0.4, np.radians(270.))
                    res[j, i] = smoothstep(a, b, d)
                # elif uv[0] > -0.3:
                #     d = sdParabola((uv + np.array([0.3, 0.]) * prism_s, 3., np.radians(-90.)))
                #     res[j, i] = smoothstep(a, b, d)

            # d = sd_equilateral_triangle((uv + prism_pos) * prism_s)
            # d = sd_rectangle((uv + prism_pos) * prism_s)
            # d = sdCircle((uv + prism_pos) * prism_s, 1.)
            # d = sdParabola((uv + prism_pos) * prism_s, 1., np.radians(-90.))
            # res[j, i] = smoothstep(a, b, d)
    return res

tmask = prism_mask(prism_pos, prism_pos_2, 0.01, 0.)
kappa.from_numpy((c * dt / h) * (tmask[...,None] / n[None, None,...] + (1.0 - tmask[...,None])).astype(np.float32))


def wave_impulse(point: np.ndarray,  # (n,m,2)
                 pos: np.ndarray,
                 freq: float,  # float
                 sigma: np.ndarray,  # (2,)
                 ):
    '''
    Импульс в виде нескольких сконцентрированных волн специальной формы для уменьшения расхождения "пучка".
    Форма - синусоида по направлению x под куполом функции Гаусса в направлениях x и y.

    :param point: numpy.ndarray
                  Координаты точки, в которой вычисляется импульс волны.
    :param pos: numpy.ndarray
                Координаты положения импульса.
    :param freq: float
                 Частота волны.
    :param sigma: numpy.ndarray
                  Размах купола Гаусса по осям x и y.

    :return: float
             Амплитуда импульса в точки.
    '''
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


def impulse():
    '''
    Создает массив с импульсами волн для каждой точки.

    :return: numpy.ndarray
             Массив с импульсами волн для каждой точки.
    '''
    res = np.zeros((nx, ny))
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            res[j, i] += wave_impulse(s_rot @ uv, s_pos, imp_freq, imp_sigma)
    return res


temp_impuls = np.zeros((res[0],res[1] ,3))
temp_impuls = temp_impuls + impulse()[..., None]
past_u.from_numpy(temp_impuls.astype(np.float32))
current_u.copy_from(past_u)
future_u.copy_from(past_u)
accum_u.copy_from(past_u)


@ti.func
def propogate(x, y):
    '''
    Один шаг интегрирования уравнений распространения волны по Эйлеру

    :param x: float
              x-координата
    :param y: float
              y-координата

    :return: no return value
    '''
    future_u[x, y] = (
            kappa[x, y] ** 2 * (
            current_u[x - 1, y] +
            current_u[x + 1, y] +
            current_u[x, y - 1] +
            current_u[x, y + 1] -
            4 * current_u[x, y])
            + 2 * current_u[x, y] - past_u[x, y]
    )


@ti.func
def accumulate(x, y):
    '''
    Накопление возмущений, создаваемых волнами.

    :param x: float
              x-координата
    :param y: float
              y-координата

    :return: no return value
    '''
    accum_u[x, y] += acc * ti.abs(current_u[x, y]) * kappa[x, y] / (c * dt / h)


@ti.kernel
def render(time: ti.f32):
    '''
    Расчет всей математики и рендер изображения.

    :param time: ti.f32
                 Время с начала работы программы.

    :return: no return value
    '''
    for y in ti.ndrange(res[1]):
        # граничные условия открытой границы (y = 0)
        future_u[0, y] = current_u[1, y] + (kappa[0, y] - 1) / (kappa[0, y] + 1) * (future_u[1, y]- current_u[0, y])
        future_u[res[0]-1, y] = current_u[res[0]-2, y] + (kappa[res[0]-1, y] - 1) / (kappa[res[0]-1, y] + 1) * (future_u[res[0]-2, y] - current_u[res[0]-1, y])


    for x in ti.ndrange(res[0]):
        # граничные условия открытой границы (x = 0)
        future_u[x,0] = current_u[x,1] + (kappa[x,0] - 1) / (kappa[x,0] + 1) * (future_u[x,1] - current_u[x,0])
        future_u[x,res[1]-1] = current_u[x,res[1]-2] + (kappa[x,res[1]-1] - 1) / (kappa[x,res[1]-1] + 1) * (future_u[x,res[1]-2] - current_u[x,res[1]-1])

    for x in ti.ndrange(res[0]):
        for y in ti.ndrange(res[1]):
            past_u[x, y] = current_u[x, y]
            current_u[x, y] = future_u[x, y]

    for x in ti.ndrange(res[0]):
        for y in ti.ndrange(res[1]):
            propogate(x,y)

    for x in ti.ndrange(res[0]):
        for y in ti.ndrange(res[1]):
            accumulate(x,y)




if __name__ == "__main__":
    gui = ti.GUI("Light", res=res, fast_gui=True)
    frame = 0
    start = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        t = time.time() - start
        render(t)
        gui.set_image(accum_u)
        gui.show()
        frame += 1

    gui.close()

