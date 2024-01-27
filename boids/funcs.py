import numpy as np
from numba import njit,prange


def init_boids(boids, asp, vrange = (0., 1.)):
    """
    Рандомно заполняет массив агентов (начальные координаты, скорости, ускорения)

    :param boids: np.array
        Пустой массив numpy (агенты)
    :param asp: float
        Соотношение сторон окна
    :param vrange: numpy array
        Диапазон скоростей

    :return:
    no return object
    """
    n = boids.shape[0]  # кол-во агентов
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)  # по x
    boids[:, 1] = rng.uniform(0., 1., size=n)  # по y
    v = rng.uniform(*vrange, size=n)
    a = rng.uniform(0., 2 * np.pi, size=n)
    boids[:, 2] = v * np.cos(a)  # проекция скорости на x
    boids[:, 3] = v * np.sin(a)  # проекция скорости на y


@njit(fastmath=True)
def clip_mag(arr, lims = (0., 1.)):
    """
    Ограничение длины вектора

    :param arr: np.array
        Массив скоростей агентов
    :param lims: np.array
        Диапазон скоростей

    :return:
    no return object
    """
    v = np.sqrt(np.sum(arr * arr, axis=1))
    m = v > 1e-16
    v_clip = np.clip(v, *lims)
    arr[m] *= (v_clip[m] / v[m]).reshape(-1, 1)


@njit(parallel=True)
def directions(boids):
    """
    Устанавливает направления агентов

    :param boids: np.array
        Массив агентов

    :return: np.array
        Массив агентов
    """
    return np.hstack((boids[:, :2] - boids[:, 2:4],
                      boids[:, :2]))


@njit(fastmath=True)
def propagate(boids, dt, vrange):
    """
    Расчет перемещения и изменения скоростей агентов за один временной шаг

    :param boids: np.array
        Массив агентов
    :param dt: float
        Временной шаг
    :param vrange: np.array
        Диапазон скоростей

    :return:
    no return object
    """
    boids[:, 0:2] += dt * boids[:, 2:4] + 0.5 * dt**2 * boids[:, 4:6]
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], lims=vrange)


@njit(fastmath=True)
def periodic_walls(boids, asp):
    """
    Ограничение местонахождения агентов - стена,
     при попадании в которую, агент выходит из противоположной точки окна.

    :param boids: np.array
        Массив агентов
    :param asp: float
        Соотношение сторон окна

    :return:
    no return object
    """
    boids[:, :2] %= np.array([asp, 1.])


@njit(fastmath=True)
def wall_avoidance(boids, asp):
    """
    Агенты стараются избегать стен, чтобы не врезаться в них.
    Чем ближе они к стене, тем сильнее их отталкивает от нее.

    :param boids: np.array
        Массив агентов
    :param asp: float
        Соотношение сторон окна

    :return: np.array
        Массив проекций ускорений по осям x, y
    """
    left = boids[:, 0]
    right = asp - boids[:, 0]
    bottom = boids[:, 1]
    top = 1 - boids[:, 1]

    ax = 1 / left ** 2 - 1 / right ** 2
    ay = 1 / bottom ** 2 - 1 / top ** 2

    return np.column_stack((ax, ay))


@njit(fastmath=True)
def distance(boids, dzeros):
    """
    Функция подсчета расстояния между всеми агентами

    :param boids: np.array
        Массив агентов
    :param dzeros: np.array
        Пустой массив

    :return: np.array
        Двумерный массив расстояний между каждым агентом
    """
    N = boids.shape[0]
    d = dzeros
    for i in prange(N):
        for j in prange(N):
            d[i][j] = ((boids[i][0] - boids[j][0])**2 + (boids[i][1] - boids[j][1])**2)**0.5
    return d


@njit(parallel=True, fastmath=True)
def herding(boids, perception, coeffs, asp, dzeros):
    """
    Основная функция, меняющая состояние всех агентов в каждый тик времени.

    :param boids: np.array
        Массив агентов
    :param perception: float
        Радиус, в котором ищутся соседи
    :param coeffs: np.array
        Массив коэффициентов(весов) взаимодействий
    :param asp: float
        Соотношение сторон окна
    :param dzeros: np.array
        Пустой массив

    :return:
    no return object
    """
    D = distance(boids, dzeros)
    mask = D < perception

    N = boids.shape[0]
    wa = wall_avoidance(boids, asp)

    for i in prange(N):
        accels = np.zeros((5, 2))
        accels[0] = alignment(boids, i, mask[i])
        accels[1] = cohesion(boids, i, mask[i])
        accels[2] = separation(boids, i, mask[i])

        x_rand = np.random.rand(1)
        y_rand = np.random.rand(1)
        if ((x_rand * 123) // 1) % 2 == 0:
            x_rand = -x_rand
        if ((y_rand * 123) // 1) % 2 == 0:
            y_rand = -y_rand

        accels[3] = wa[i]
        accels[4, 0], accels[4, 1] = x_rand[0], y_rand[0]

        boids[i, 4:6] = np.sum(accels * coeffs.reshape(-1, 1), axis=0)


@njit(fastmath=True)
def cohesion(boids, i, neigh_mask):
    """
    Агент стремится к медианному центру своих соседей.
    Мы выбираем такое ускорение, чтобы агент постепенно приближался к этому центру.

    :param boids: np.array
        Массив агентов
    :param i: int
        Индекс текущего агента
    :param neigh_mask: np.array
        Индексы агентов внутри круга perception

    :return: np.array
        Полученное ускорение (проекции на x и y)
    """
    N = boids[neigh_mask, ].shape[0]

    # пробовал и среднее, и медианное направления, разницы глазу не видно
    x = np.median(boids[neigh_mask, 0])
    y = np.median(boids[neigh_mask, 1])
    a = (np.array([x, y]) - boids[i, :2])

    d = boids[i, :2] - boids[neigh_mask, :2]
    dists = np.sum(d, axis=0) / N
    return a * dists  # чем ближе к остальным, тем меньше к ним стремится (чтобы не накладывались друг на друга)


@njit(fastmath=True)
def separation(boids, i, neigh_mask):
    """
    Агент выбирает среднее направление по всем агентам-соседям и движется в противоположном от него направлении.

    :param boids: np.array
        Массив агентов
    :param i: int
        Индекс текущего агента
    :param neigh_mask: np.array
        Индексы агентов внутри круга perception

    :return: np.array
        Полученное ускорение (проекции на x и y)
    """
    N = boids[neigh_mask].shape[0]
    d = boids[i, :2] - boids[neigh_mask, :2]

    a = np.sum(d, axis=0) / N
    a = a * (1 / (np.sqrt(a[0]**2 + a[1]**2) + 0.000000001))  # чем ближе к остальным, тем сильнее отталкиваются
    return a


@njit(fastmath=True)
def alignment(boids, i, neigh_mask):
    """
    Выравнивание скорости агента в соответствии с медианным направлением скоростей его агентов.

    :param boids: np.array
        Массив агентов
    :param i: int
        Индекс текущего агента
    :param neigh_mask: np.array
        Индексы агентов внутри круга perception

    :return: np.array
        Полученное ускорение (проекции на x и y)
    """
    # медианное направление лучше работает, чем среднее (меньше разлетаются в стороны при одном и том же коэффициенте)
    x = np.median(boids[neigh_mask, 2])
    y = np.median(boids[neigh_mask, 3])

    a = np.array([x, y]) - boids[i, 2:4]
    return a



