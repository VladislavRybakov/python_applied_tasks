import taichi as ti
import taichi_glsl as ts
import time

ti.init(arch=ti.gpu)

#   !!!!!!!!!!!!!!!!!!!2400 символов!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@ti.func
def hash21(p):
    '''
    Хэш-функция, получающая на вход два числа (вектор), а возвращающая одно

    :param p: ts.vec2
              Входной вектор

    :return: float
             Число с плавающей точкой, полученное в результате хэш-функции
    '''
    q = ts.fract(p * ts.vec2(123.34, 345.56))
    q += q @ (q + 34.23)
    return ts.fract(q.x * q.y)


@ti.func
def hash22(p):
    '''
    Хэш-функция, получающая на вход два числа (вектор), и возвращающая два (вектор)

    :param p: ts.vec2
              Входной вектор

    :return: ts.vec2
             Вектор, полученный в результате хэш-функции
    '''
    x = hash21(p)
    y = hash21(p + x)
    return ts.vec2(x, y)


@ti.func
def normalize(p):
    '''
    Функция нормирования вектора

    :param p: ts.vec
              Входной вектор

    :return: ts.vec
             Нормированный вектор
    '''
    n = p.norm()
    if n != 0:
        p = p / n
    return p


@ti.func
def length(p):
    '''
    Функция вычисления длины вектора

    :param p: ts.vec
              Входной вектор

    :return: float
             Длина вектора
    '''
    return ti.sqrt(p.dot(p))


@ti.func
def voronoi(p, t):
    '''
    Функция рассчета диаграммы Вороного

    :param p: ts.vec2
              Входной вектор исходного пространства

    :param t: float
              Текущее время

    :return: ts.vec3
             Вектор рассчитанных дистанций и координат точек внутри многоугольников
    '''
    n = ts.floor(p)
    f = ts.fract(p)
    mg = ts.vec2(0.)
    mr = ts.vec2(0.)
    md = 8.0
    d = 0.
    for j in range(-1, 2):
        for i in range(-1, 2):
            g = ts.vec2(float(i), float(j))
            o = hash22(n + g)
            o = 0.5 + 0.5 * ts.sin(t * 2 * ts.pi * o)
            r = g + o - f
            d = r @ r
            if d < md:
                md = d
                mr = r
                mg = g

    md = 8.0
    for j in range(-2, 3):
        for i in range(-2, 3):
            g = mg + ts.vec2(float(i), float(j))
            o = hash22(n+g)
            o = 0.5 + 0.5 * ts.sin(t * 6.2831 * o)
            r = g + o - f
            if (mr - r) @ (mr - r) > 0.00001:
                md = ti.min(md, (0.5*(mr+r)) @ (normalize(r - mr)))

    return ts.vec3(md, mr.x, mr.y)


asp = 16/9
h = 600
w = int(asp * h)
res = w, h
resf = ts.vec2(float(w), float(h))
layers = 5

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

#%% Kernel function

@ti.kernel
def render(t: ti.f32, frame: ti.int32):
    '''
    Основная функция ti, формирует изображение

    :param t: float
              Текущее время

    :param frame: int
                  Граница

    :return: No return value
    '''
    #col0 = ts.vec3(255/255, 131/255, 137/255)

    for fragCoord in ti.grouped(pixels):
        uv = fragCoord / resf.x

        c = voronoi(8. * uv, 0.3 * t)

        col = c.x * (0.5 + 0.5 * ts.sin(64. * c.x)) * ts.vec3(1.0)
        col = ts.mix(ts.vec3(1.0, 0.6, 0.0), col, ts.smoothstep(c.x, 0.04, 0.07))
        dd = c.yz.norm()
        col = ts.mix(ts.vec3(1.0, 0.6, 0.1), col, ts.smoothstep(dd, 0.0, 0.12))
        col += ts.vec3(1.0, 0.6, 0.1) * (1.0 - ts.smoothstep(dd, 0.0, 0.04))

        pixels[fragCoord] = ts.clamp(col, 0., 1.)  # clamp(col ** (1 / 2.2), 0., 1.)


#%% GUI and main loop

gui = ti.GUI("Taichi example shader", res=res, fast_gui=True)
frame = 0
start = time.time()

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break

    t = time.time() - start
    render(t, frame)
    gui.set_image(pixels)
    gui.show()
    frame += 1

gui.close()
