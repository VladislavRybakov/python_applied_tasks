import taichi as ti
import taichi_glsl as ts
import time

ti.init(arch=ti.gpu)

#   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 5800 символов - 110 документации - 200 комментариев = 5490 символов !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   что дает различие в 690 символов !!!

# %% Resolution and pixel buffer

asp = 16 / 9
h = 600
w = int(asp * h)
res = w, h
resf = ts.vec2(float(w), float(h))
layers = 5

pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)


# %% Kernel function

@ti.func
def fract(x):
    '''
    Вычисляет дробную часть аргумента х

    :param x: float
              Число с плавающей точкой

    :return: float
             Дробная часть аргумета х
    '''
    return x - ti.floor(x)


@ti.func
def hash33(p):
    '''
    Хэш-функция, принимающая на вход трехмерный вектор, и возвращаюшая тоже трехмерный

    :param p: ts.vec3
              Входной вектор

    :return: ts.vec3
             Вектор, полученный в результате работы хэш-функции
    '''
    p = ts.fract(p * ts.vec3(0.1031, 0.11369, 0.13787))
    p += p.dot(ts.vec3(p.y, p.x, p.z) + 19.19)
    return -1.0 + 2.0 * ts.fract(ts.vec3(p.x + p.y, p.x + p.z, p.y + p.z) * ts.vec3(p.z, p.y, p.x))


@ti.func
def snoise3(p):
    '''
    Функция рассчета "случайного шума" с помощью хэш-функций

    :param p: ts.vec3
              Входной трехмерный вектор пространства-времени

    :return: float
             Значение шума
    '''
    K1 = 0.333333333
    K2 = 0.166666667

    i = ti.floor(p + (p.x + p.y + p.z) * K1)
    d0 = p - (i - (i.x + i.y + i.z) * K2)

    e = ts.step(ts.vec3(0.0), d0 - ts.vec3(d0.y, d0.z, d0.x))
    i1 = e * (1.0 - e.zxy)
    i2 = 1.0 - e.zxy * (1.0 - e)

    d1 = d0 - (i1 - K2)
    d2 = d0 - (i2 - K1)
    d3 = d0 - 0.5

    h = ti.max(0.6 - ts.vec4(d0.dot(d0), d1.dot(d1), d2.dot(d2), d3.dot(d3)), 0.0)
    n = h * h * h * h * ts.vec4(d0.dot(hash33(i)), d1.dot(hash33(i + i1)), d2.dot(hash33(i + i2)),
                                d3.dot(hash33(i + 1.0)))
    return ts.vec4(31.316) @ n


@ti.func
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
    return ti.max(ti.min(x, high), low)


@ti.func
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


@ti.func
def extractAlpha(colorIn):
    '''
    Рассчитывает плавный переход между двигающейся картинкой и фоном

    :param colorIn: ts.vec3
                    Вектор цвета RGB

    :return: ts.vec4
             Вектор пересчитанного цвета и его яркость
    '''
    colorOut = ts.vec4(0.0)
    maxValue = ti.min(ti.max(ti.max(colorIn.r, colorIn.g), colorIn.b), 1.0)
    if maxValue > 1e-5:
        colorOut.rgb = colorIn.rgb * (1.0 / maxValue)
        colorOut.a = maxValue
    else:
        colorOut = ts.vec4(0.)
    return colorOut


@ti.func
def light1(intensity, attenuation, dist):
    '''
    Рассчитывает длину распространения свечения двигающегося света

    :param intensity: float
                      Интенсивность света
    :param attenuation: float
                        Сила затухания света
    :param dist: float
                 Расстояние от окружности

    :return: float
             Сила свечения (как далеко оно распространяется)
    '''
    return intensity / (1.0 + dist * attenuation)


@ti.func
def light2(intensity, attenuation, dist):
    '''
    Рассчитывает силу свечения всей окружности

    :param intensity: float
                      Интенсивность света
    :param attenuation: float
                        Сила затухания света
    :param dist: float
                 Расстояние от окружности

    :return: float
             Яркость свечения
    '''
    return intensity / (1.0 + dist * dist * attenuation)


@ti.func
def length(p):
    '''
    Рассчитывает длину вектора

    :param p: ts.vec
              Вектор

    :return: float
             Длина вектора
    '''
    return ti.sqrt(p.dot(p))


@ti.func
def draw(vUv, time):
    '''
    Основная функция, рассчитывающая всю графику

    :param vUv: ts.vec2
                Входной вектор исходного пространства
    :param time: float
                 Текущее время

    :return: ts.vec4
             Четырехмерный вектор, содержащий в себе цвет пикселя в RGB и его яркость
    '''
    uv = vUv
    ang = ts.atan(uv.y, uv.x)
    len = length(uv)
    color1 = ts.vec3(0.611765, 0.262745, 0.996078)
    color2 = ts.vec3(0.298039, 0.760784, 0.913725)
    color3 = ts.vec3(0.062745, 0.078431, 0.600000)
    innerRadius = 0.6
    noiseScale = 0.65

    # ring
    n0 = snoise3(ts.vec3(uv.x * noiseScale, uv.y * noiseScale, time * 0.5)) * 0.5 + 0.5
    r0 = ts.mix(ts.mix(innerRadius, 1.0, 0.4), ts.mix(innerRadius, 1.0, 0.6), n0)
    d0 = ts.distance(uv, r0 / len * uv)
    v0 = light1(1.0, 10.0, d0)
    v0 *= smoothstep(r0 * 1.05, r0, len)
    cl = ti.cos(ang + time * 2.0) * 0.5 + 0.5

    # high light
    a = time * (-1.0)
    pos = ts.vec2(ti.cos(a), ti.sin(a)) * r0
    d = ts.distance(uv, pos)
    v1 = light2(1.5, 5.0, d)
    v1 *= light1(1.0, 50.0, d0)

    # back decay
    v2 = smoothstep(1.0, ts.mix(innerRadius, 1.0, n0 * 0.5), len)

    # hole
    v3 = smoothstep(innerRadius, ts.mix(innerRadius, 1.0, 0.5), len)

    # color
    c = ts.mix(color1, color2, cl)
    col = ts.mix(color1, color2, cl)
    col = ts.mix(color3, col, v0)
    col = (col + v1) * v2 * v3

    col.rgb = ts.clamp(col.rgb, 0.0, 1.)

    fragColor = extractAlpha(col)
    return fragColor


# нововведение!
@ti.func
def normalize(p):
    '''
    (документация добавляет 110 символов)
    Нормирует вектор

    :param p: ts.vec
              Вектор

    :return: ts.vec
             Отнормированный вектор
    '''
    n = p.norm()
    if n != 0:
        p = p / n
    return p


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
    BG_COLOR = ts.vec3(ti.sin(t) * 0.5 + 0.5) * 0.0 + ts.vec3(0.0)
    col0 = ts.vec3(255 / 255, 131 / 255, 137 / 255)

    for fragCoord in ti.grouped(pixels):
        uv = (fragCoord * 2.0 - resf) / resf[1]

        ######## нововведения ########
        noise1 = snoise3(ts.vec3(uv.x, uv.y, t))  # шум для искажения пространства и создания эффекта "волнения" пространства
        noise2 = snoise3(ts.vec3(fragCoord.x, fragCoord.y, 0.0))  # шум для создания эффекта шума - пиксельного размытия изображения
        uv = (2.0 * fragCoord - resf + ts.vec2(ts.sin(t / 5), ts.cos(t / 5))) / resf[1] + ts.vec2(noise1 * 0.1, noise1 * 0.1) + ts.vec2(noise2 * 0.04, noise2 * 0.04)

        ## далее идет добавление еще одной окружности и эффекта внутри нее
        offset = ts.vec2(ts.cos(t / 2.0) * uv.x, ts.sin(t / 2.0) * uv.y)

        light_color = ts.vec3(ts.sin(t / 2), ts.cos(t / 3), ts.sin(t / 5))  # цвет изменяется со временем
        light = 0.05 / ts.distance(normalize(uv), uv)

        if length(uv) < 1.0:
            light *= 0.1 / ts.distance(normalize(uv - offset), uv - offset)

        col_plus = light * light_color
        ##############################

        col = draw(uv, t)  # основное кольцо

        pixels[fragCoord] = ts.mix(col_plus, col.rgb, col.a)


# %% GUI and main loop


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
