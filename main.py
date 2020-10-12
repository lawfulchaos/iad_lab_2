import numpy as np
from PIL import Image
from numba import njit  # Компиляция функций для ускорения выполнения


DIMENSIONS = 5
REF_AREAS = ((0, 190, 100, 500), (250, 200, 400, 350), (20, 0, 120, 17))
CLASSES_TOTAL = len(REF_AREAS)

IMAGES = tuple(
    np.array(Image.open("bands/band{0}.bmp".format(i))) for i in range(DIMENSIONS)
              )  # numpy-массив значений пикселей для каждого изображения


@njit(cache=True)
def get_m_dist(class_num, point):
    """
    Дистанция по Махаланобису

    :param class_num: номер соответствующего класса
    :param point: Массив значений точки во всех измерениях
    :return: Число-расстояние до класса
    """
    euc_d = CLASS_CENTERS[class_num] - point

    return np.sqrt(np.dot(np.dot(np.transpose(euc_d), COV[class_num]), euc_d))


@njit(cache=True)
def get_dists(point):
    """
    Дистанция до каждого из классов

    :param point: Массив значений точки во всех измерениях
    :return: numpy-массив расстояний до каждого класса
    """
    return np.array([get_m_dist(c, point) for c in range(CLASSES_TOTAL)])


@njit(cache=True)
def get_pixel(x, y):
    """
    Значения точки по координатам в каждом из измерений

    :param x: x-координата на изображении
    :param y: x-координата на изображении
    :return: numpy-массив с значениями точки
    """
    return np.array([IMAGES[n][x][y] for n in range(DIMENSIONS)])


@njit(cache=True)
def get_pixel_area(class_num):
    """
    Значения точек в каждом из измерений для заданного класса

    :param class_num: номер класса
    :return: Список точек класса со значениями в каждом из измерений
    """
    return [
        get_pixel(x, y)
        for x in range(REF_AREAS[class_num][0], REF_AREAS[class_num][2])
        for y in range(REF_AREAS[class_num][1], REF_AREAS[class_num][3])
    ]


@njit(cache=True)
def get_colors(x, y):
    """
    Установление цвета для пикселя.
    Каждое из значений RGB обратно зависимо от расстояния до соответствующего класса

    :param x: x-координата на изображении
    :param y: x-координата на изображении
    :return: numpy-массив с значениями RGB для пикселя
    """
    dists = get_dists(get_pixel(x, y))
    if dists.all():  # Если расстояние до каждого класса > 0
        return 1 / dists * 255
    else:
        colors = np.zeros(3)
        colors[np.argmin(dists)] = 255
        return colors


@njit(cache=True)
def make_img_data(w, h):
    """
    Создание списка пикселей для раскрашенного изображения

    :param w: ширина
    :param h: высота
    :return: numpy-массив с RGB значениями для каждого пикселя
    """
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            data[x][y] = get_colors(x, y)

    return data


def make_img(w, h):
    """
    Сохранение изображения в файл

    :param w: ширина
    :param h: высота
    """
    data = make_img_data(w, h)
    img = Image.fromarray(data, "RGB")
    img.save("done.png")


@njit(cache=True)
def get_cov(c):
    """
    Высчитывание матрицы ковариации для класса

    :param c: Номер класса
    :return: Numpy-массив - матрица ковариации
    """
    class_size = [
        (REF_AREAS[c][2] - REF_AREAS[c][0]) * (REF_AREAS[c][3] - REF_AREAS[c][1])
        for c in range(CLASSES_TOTAL)
    ]

    return np.linalg.inv(1 / (class_size[c] - 1) * a_transposed[c] + np.eye(DIMENSIONS))


if __name__ == '__main__':
    # Ковариантность и центры
    CLASS_CENTERS = tuple(np.mean(get_pixel_area(c), axis=0) for c in range(CLASSES_TOTAL))

    a_mtrx = tuple(
        np.array([pixel - CLASS_CENTERS[c] for pixel in get_pixel_area(c)])
        for c in range(CLASSES_TOTAL)
    )

    a_transposed = tuple(
        np.matmul(a_mtrx[c].transpose(), a_mtrx[c]) for c in range(CLASSES_TOTAL)
    )

    COV = tuple(get_cov(c) for c in range(CLASSES_TOTAL))  # Матрицы ковариации

    make_img(500, 475)  # Запуск создания изображения
