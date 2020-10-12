import numpy as np
from PIL import Image
from numba import njit

DIMENSIONS = 5
REF_AREAS = ((0, 190, 100, 500), (250, 200, 400, 350), (20, 0, 120, 17))
CLASSES_TOTAL = len(REF_AREAS)

IMAGES = tuple(
    np.array(Image.open("bands/band{0}.bmp".format(i))) for i in range(DIMENSIONS)
)


@njit(cache=True)
def get_m_dist(class_num, point):
    euc_d = CLASS_CENTERS[class_num] - point

    return np.sqrt(np.dot(np.dot(np.transpose(euc_d), COV[class_num]), euc_d))


@njit(cache=True)
def get_dists(point):
    return np.array([get_m_dist(c, point) for c in range(CLASSES_TOTAL)])


@njit(cache=True)
def get_pixel(x, y):
    return np.array([IMAGES[n][x][y] for n in range(DIMENSIONS)])


@njit(cache=True)
def get_pixel_area(class_num):
    return [
        get_pixel(x, y)
        for x in range(REF_AREAS[class_num][0], REF_AREAS[class_num][2])
        for y in range(REF_AREAS[class_num][1], REF_AREAS[class_num][3])
    ]


@njit(cache=True)
def get_colors(x, y):
    dists = get_dists(get_pixel(x, y))
    if dists.all():
        return 1 / dists * 255
    else:
        colors = np.zeros(3)
        colors[np.argmin(dists)] = 255
        return colors


@njit(cache=True)
def make_img_data(w, h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            data[x][y] = get_colors(x, y)

    return data


def make_img(w, h):
    data = make_img_data(w, h)
    img = Image.fromarray(data, "RGB")
    img.save("done.png")


@njit(cache=True)
def get_cov(c):
    class_size = [
        (REF_AREAS[c][2] - REF_AREAS[c][0]) * (REF_AREAS[c][3] - REF_AREAS[c][1])
        for c in range(CLASSES_TOTAL)
    ]

    return np.linalg.inv(1 / (class_size[c] - 1) * a_transposed[c] + np.eye(DIMENSIONS))


# Ковариантность и центры
CLASS_CENTERS = tuple(np.mean(get_pixel_area(c), axis=0) for c in range(CLASSES_TOTAL))

a_mtrx = tuple(
    np.array([pixel - CLASS_CENTERS[c] for pixel in get_pixel_area(c)])
    for c in range(CLASSES_TOTAL)
)

a_transposed = tuple(
    np.matmul(a_mtrx[c].transpose(), a_mtrx[c]) for c in range(CLASSES_TOTAL)
)

COV = tuple(get_cov(c) for c in range(CLASSES_TOTAL))

make_img(500, 475)
