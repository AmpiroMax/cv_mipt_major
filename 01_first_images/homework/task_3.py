import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    h, w, _ = image.shape

    # для поворота я использую аффинкук реализованную ниже
    # определяем куда перейдут точки после аффинки
    pts1 = np.float32([[w/2, 0], [w, h / 2], [w/2, h/2]])
    angle = angle * np.pi / 180
    pts2 = np.float32([[w/2 - h/2 * np.sin(angle), h/2 - h/2 * np.cos(angle)],
                      [w/2 + w/2 * np.cos(angle), h/2 - w/2 * np.sin(angle)], [w/2, h/2]])

    # возвращаем результат аффинного преобразоваания
    return apply_warpAffine(image, pts1, pts2)


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    h, w, _ = image.shape
    picture = image.copy()
    M = cv2.getAffineTransform(points1, points2)

    x1 = 0
    y1 = 0

    x2 = w
    y2 = h

    # координаты 4 начальных точек
    dot1 = [x1, y1, 1]
    dot2 = [x1, y2, 1]
    dot3 = [x2, y1, 1]
    dot4 = [x2, y2, 1]

    # координаты точек после преобразования
    dot11 = M @ dot1
    dot21 = M @ dot2
    dot31 = M @ dot3
    dot41 = M @ dot4

    # размеры окна в котором будем отрисовывать картинку
    w1 = max(abs(dot41[0] - dot11[0]), abs(dot31[0] - dot21[0]))
    h1 = max(abs(dot41[1] - dot11[1]), abs(dot31[1] - dot21[1]))

    # сдвиг в начало координат
    dx = -min(dot11[0], dot21[0], dot31[0], dot41[0])
    dy = -min(dot11[1], dot21[1], dot31[1], dot41[1])
    Move = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    # я использовал матричные преобразования, поэтому добавляю
    # ещё одну строчку, чтобы можно было перемножить матрицы
    M1 = list(M)
    M1 += [[0, 0, 1]]

    # получаем итоговую матрицу преобразования
    Matr = Move @ M1
    Matr = Matr[:-1]

    picture = cv2.warpAffine(picture, Matr, (int(w1), int(h1)))
    return picture
