import cv2
import numpy as np


def get_the_point(cur_point, d, array):
    """
    Возвращает следующую точку в алгоритме востановления пути

    Parametrs
    ---------
    cur_point: tuple
      Точка из которой мы востанавливаем сейчас путь

    array: np.array
      Массив получившийся из волнового алгоритма.
      По этому массиву мы востанавливаем путь

    Returns
    -------
    tuple
      Возвращает следующую точку в пути
    """
    i = cur_point[0]
    j = cur_point[1]

    if(i - 1 >= 0):
        if(array[i - 1, j] == d - 1):
            return (i - 1, j)

    if(i + 1 < array.shape[0]):
        if(array[i + 1, j] == d - 1):
            return (i + 1, j)

    if(j - 1 >= 0):
        if(array[i, j - 1] == d - 1):
            return (i, j - 1)

    if(j + 1 < array.shape[1]):
        if(array[i, j + 1] == d - 1):
            return (i, j + 1)

    # ошибка в доступе к элементам массива
    print("TROUBLE")
    print(cur_point)
    print(array[i, j - 1], d)
    print(array[i - 1, j], d)
    return(cur_point)


def marker(cur_point: tuple, aim_point: tuple, d: int, array):
    """
    Реализация волнового алгоритма поиска пути
    от cur_point точки до aim_point номер волны записывается в массив array
    По нему восстанавливается в дальнейшем путь

    Parametrs
    ---------
    cur_point: tuple
      начальная точка алгоритма

    aim_point: tuple
      целевая точка алгоритма

    d: int
      начальный номер волны

    array: np.array 
      массив в который записывается результат 

    Returns
    -------
    int
      Число итераций проверки гипотезы
    """
    stk = []
    buf_stk = []

    stk += [cur_point]

    x = array.shape[0]
    y = array.shape[1]

    while(len(stk) != 0):
        while(len(stk) != 0):
            cur_point = stk[-1]
            stk.pop()

            if(cur_point[0] == aim_point[0] and cur_point[1] == aim_point[1]):
                return d - 1

            i = cur_point[0]
            j = cur_point[1]

            if(i - 1 >= 0):
                if(array[i - 1, j] == 0):
                    array[i - 1, j] = d
                    buf_stk += [(i - 1, j)]

            if(i + 1 < x):
                if(array[i + 1, j] == 0):
                    array[i + 1, j] = d
                    buf_stk += [(i + 1, j)]

            if(j - 1 >= 0):
                if(array[i, j - 1] == 0):
                    array[i, j - 1] = d
                    buf_stk += [(i, j - 1)]

            if(j + 1 < y):
                if(array[i, j + 1] == 0):
                    array[i, j + 1] = d
                    buf_stk += [(i, j + 1)]
        d += 1
        stk = buf_stk.copy()
        buf_stk = []


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    # ---------------------------

    # Определяем размеры еденичного квадрата лабиринта
    # проходим по верхней полоске и измеряем длину прохода,
    # которую принимаем за размер квадрата
    size_box = 0
    for i in range(image.shape[0]):
        if(image[0, i, 0] != 0):
            size_box += 1
        if(image[0, i, 0] == 0 and size_box != 0):
            break

    # Определяем толщину линии
    size_line = 0
    for i in range(image.shape[1]):
        if(image[i, size_box//2, 0] == 0):
            size_line += 1
        if(image[i, size_box//2, 0] != 0 and size_line != 0):
            break

    # Инициализируем массив
    # пробегаем по краям и серединам, чтоб определить
    # горизонтальные и вертикальные стенки
    n = 1 + 2 * (image.shape[0] - size_line) // (size_line + size_box)
    m = 1 + 2 * (image.shape[1] - size_line) // (size_line + size_box)

    maze_array = np.zeros((n, m), int)
    delta = (size_box + size_line) // 2

    for i in range(n):
        for j in range(m):
            if image[(i) * delta + 0, (j) * delta + 0, 0] == 0:
                maze_array[i, j] = 1
            else:
                maze_array[i, j] = 0

    # ищем вход и выход лабиринта
    start_point = ()
    finish_point = ()

    for j in range(m):
        if(maze_array[0, j] == 0):
            start_point = (0, j)

    for j in range(m):
        if(maze_array[n - 1, j] == 0):
            finish_point = (n - 1, j)

    # запускаем волновой алгоритм
    d = 2
    maze_array[start_point[0], start_point[1]] = d
    d = marker(start_point, finish_point, d + 1, maze_array)

    path = []
    curr_point = finish_point

    while(d != 2):
        curr_point = get_the_point(curr_point, d, maze_array)
        path += [(curr_point[0] * delta, curr_point[1] * delta)]
        d -= 1

    # дополняем путь точками для того, чтоб
    # путь рисовался прямыми линиями
    path = np.array(path)
    new_path = []

    for i in range(path.shape[0] - 1):
        dot1 = path[i]
        dot2 = path[i + 1]

        if(dot1[0] == dot2[0]):
            begin = min(dot1[1], dot2[1])
            end = max(dot1[1], dot2[1])
            for j in range(end, begin, -1):
                new_path += [[dot1[0], j]]

        if(dot1[1] == dot2[1]):
            begin = min(dot1[0], dot2[0])
            end = max(dot1[0], dot2[0])
            for j in range(end, begin, -1):
                new_path += [[j, dot1[1]]]

    new_path = np.array(new_path).T
    return new_path
