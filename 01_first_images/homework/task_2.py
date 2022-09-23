import cv2
import numpy as np


def find_road_number(image):
    image_hsv = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)

    # подбираем маски

    # маска препятствий
    hsv_low = (1, 250, 250)
    hsv_high = (255, 255, 255)
    obstacle = cv2.inRange(image_hsv, hsv_low, hsv_high)

    # маска на машину
    hsv_low = (109, 0, 0)
    hsv_high = (112, 255, 255)
    car = cv2.inRange(image_hsv, hsv_low, hsv_high)

    # маска на поолосы
    hsv_low = (25, 0, 0)
    hsv_high = (32, 255, 255)
    lines = cv2.inRange(image_hsv, hsv_low, hsv_high)

    # дополнительная маска на костомизацию
    hsv_low = (1, 250, 250)
    hsv_high = (255, 255, 255)
    additional_mask = cv2.inRange(image_hsv, hsv_low, hsv_high)

    # несмотря на все маски получить ровную и аккуратную картинку не удалось
    # masked_image
    mimage = additional_mask + lines + car + obstacle

    h, w = mimage.shape
    line_size = 0
    road_size = 0
    for i in range(w):
        if(mimage[h//2, i] != 0 and road_size != 0):
            break
        if(mimage[h//2, i] != 0):
            line_size += 1
        if(mimage[h//2, i] == 0 and line_size != 0):
            road_size += 1

    road_count = round((h - line_size) / (line_size + road_size))
    roads = np.zeros(road_count)

    # определяем есть на дороге препятствие или нет
    delta = int(((w - line_size) / road_count) // 2)
    for road_num in range(road_count):
        for dh in range(h // 2):
            # у меня почему-то после маски картинка не бинарной получилась
            if(mimage[dh, delta + 2 * delta * road_num] == 255 or mimage[dh, delta + 2 * delta * road_num] == 254):
                roads[road_num] = -1
                break

    # определяем есть на дороге машина или нет
    cars = np.zeros(road_count)
    for road_num in range(road_count):
        for dh in range(h // 2, h, 1):
            if(mimage[dh, delta + 2 * delta * road_num] == 255 or mimage[dh, delta + 2 * delta * road_num] == 254):
                cars[road_num] = -1
                break

    curr_num = 0
    num = 0
    for i in range(road_count):
        if(roads[i] == 0):
            num = i
        if(cars[i] == -1):
            curr_num = i

    if(curr_num == num):
        print("Перестраиваться не нужно")

    print(roads)
    print(cars)
    # plot_one_image(mimage)

    return num+1
