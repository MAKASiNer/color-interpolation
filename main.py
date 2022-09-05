import cv2
import numpy as np


# размеры изображения
SIZE = 200, 200

# вершины многоугольника
VERTEXES = np.array([
    [50, 50],
    [50, 100],
    [100, 150],
    [150, 150],
    [150, 100],
    [100, 50],
], dtype=np.int32)

# цвета многоугольника
COLORS = np.array([
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 50],
    [0, 255, 255],
    [0, 50, 255],
    [255, 0, 255]
], dtype=np.uint8)


def points_inside_polygon(img, vertexes):
    p = {}
    mask = img.copy()
    cv2.fillConvexPoly(mask, VERTEXES, (1, 1, 1))
    for y, arr in enumerate(mask):
        p[y] = {}
        for x, clr in enumerate(arr):
            if clr.all():
                p[y][x] = [0, 0, 0]
        if not p[y]:
            p.pop(y)
    return p


def distance_between_point_and_line(xy, p1, p2):
    ab = np.linalg.norm(xy - p1)
    bc = np.linalg.norm(p1 - p2)
    ca = np.linalg.norm(p2 - xy)
    p = (ab + bc + ca) / 2
    area = np.sqrt(p * (p - ab) * (p - bc) * (p - ca))
    return 2 * area / bc


def distance_to_opposite_segment(xy, i, vertexes):
    """
    xy       - точка
    i        - индекс вершины, для которой считаем противоположенный сегмент
    vertexes - вершины
    """
    n = len(vertexes)
    if n % 2:
        # для нечетного кол-ва вершин грань есть ребро полигона
        p1 = vertexes[(i + n // 2) % n]
        p2 = vertexes[(i + n // 2 + 1) % n]
        l = distance_between_point_and_line(vertexes[i], p1, p2)
    else:
        # для четного кол-ва вершин грань есть перепендикуляр к линии,
        # образованной i-той точкой и противолежащей ей
        p1 = vertexes[(i + n // 2) % n]
        p2 = p1 + (lambda x: [-x[1], x[0]])(vertexes[i] - p1)
        l = np.linalg.norm(p1 - p2)

    return distance_between_point_and_line(xy, p1, p2) / l


img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)


POINTS = points_inside_polygon(img, VERTEXES)

progress = 0 
for y in POINTS:
    for x, clr in POINTS[y].items():
        R = [distance_to_opposite_segment((x, y), i, VERTEXES) for i, _ in enumerate(VERTEXES)]
        for i, r in enumerate(R):
            clr = np.array(COLORS[i] * r / np.sum(R), dtype=np.uint8)
            img[y][x] += clr

    print(f"{100 * (progress := progress + 1) / len(POINTS) :5.2f} %", end='\r')


for p, clr in zip(VERTEXES, COLORS):
    cv2.circle(img, p, 3, [int(x) for x in clr], -1)
    cv2.circle(img, p, 3, (0, 0, 0), 1)

cv2.imwrite("res.png", img)
