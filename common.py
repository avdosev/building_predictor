import numba
import numpy as np

# @numba.njit(numba.int32(numba.int32, numba.int32, numba.int32, numba.int32))
@numba.njit()
def update_points(point, i, water, building):
    if point == 1:
        water = min(water, i)
    if 3 < point <= 6:
        building = min(building, i)
    return water, building

@numba.njit()
def bfs(posY, posX, map, limit=100):
    water = limit+1
    building = limit+1

    maxY, maxX = map.shape
    for i in range(0, limit):
        # down
        y = posY + i
        if y < maxY:
            for x in range(max(posX-i, 0), min(posX+i+1, maxX)):
                point = map[y, x]
                water, building = update_points(point, i, water, building)
        # up
        y = posY - i
        if y >= 0:
            for x in range(max(posX-i, 0), min(posX+i+1, maxX)):
                point = map[y, x]
                water, building = update_points(point, i, water, building)
        # right
        x = posX + i
        if x < maxX:
            for y in range(max(posY-i, 0), min(posY+i+1, maxY)):
                point = map[y, x]
                water, building = update_points(point, i, water, building)
        # left
        x = posX - i
        if x >= 0:
            for y in range(max(posY-i, 0), min(posY+i+1, maxY)):
                point = map[y, x]
                water, building = update_points(point, i, water, building)

        if water != -1 and building != -1:
            break
        

    if water == limit + 1: water = -1
    if building == limit + 1: building = -1

    return np.array((water, building))

def train_pipe(val):
    # 0 = no data
    # 1 = water surface
    # 2 = land no built-up in any epoch
    # 3 = built-up
    i = np.where(val == 3, 2, val)
    i = np.where(i > 3, 3, i)
    return i

def test_pipe(data):
    i = np.where(data > 3, 3, data)
    res = np.zeros((i.shape[0], 4), dtype=bool)
    for j, item in enumerate(i):
        res[j, item] = True
    return res