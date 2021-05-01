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
def bfs(posY, posX, map, limit):
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

    return [water, building]

@numba.njit()
def count(posY, posX, map, limit):
    maxY, maxX = map.shape
    count_build = 0
    count_land = 0
    for y in range(max(posY-limit, 0), min(posY+limit, maxY)):
        for x in range(max(posX-limit, 0), min(posX+limit, maxX)):
            if map[y, x] > 3:
                count_build += 1
            if map[y, x] == 2 or map[y, x] == 3:
                count_land += 1
    return [count_build, count_land]

@numba.njit()
def find_info(posY, posX, map, limit=100, radius_build_limit=14):
    i1 = bfs(posY, posX, map, limit)
    i2 = count(posY, posX, map, radius_build_limit)
    on_water = map[posY, posX] == 1
    on_land = map[posY, posX] == 2 or map[posY, posX] == 3
    on_build = map[posY, posX] > 3
    return np.array(i1 + i2 + [on_water, on_land, on_build])

def train_pipe(val):
    # 0 = no data
    # 1 = water surface
    # 2 = land no built-up in any epoch
    # 3 = built-up
    i = np.where(val == 3, 2, val)
    i = np.where(i > 3, 3, i)
    return i

@numba.njit()
def test_pipe(data):
    i = np.where(data > 3, 3, data)
    res = np.zeros((i.shape[0], 4))
    for j, item in enumerate(i):
        res[j, item] = 1
    return res