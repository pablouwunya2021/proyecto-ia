# core/image_parser.py
from PIL import Image
import numpy as np

TILE = 1

def parse_image(path):
    img = Image.open(path).convert("RGB")
    img_array = np.array(img)

    rows = img_array.shape[0] // TILE
    cols = img_array.shape[1] // TILE

    grid = np.zeros((rows, cols))
    start = None
    goals = []

    for i in range(rows):
        for j in range(cols):
            block = img_array[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE]
            avg = block.mean(axis=(0,1))
            r,g,b = avg

            if r<50 and g<50 and b<50:      # pared
                grid[i,j] = 1

            elif r>200 and g<50 and b<50:   # inicio
                grid[i,j] = 2
                start = (i,j)

            elif g>200 and r<50 and b<50:   # meta
                grid[i,j] = 3
                goals.append((i,j))

            else:
                grid[i,j] = 0               # camino

    return img_array, grid, start, goals
