# core/utils.py
import matplotlib.pyplot as plt

TILE = 1

def draw_path(img_array, path):
    for x,y in path:
        img_array[x*TILE:(x+1)*TILE, y*TILE:(y+1)*TILE] = [0,0,255]

    plt.imshow(img_array)
    plt.title("Ruta encontrada")
    plt.axis("off")
    plt.show()
