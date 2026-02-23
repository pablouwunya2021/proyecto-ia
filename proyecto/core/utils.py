# core/utils.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch

TILE = 1

def draw_path(img_array, path):
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_array)
    ax.axis("off")
    
    # Extraer coordenadas x, y del camino
    if len(path) > 0:
        y_coords = [p[0] * TILE + TILE/2 for p in path] 
        x_coords = [p[1] * TILE + TILE/2 for p in path] 
        
        ax.plot(x_coords, y_coords, 
                color='cyan', 
                linewidth=5, 
                linestyle='-', 
                alpha=0.8,
                zorder=10)
        
        
        # Inicio
        start_circle = Circle((x_coords[0], y_coords[0]), 
                             radius=10, 
                             color='yellow', 
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.9,
                             zorder=12)
        ax.add_patch(start_circle)
        
        # Fin
        end_circle = Circle((x_coords[-1], y_coords[-1]), 
                           radius=10, 
                           color='yellow', 
                           edgecolor='black',
                           linewidth=2,
                           alpha=0.9,
                           zorder=12)
        ax.add_patch(end_circle)
    
    # Título
    plt.title("Ruta encontrada por A* con costos de terreno", 
             fontsize=16, 
             fontweight='bold',
             pad=20)
    
    # Agregar leyenda
    legend_elements = [
        Patch(facecolor='red', label='Inicio'),
        Patch(facecolor='lime', label='Meta'),
        Patch(facecolor='blue', label='Agua (costo 10)'),
        Patch(facecolor='gray', label='Pavimento (costo 1)'),
        Patch(facecolor='cyan', edgecolor='black', linewidth=2, label='Ruta elegida'),
        Patch(facecolor='yellow', edgecolor='black', linewidth=2, label='Start/End')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()