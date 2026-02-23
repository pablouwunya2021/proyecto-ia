from core.image_parser import parse_image
from core.maze_problem import MazeProblem
from core.search_algorithms import bfs,dfs,astar
from core.utils import draw_path
from core.color_model import ColorMLP

# ===== TASK 2.2: Integración de Red Neuronal con A* =====

# PASO 1: Entrenar el modelo de colores
COLOR_DATASET = "final_data_colors.csv"
print("=" * 60)
print("TASK 2.1 - Entrenando modelo de colores RGB")
print("=" * 60)
color_model = ColorMLP()
color_model.train(COLOR_DATASET, epochs=200, lr=0.05, batch_size=32)

# PASO 2: Cargar el mapa de prueba
IMAGE_PATH = "maps/mapa_colores.png"
print("\n" + "=" * 60)
print("TASK 2.2 - Cargando mapa con colores")
print("=" * 60)
img_array, grid, start, goals = parse_image(IMAGE_PATH)

print(f"Inicio: {start}")
print(f"Metas: {goals}")
print(f"Dimensiones del grid: {grid.shape}")

# PASO 3: Crear el problema
problem = MazeProblem(grid, start, goals)

# PASO 4: Ejecutar A* con el modelo de colores
print("\n" + "=" * 60)
print("TASK 2.2 - Ejecutando A* con costos de terreno")
print("=" * 60)
print("El robot está evaluando los costos de cada terreno...")
print("  🔵 AZUL (Agua) = Costo 10")
print("  ⚪ GRIS (Pavimento) = Costo 1")
print()

path = astar(problem, img_array, color_model)

# PASO 5: Mostrar resultados
if path:
    print(f"Camino encontrado!")
    print(f"   Longitud del camino: {len(path)} nodos")
    print(f"\n Resultado esperado:")
    print(f"   El robot debería elegir el camino GRIS (largo pero barato)")
    print(f"   en lugar del camino AZUL (corto pero costoso)")
    print(f"\n Mostrando visualización...")
    draw_path(img_array, path)
else:
    print("No se encontró solución")

print("\n" + "=" * 60)
print("TASK 2.2 COMPLETADO")
print("=" * 60)