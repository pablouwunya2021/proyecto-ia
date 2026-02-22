from core.image_parser import parse_image
from core.maze_problem import MazeProblem
from core.search_algorithms import bfs,dfs,astar
from core.utils import draw_path
from core.color_model import ColorMLP

IMAGE_PATH="maps/mapa1.png"

img_array,grid,start,goals=parse_image(IMAGE_PATH)

print("Start:",start)
print("Goals:",goals)

problem=MazeProblem(grid,start,goals)

# elegir algoritmo
path=astar(problem)
# path=bfs(problem)
# path=dfs(problem)

if path:
    print("Path length:",len(path))
    draw_path(img_array,path)
else:
    print("No solution found")

# Task 2

COLOR_DATASET = "final_data_colors.csv"

print("Training color model...")
color_model = ColorMLP()
color_model.train(COLOR_DATASET)
