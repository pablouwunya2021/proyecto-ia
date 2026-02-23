# core/maze_problem.py

class MazeProblem:

    def __init__(self, grid, start, goals):
        self.grid = grid
        self.start = start
        self.goals = goals
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_goal(self, node):
        return node in self.goals

    def neighbors(self, node):
        x,y = node
        moves = [(1,0),(-1,0),(0,1),(0,-1)]
        result=[]

        for dx,dy in moves:
            nx,ny=x+dx,y+dy
            if 0<=nx<self.rows and 0<=ny<self.cols:
                if self.grid[nx][ny]!=1:
                    result.append((nx,ny))
        return result

    def step_cost(self, current, neighbor, img_array, color_model):

        x, y = neighbor
        
        TILE = 1
        block = img_array[x*TILE:(x+1)*TILE, y*TILE:(y+1)*TILE]
        avg_color = block.mean(axis=(0, 1))  # [R, G, B]
        
        cost = color_model.predict_cost(avg_color)
        
        return cost