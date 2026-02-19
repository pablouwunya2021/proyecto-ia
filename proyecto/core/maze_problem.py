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
