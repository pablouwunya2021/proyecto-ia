# core/search_algorithms.py
from collections import deque
import heapq

def reconstruct_path(came_from, goal):
    path=[goal]
    while goal in came_from:
        goal=came_from[goal]
        path.append(goal)
    return path[::-1]


# -------- BFS --------
def bfs(problem):
    frontier=deque([problem.start])
    visited=set([problem.start])
    came_from={}

    while frontier:
        node=frontier.popleft()

        if problem.is_goal(node):
            return reconstruct_path(came_from,node)

        for n in problem.neighbors(node):
            if n not in visited:
                visited.add(n)
                came_from[n]=node
                frontier.append(n)
    return None


# -------- DFS --------
def dfs(problem):
    stack=[problem.start]
    visited=set()
    came_from={}

    while stack:
        node=stack.pop()

        if problem.is_goal(node):
            return reconstruct_path(came_from,node)

        if node in visited:
            continue

        visited.add(node)

        for n in problem.neighbors(node):
            if n not in visited:
                came_from[n]=node
                stack.append(n)
    return None


# -------- A* --------
def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(problem):
    goal=problem.goals[0]

    pq=[]
    heapq.heappush(pq,(0,problem.start))

    came_from={}
    cost={problem.start:0}

    while pq:
        _,node=heapq.heappop(pq)

        if problem.is_goal(node):
            return reconstruct_path(came_from,node)

        for n in problem.neighbors(node):
            new_cost=cost[node]+1

            if n not in cost or new_cost<cost[n]:
                cost[n]=new_cost
                priority=new_cost+heuristic(n,goal)
                heapq.heappush(pq,(priority,n))
                came_from[n]=node
    return None
