# February 2022
# Implement several pathfinding algorithms to be visualized

from __future__ import annotations

from random import choices

from datastructures.priority_queue import PriorityQueue

class Grid:
    def __init__(self, data: list[list[int]] | None = None) -> None:
        if data is not None:
            self.data = data
        else:
            self.generate()

    def generate(self, n: int = 10) -> None:
        self.data = []
        for _ in range(n):
            self.data.append(choices([0, 1], cum_weights=[90, 100], k=n))

    def adjacent(self, coord: tuple[int]) -> set[tuple[int]]:
        n = len(self.data)
        transforms = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        tests = [tuple(map(int.__add__, coord, transform)) for transform in transforms]

        adjacent = set()
        for point in tests:
            i, j = point
            if i < 0 or i == n or j < 0 or j == len(self.data[i]) or self.data[i][j]:
                continue
            adjacent.add(point)
        return adjacent

    def highlight_path(self, curr: tuple[int], prev_table: dict[tuple[int]: tuple[int] | None]) -> Grid:
        data = [x[:] for x in self.data] # performs a shallow copy

        while curr:
            i, j = curr
            data[i][j] = 2
            curr = curr in prev_table and prev_table[curr]

        if sum(row.count(2) for row in data) == 1:
            raise ValueError("No available path to end.")

        return Grid(data)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, index: slice | int) -> list[list[int]] | list[int]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return str(self.data).replace("],", "]\n").replace(",", "").replace("0", " ").replace("1", "\N{FULL BLOCK}").replace("2", "-")

def dijkstra(grid: Grid, start: tuple[int], end: tuple[int]) -> Grid:
    """Performs Dijkstra's search algorithm to find the shortest path between start and end.
       Returns the result as a grid that highlights the shortest path"""
    visited = set()
    unvisited = set((i, j) for i in range(len(grid)) for j in range(len(grid[i])) if not grid[i][j])

    if start not in unvisited or end not in unvisited:
        raise ValueError("Invalid starting or ending point.")

    dist_table = dict.fromkeys(unvisited, float("inf"))
    dist_table[start] = 0
    prev_table = dict.fromkeys(unvisited, None)

    curr = start
    while unvisited:
        adjacent = grid.adjacent(curr) - visited
        for point in adjacent:
            dist = dist_table[curr] + 1
            if dist < dist_table[point]:
                dist_table[point] = dist
                prev_table[point] = curr
        unvisited.remove(curr)
        visited.add(curr)
        curr = min(unvisited, key=dist_table.get, default=None)

    return grid.highlight_path(end, prev_table)

def manhattan_distance(pointA: tuple[int], pointB: tuple[int]) -> int:
    """Returns the Manhattan distance between points A and B, assuming a weight of 1"""
    return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])

def astar(grid: Grid, start: tuple[int], end: tuple[int]) -> Grid:
    """Performs A* search algorithm to find the shortest path between start and end.
       Uses Manhattan distance as the heuristic function.
       Returns the result as a grid that highlights the shortest path"""
    if grid[start[0]][start[1]] or grid[end[0]][end[1]]:
        raise ValueError("Invalid starting or ending point.")

    queue = PriorityQueue()
    queue.put(start, 0)
    dist_table = {start: 0}
    prev_table = {start: None}

    while not queue.empty():
        curr = queue.get()
        if curr is end:
            break

        for point in grid.adjacent(curr):
            dist = dist_table[curr] + 1
            if point not in dist_table or dist < dist_table[point]:
                dist_table[point] = dist
                prev_table[point] = curr
                queue.put(point, -1*(dist + manhattan_distance(point, end)))

    return grid.highlight_path(end, prev_table)

def main() -> None:
    """Code execution starts here"""
    _ = [
        [0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0,1,0],
        [0,1,0,0,0,0,1,0,1,0],
        [0,1,0,1,1,0,1,0,1,0],
        [0,1,0,1,1,1,1,0,1,0],
        [0,1,0,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0]]
    g = Grid()
    s = (0, 0)
    e = (7, 7)
    print(dijkstra(g, s, e))
    print(astar(g, s, e))

if __name__=="__main__":
    main()
