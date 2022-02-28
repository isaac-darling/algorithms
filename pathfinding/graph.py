# January 2022
# Implement several pathfinding algorithms to be visualized

from __future__ import annotations

from math import sqrt
from random import randint, sample

from datastructures.priority_queue import PriorityQueue

class Node:
    """Represents a node inside of a graph"""
    _coords = set()

    def __init__(self) -> None:
        x, y = randint(0, 500), randint(0, 500)
        while (x, y) in Node._coords:
            x, y = randint(0, 500), randint(0, 500)

        self.x = x
        self.y = y
        Node._coords.add((x, y))

    def distance_to(self, node: Node) -> float:
        return sqrt((self.x - node.x)**2 + (self.y - node.y)**2)

    def __del__(self) -> None:
        Node._coords.remove((self.x, self.y))

class Graph:
    """Represents a system of nodes and edges"""
    def __init__(self, data: tuple[list[Node], list[set[Node]]] | None = None) -> None:
        if data is not None:
            self.nodes, self.edges = data
        else:
            self.nodes = []
            self.edges = []

    @staticmethod
    def _assemble(curr: Node, prev_table: dict[Node: Node | None]) -> Graph:
        """Assembles a Graph object from a starting node and the table of parent nodes"""
        nodes = []
        while curr:
            nodes.append(curr)
            curr = curr in prev_table and prev_table[curr]
        nodes = nodes[::-1]

        num_edges = len(nodes) - 1
        if num_edges <= 0:
            assert num_edges > 0, "No available path to end."

        edges = []
        for i in range(num_edges):
            edges.append({nodes[i], nodes[i+1]})

        data = nodes, edges
        return Graph(data)

    def add_node(self) -> None:
        self.nodes.append(Node())

    def add_edge(self, nodeA: Node, nodeB: Node) -> None:
        for edge in self.edges:
            if nodeA in edge and nodeB in edge:
                return
        self.edges.append({nodeA, nodeB})

    def populate(self, n: int = 10) -> None:
        for _ in range(n):
            self.add_node()
        for _ in range(n):
            node1, node2 = sample(self.nodes, k=2)
            self.add_edge(node1, node2)

    def neighbors(self, node: Node) -> set[Node]:
        neighbors = set()
        for edge in self.edges:
            if node in edge:
                for item in edge:
                    if item is not node:
                        neighbors.add(item)
                        break
        return neighbors

    def __str__(self) -> str:
        return f"{[(node.x, node.y) for node in self.nodes]}\n{len(self.edges)}"

def dijkstra(graph: Graph, start: Node, end: Node) -> Graph:
    """Performs Dijkstra's search algorithm to find the shortest path between start and end.
       Returns the result as a Graph that contains only the shortest path"""
    visited = set()
    unvisited = set(graph.nodes)
    dist_table = dict.fromkeys(unvisited, float("inf"))
    dist_table[start] = 0
    prev_table = dict.fromkeys(unvisited, None)

    curr = start
    while unvisited:
        adjacent = graph.neighbors(curr) - visited
        for node in adjacent:
            dist = curr.distance_to(node) + dist_table[curr]
            if dist < dist_table[node]:
                dist_table[node] = dist
                prev_table[node] = curr
        unvisited.remove(curr)
        visited.add(curr)
        curr = min(unvisited, key=dist_table.get, default=None)

    return Graph._assemble(end, prev_table)

def astar(graph: Graph, start: Node, end: Node) -> Graph:
    """Performs A* search algorithm to find the shortest path between start and end.
       Uses Euclidean distance as the heuristic function.
       Returns the result as a Graph that contains only the shortest path"""
    queue = PriorityQueue()
    queue.put(start, 0)
    dist_table = {start: 0}
    prev_table = {start: None}

    while not queue.empty():
        curr = queue.get()
        if curr is end:
            break

        for node in graph.neighbors(curr):
            dist = dist_table[curr] + curr.distance_to(node)
            if node not in dist_table or dist < dist_table[node]:
                dist_table[node] = dist
                prev_table[node] = curr
                queue.put(node, -1*(dist + node.distance_to(end)))

    return Graph._assemble(end, prev_table)

def main() -> None:
    """Code execution starts here"""

if __name__=="__main__":
    main()
