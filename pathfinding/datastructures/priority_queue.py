from typing import Any

class PriorityQueue:
    """Priority queue data structure"""
    def __init__(self, data: list[tuple[Any, float]] | None = None) -> None:
        if data is not None:
            self._q = sorted(data, key=lambda x: x[1])
        else:
            self._q = []

    def put(self, item: Any, priority: float) -> None:
        """Adds an item to the correct place in the queue"""
        n = len(self._q)

        i = 0
        while i < n and priority > self._q[i][1]:
            i += 1
        self._q.insert(i, (item, priority))

    def peek(self) -> tuple[Any, float]:
        """Returns the next item to be dequeued"""
        return self._q[-1]

    def get(self) -> Any:
        """Dequeues the next item and returns its value"""
        return self._q.pop()[0]

    def drop(self) -> None:
        """Removes all items with zero priority from the queue"""
        self._q = [item for item in self._q if item[1]]

    def empty(self) -> bool:
        """Returns True if the queue is empty, otherwise False"""
        return len(self._q) == 0

    def __iter__(self):
        yield from self._q

    def __len__(self) -> int:
        return len(self._q)

    def __str__(self) -> str:
        return str(self._q)
