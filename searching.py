# January 2022
# Implement several seacrh algorithms to be visualized

from math import isqrt
from typing import Any

def linear(arr: list, target: Any) -> int:
    """Implements linear search to return the index of target in arr, or -1 if not present"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def binary(arr: list, target: Any, left: int = 0, right: int | None = None) -> int:
    """Implements binary search to return the index of target in sorted arr, or -1 if not present"""
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1

    mid = (left+right) // 2

    if arr[mid] == target:
        return mid
    if target < arr[mid]:
        return binary(arr, target, left, mid-1)

    return binary(arr, target, mid+1, right)

def jump(arr: list, target: Any) -> int:
    """Implements jump search to return the index of target in sorted arr, or -1 if not present"""
    n = len(arr)
    if n == 0:
        return -1

    step = isqrt(n)

    i = 0
    while arr[i] <= target:
        if arr[i] == target:
            return i
        i += step
        if i >= n:
            return -1

    for j in range(i-step+1, i):
        if arr[j] == target:
            return j
    return -1

def main() -> None:
    """Code execution starts here"""

if __name__=="__main__":
    main()
