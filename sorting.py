# December 2021
# Implement several sorting algorithms to be visualized

from random import shuffle
from time import perf_counter

def _test(func):
    """Decorator for running tests of sorting algorithms"""
    def wrapper() -> None:
        arr = list(range(1000))
        shuffle(arr)

        print(arr[:5])

        start = perf_counter()
        func(arr)
        end = perf_counter()

        print(arr[:5])
        print(f"Time: {(end-start):.5f}s")

    return wrapper

def selection(arr: list) -> None:
    """In place selection sort implementation, O(n^2)"""
    n = len(arr)

    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def insertion(arr: list) -> None:
    """In place insertion sort implementation, O(n^2)"""
    for i in range(1, len(arr)):
        j = i
        while arr[j] < arr[j-1] and j > 0:
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1

def merge(arr: list) -> None:
    """In place merge sort implementation, O(n*logn)"""
    if len(arr) > 1:
        mid = len(arr) // 2

        larr = arr[:mid]
        rarr = arr[mid:]

        merge(larr)
        merge(rarr)

        i = j = k = 0

        while i < len(larr) and j < len(rarr):
            if larr[i] < rarr[j]:
                arr[k] = larr[i]
                i += 1
            else:
                arr[k] = rarr[j]
                j += 1
            k += 1

        while i < len(larr):
            arr[k] = larr[i]
            i += 1
            k += 1

        while j < len(rarr):
            arr[k] = rarr[j]
            j += 1
            k += 1

def pigeonhole(arr: list) -> None:
    """In place pigeonhole sort implementation, O(n+k)"""
    min_val = min(arr)
    counts = [0]*(max(arr) - min_val + 1)

    for val in arr:
        counts[val-min_val] += 1

    i = 0
    for j, count in enumerate(counts):
        while count > 0:
            arr[i] = min_val + j
            count -= 1
            i += 1

def bubble(arr: list) -> None:
    """In place bubble sort implementation, O(n^2)"""
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True

        if not swapped:
            break

def gnome(arr: list) -> None:
    """In place gnome sort implementation, O(n^2)"""
    n = len(arr)
    i = 0

    while i < n:
        if i == 0:
            i += 1
        if arr[i-1] > arr[i]:
            arr[i-1], arr[i] = arr[i], arr[i-1]
            i -= 1
        else:
            i += 1

def bogo(arr: list) -> None:
    """In place bogosort implementation, O(inf)"""
    n = len(arr)

    while True:
        for i in range(1, n):
            if arr[i-1] > arr[i]:
                shuffle(arr)
                break
        else:
            break

def counting(arr: list) -> None:
    """In place counting sort implementation, O(n+k)"""
    n = len(arr)
    min_val = min(arr)
    counts = [0]*(max(arr) - min_val + 1)
    output = [0]*n

    for val in arr:
        counts[val-min_val] += 1

    for i in range(1, n):
        counts[i] += counts[i-1]

    for i in range(n-1, -1, -1):
        counts[arr[i]-min_val] -= 1
        output[counts[arr[i]-min_val]] = arr[i]

    for i in range(n):
        arr[i] = output[i]

def radix_counting(arr: list, exp: int) -> None:
    """In place counting sort implementation for radix sort subroutine, O(n+k)"""
    n = len(arr)
    counts = [0]*10
    output = [0]*n

    for val in arr:
        digit = (val // exp) % 10
        counts[digit] += 1

    for i in range(1, 10):
        counts[i] += counts[i-1]

    for i in range(n-1, -1, -1):
        digit = (arr[i] // exp) % 10
        counts[digit] -= 1
        output[counts[digit]] = arr[i]

    for i in range(n):
        arr[i] = output[i]

def radix(arr: list) -> None:
    """In place radix sort implementation, O(n*k)"""
    max_val = max(arr)
    exp = 1

    while max_val / exp > 1:
        radix_counting(arr, exp)
        exp *= 10

def median_of_three(arr: list, left: int, mid: int, right: int) -> int:
    """Finds the corresponding index to the median of three numbers, O(1)"""
    a, b, c = arr[left], arr[mid], arr[right]

    if (a-b)*(b-c) > 0:
        return mid
    if (a-b)*(a-c) > 0:
        return right
    return left

def partition(arr: list, left: int, right: int) -> int:
    """Partition subroutine for quicksort implementation"""
    pivot = median_of_three(arr, left, (left+right)//2, right)

    while left < right:
        while left < len(arr) and arr[left] <= arr[pivot]:
            left += 1
        while arr[right] > arr[pivot]:
            right -= 1

        if left < right:
            arr[left], arr[right] = arr[right], arr[left]

    arr[right], arr[pivot] = arr[pivot], arr[right]
    return right

def quick(arr: list, left: int = 0, right: int | None = None) -> None:
    """In place quicksort implementation, O(n^2)"""
    if right is None:
        right = len(arr) - 1

    if left < right:
        idx = partition(arr, left, right)

        quick(arr, left, idx-1)
        quick(arr, idx+1, right)

def main() -> None:
    """Code execution starts here"""

if __name__=="__main__":
    main()
