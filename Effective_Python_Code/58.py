# Example 1
def insertion_sort(data):
    result = []
    for value in data:
        insert_value(result, value)
    return result


# Example 2
def insert_value(array, value):
    for i, existing in enumerate(array):
        if existing > value:
            array.insert(i, value)
            return
    array.append(value)


# Example 3
from random import randint

max_size = 10**4
data = [randint(0, max_size) for _ in range(max_size)]
test = lambda: insertion_sort(data)



# Example 4
from cProfile import Profile

profiler = Profile()
profiler.runcall(test)


# Example 5
import sys
from pstats import Stats

stats = Stats(profiler)
# stats = Stats(profiler, stream=STDOUT)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()
