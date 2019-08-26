from cProfile import Profile
import sys
from pstats import Stats

# Example 7
def my_utility(a, b):
    c = 1
    for i in range(100):
        c += a * b

def first_func():
    for _ in range(1000):
        my_utility(4, 5)

def second_func():
    for _ in range(10):
        my_utility(1, 3)

def my_program():
    for _ in range(20):
        first_func()
        second_func()


from cProfile import Profile



# Example 8

profiler = Profile()
profiler.runcall(my_program)
# stats = Stats(profiler, stream=STDOUT)
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()

# Example 9
# stats = Stats(profiler, stream=STDOUT)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_callers()
