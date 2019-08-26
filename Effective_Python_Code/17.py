def normalize(numbers):
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result

visits = [15, 35, 80]
percentages = normalize(visits)
print(percentages)

# Taesun's generator function
def normalize_iter(numbers):
    total = sum(numbers)
    for value in numbers:
        percent = 100 * value / total
        yield percent

print(list(normalize_iter(visits)))

# Books
def read_visits(data_path):
    with open(data_path) as f:
        for line in f:
            yield int(line)

import os
import sys
file_path = os.path.join(sys.path[0], 'tmp', 'my_numbers.txt')
it = read_visits(file_path)
percentages = normalize(it)
print(percentages)

it = read_visits(file_path)
print(list(it))
print(list(it))

def normalize_copy(numbers):
    numbers = list(numbers) # Copy Iterator
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result

it = read_visits(file_path)
percentages = normalize_copy(it)
print(percentages)

def normalize_func(get_iter):
    total = sum(get_iter()) # New iterator
    result = []
    for value in get_iter(): # New iterator
        percent = 100 * value / total
        result.append(percent)
    return result
print('111')
percentages = normalize_func(lambda: read_visits(file_path))
print(percentages)

class ReadVisits(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                yield int(line)

print('222')
vistis = ReadVisits(file_path)
percentages = normalize(visits)
print(percentages)

def normalize_defensive(numbers):
    if iter(numbers) is iter(numbers): # Iterator --deny
        raise TypeError('Most supply a container')
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result
