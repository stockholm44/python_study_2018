def index_words(text):
    result = []
    if text:
        result.append(0)
    for index, letter in enumerate(text):
        if letter == ' ':
            result.append(index + 1)
    return result

address = 'Four score and sevend years ago...'
result = index_words(address)
print(result[:3])
print(result)

def index_words_iter(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1

result = list(index_words_iter(address))
print(result)
print(index_words_iter(address))


def index_file(handle):
    offset = 0
    for line in handle:
        if line:
            yield offset
        for letter in line:
            offset += 1
            if letter == ' ':
                yield offset


import os
import sys
from itertools import islice
# print('Directory : ', sys.path[0])
# print('Join Result : ', os.path.join(sys.path[0], 'tmp', 'address.txt'))
#
with open(os.path.join(sys.path[0], 'tmp', 'address.txt'), 'r') as f:
    it = index_file(f)
    results = islice(it, 0, 3)
    print(list(results))
