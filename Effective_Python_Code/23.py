names = ['Socrates', 'Archimedes','Plato','Aristotle']
names.sort(key=lambda x: len(x))
print(names)

from collections import defaultdict

def log_missing():
    print('Key Added')
    return 0
current = {'green':12, 'blue':3}
increments = [
    ('red',5),
    ('blue', 17),
    ('orange',9),
]
result = defaultdict(log_missing, current)
print('Before:', dict(result))
for key, amount in increments:
    result[key] += amount
print('After:', dict(result))



def increment_with_report(current, increments):
    added_count = 0

    def missing():
        nonlocal added_count # 상태보존 클로저
        added_count += 1
        return 0
    result = defaultdict(missing, current)
    for key, amount in increments:
        result[key] += amount

    return result, added_count
result, count = increment_with_report(current, increments)
# assert count ==2
print(result, count)


class CountMissing(object):
    def __init__(self):
        self.added = 0

    def missing(self):
        self.added += 1
        return 0

counter = CountMissing()
result = defaultdict(counter.missing, current)

for key, amount in increments:
    result[key] += amount
assert counter.added == 2
print(dict(result), counter.added)

# __call__ 메서드이용
class BetterCountMissing(object):
    def __init__(self):
        self.added = 0

    def __call__(self):
        self.added += 1
        return 0

counter = BetterCountMissing()
counter()
assert callable(counter)

counter = BetterCountMissing()
result = defaultdict(counter, current) #__call__이 필요함.
for key, amount in increments:
    result[key] += amount
assert counter.added == 2
