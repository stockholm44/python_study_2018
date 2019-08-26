class FrequencyList(list):
    def __init__(self, members):
        super().__init__(members)

    def frequency(self):
        counts = {}
        for item in self:
            counts.setdefault(item, 0)
            counts[item] += 1
        return counts

foo = FrequencyList(['a','b','a','c','b','a','d'])
print('Length is,', len(foo))
foo.pop()
print('After pop:',repr(foo))
print('Frequency:',foo.frequency())


class BinaryNode(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

bar = [1,2,3]
print(bar[0])
bar.__getitem__(0)

class IndexableNode(BinaryNode):
    def _search(self, count, index):

    def __getitem__(self, index):
        fount, _ = self._search(0, index)
        if not found:
            raise IndexError('Index out of range')
        return fount.value
