# 06-2 3의 배수와 5의 배수를 1-1000사이에 더해주는거. 아니다. 그냥 class로 n을 받자.
class Times:
    def __init__(self, n):
        self.n = n
    def three_five_times(self):
        sum = 0
        # return sum = sum + i for i in range(1, self.n) if i % 3 ==0 or i % 5 == 0]
        # how can i make functional sum of for loop.
        for i in range(1, self.n):
            if i % 3 == 0 or i % 5 == 0:
                sum += i
        return sum
a = Times(1000)
print(a.three_five_times())

#새로운답
"""
class Sum:
    def __init__(self, n):
        self.n = n
    def Sum_Times(self):
        return sum([x for x in range(1, self.n) if x % 3 ==0 or x % 5 == 0])

a = Sum(1000)
print(a.Sum_Times())
"""
