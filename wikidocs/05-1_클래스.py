# wikidocs

#1
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val): # --> self 넣어야함.
        self.value += val

cal = Calculator()
cal.add(3)
cal.add(4)

print(cal.value)

#2
class Calculator:
    def __init__(self, init_value):
        self.value = init_value

    def add(self, val):
        self.value += val

cal = Calculator(0) # --> init value 설정
cal.add(3)
cal.add(4)

print(cal.value)

#3
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val

class UpgradeCalculator(Calculator):
    def minus(self, val):
        self.value -= val

cal = UpgradeCalculator()
cal.add(10)
print(cal.value)
cal.minus(7)
print(cal.value)  # 10에서 7을 뺀 3을 출력

#4
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val

class MaxLimitCalculator(Calculator):
    def add(self, val):
        self.value += val
        # if self.value > 100:
        #     self.value = 100
        self.value = self.value if self.value < 100 else 100

cal = MaxLimitCalculator()
cal.add(50)  # 50 더하기
print(cal.value)
cal.add(60)  # 60 더하기
print(cal.value)

#5
class Calculator:
    def __init__(self, list):
        self.list = list
    def sum(self):
        self.sum = 0
        for i in self.list:
            self.sum += i
        return self.sum
    def avg(self):
        # self.sum = 0
        # for i in self.list:
        #     self.sum += i
        return self.sum/len(self.list)

cal1 = Calculator([1,2,3,4,5])
print(cal1.sum())  # 15 출력
print(cal1.avg())  # 3.0 출력

cal2 = Calculator([6,7,8,9,10])
print(cal2.sum())  # 40 출력
print(cal2.avg())  # 8.0 출력
