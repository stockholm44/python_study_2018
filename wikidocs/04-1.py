#1
print('-------------------')

def is_odd(a):
    if a % 2 == 0:
        return "even"
    elif a % 2 == 1:
        return "odd"

print(is_odd(10))

#2
print('-------------------')

def average_num(*args):
    sum=0
    for i in args:
        sum = sum + i
    return sum/len(args)
b=[1,2,2,3]
print(average_num(1,2,2,3))


#3
print('-------------------')
def gugudan(n):
    for i in range(1,10):
        print("%d * %d = %d" %(n, i, n*i))

gugudan(2)

#4
print('-------------------')
def fib(n):
    sum = 0
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n > 1:
        sum = fib(n-1)+ fib(n-2)
        return sum
print("fib(0) = ",fib(0))
print("fib(1) = ",fib(1))
print("fib(2) = ",fib(2))
print("fib(3) = ",fib(3))
print("fib(4) = ",fib(4))


#5
print('-------------------')
result = [a for a in lambda b: if b]
result([1,2,3,4,5,6,7])
