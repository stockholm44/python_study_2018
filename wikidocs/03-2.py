#1
print('---------------')
a = 1
sum = 0
while a<=100:
     sum = sum + a
     a = a +1
print(sum)

#2
print('---------------')
a = 1
sum = 0
while a <= 1000:
    sum = sum + a if a % 3 == 0 else sum
    a = a + 1
print(sum)

#3
print('---------------')
A = [20, 55, 67, 82, 45, 33, 90, 87, 100, 25]
i = 0
sum = 0
while i <= len(A)-1:
    sum = sum + A[i] if A[i] >=50 else sum
    i = i + 1
print(sum)

#4
print('---------------')
i = 1
while i <= 4:
    print("*" * i)
    i = i + 1

#5
print('5.---------------?????')
i = 7
while i >= 1:
    print(" " * int((7-i)/2) + "*" * i)
    i = i - 2
