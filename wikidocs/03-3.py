#1
print('---------------')
sum = 0
for i in range(1, 100+1):
    sum = sum + i
print(sum)


#2
print('---------------')
for i in range(1, 1000):
    sum = sum + i if i%5==0 else sum
print(sum)

#3
print('---------------')
A = [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
sum = 0
for i in A:
    sum = sum + i
print("Average :", sum/len(A))

#4
print('---------------')
student_blood = ['A', 'B', 'A', 'O', 'AB', 'AB', 'O', 'A', 'B', 'O', 'B', 'AB']
A = 0
B = 0
AB = 0
O = 0
for blood in student_blood:
    if blood == 'A':
        A = A + 1
    elif blood == 'B':
        B = B + 1
    elif blood == 'AB':
        AB = B + 1
    elif blood == 'O':
        O = B + 1
print("A: ", A)
print("B: ", B)
print("O: ", O)
print("AB: ", AB)


#5
print('---------------')
numbers = [1, 2, 3, 4, 5]
odd_list = [num * 2 for num in numbers if num % 2 == 1]
print(odd_list)

#6
print('---------------')
a = "Life is too short, you need python"
aeiou = "aeiou"
b = [i for i in a if i not in aeiou]

for i in b:
    print(i, end ='')
