num_count = input("INSERT THE COUNT OF NUMBERS 2<= X <=11")
num_set = input("INSERT THE NUMBERS.")
num_list = num_set.split(' ')
operator_set = input("INSERT THE 4 OPERATOR COUNT NUMBER WHICH IS SMALLER THEN NUMBER SET.")
operator_list = operator_set.split(' ')

def operator_case_count(n):
    if n == 1:
        return 1
    elif n > 1:
        return (n - 1) * operator_case_count(n-1)

operator_case_count(5)

def operator_count_to_list(*args):
    operator_element_list.append(args[0]*
def operator_cases(*args):
    args[0] =


# 1. 각각의 input들 받기.
2. 연산자 배치수를 어떻게 할지.
3. 연산자를 배치하면 또 그걸 계산하게 하는 함수를 만들어야 할듯.
4. 숫자를 받으면 그 순서대로 해야하니까 reduce함수 이용하면 될 듯.

+ - * /
n-1
+ + -
경우의수  = n!개
1 2 3 4

a = 1
b = 2
c = 3

i = "+"
j = "-"


def operators(num1, num2, operator_num):
    if operator_num == 0:
        return plus(num1, num2)
    elif operator_num == 1:
        return minus(num1, num2)
    elif operator_num == 2:
        return multiple(num1, num2)
    elif operator_num == 3:
        return divide(num1, num2)

def plus(num1, num2):
    return num1 + num2

def minus(num1, num2):
    return num1 - num2

def multiple(num1, num2):
    return num1 * num2

def divide(num1, num2):
    return num1 / num2
