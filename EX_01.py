<종합문제 1번> 1_1 김씨 박씨 출력

name= '이의덕,이재명,권종수,이재수,박철호,강동희,이재수,김지석,최승만,이성만,박영희,박수호,전경식,송우환,김재식,이유정'
name_list=name.split(",")
kim=0
park=0

for name in name_list:
    first_name=list(name)
    if first_name[0] == "김":
        kim += 1
    elif first_name[0] == "박":
        park +=1

print("김 씨 :", kim,"\n" + "박 씨 :",park)


 #<종합문제 1번> 1_2 "이재수" 갯수 출력


name= '이의덕,이재명,권종수,이재수,박철호,강동희,이재수,김지석,최승만,이성만,박영희,박수호,전경식,송우환,김재식,이유정'
name_list=name.split(",")

lee_name = list(filter(lambda x:x == "이재수", name_list))
print(len(lee_name))
'''

'''
 #<종합문제 1번> 1_3 중복을 제거한 이름 출력

name= '이의덕,이재명,권종수,이재수,박철호,강동희,이재수,김지석,최승만,이성만,박영희,박수호,전경식,송우환,김재식,이유정,이유정,이유정,이유정'
name_list=name.split(",")
new_name_list= set(name_list)
print(new_name_list)
'''
'''# <종합문제 1번> 1_4 중복을 제거한 이름 오름차순 출력

name= '이의덕,이재명,권종수,이재수,박철호,강동희,이재수,김지석,최승만,이성만,박영희,박수호,전경식,송우환,김재식,이유정,이유정,이유정,이유정'
name_list=name.split(",")
new_name_list= set(name_list)
new_name_list= list(new_name_list)
new_name_list.sort()
print(new_name_list)
'''

'''# <종합문제 2번> 합의 제곱과 제곱의 합 차.
def square(n):
    sum=0
    for i in range(1,n+1):
        sum += i*i
    return sum
def sum_square(n):
    sum=0
    for i in range(1,n+1):
        sum += i
    return sum*sum

print(sum_square(100) - square(100))
'''

'''# <종합문제 3번> 1부터 100까지 각 숫자의 갯 수 구하기

a=[]
b=[]
num_count=[0]*9

for i in range(1,101):
    b=list(str(i))
    for j in b:
        a.append(j)

a=list(map(int,a))

for j in range(0, 10):
    print(j,"의 갯수 :" , a.count(j) , "\n")
'''


'''# <종합문제 4번> DashInsert

insert_number=input("숫자 입력 \n")
list_number = list(insert_number)
i=1
j=len(list_number)
while i < j:
    if int(list_number[i]) % 2 == 0 and int(list_number[i-1]) % 2 == 0:
        list_number.insert(i, '*')
        i += 1
        j += 1
    elif int(list_number[i]) % 2 == 1 and int(list_number[i-1]) % 2 == 1:
        list_number.insert(i, '-')
        i += 1
        j += 1
    i += 1
print(''.join(list_number))
'''


'''# <종합문제 5번> 문자열 압축하기.
input_list=list(input("문자 입력 \n"))
count=1
result=[]
i=0
while i < len(input_list):
    print(i)
    result.append(input_list[i])
    if i+1 < len(input_list) and input_list[i]==input_list[i+1]:
        count+=1
        i+=1
        j = i + 1
        while j <len(input_list):
                if input_list[i] == input_list[j]:
                    count+=1
                    j+=1
                    i+=1
                else:
                    break
    i+=1
    result.append(count)
    count=1

print(result)
'''
'''# <종합문제 6번> Duplicate Numbers

number_list = list(input("숫자입력 \n").split(" "))
result=[]
h=0
for i in number_list:
    i_list=list(map(int,i))
    result.append('True')
    print(i_list)
    for j in range(0,10):
        if i_list.count(j) != 1:
            result[h]='False'
    h+=1

print(result)

'''
'''# <종합문제 7번> 모스 부호 해독
mos = input("모스부호 입력 \n")
mos_list = mos.split("  ")
print(mos_list)

result = []
for j in mos_list:
    mos_list2=j.split(" ")
    for i in mos_list2:
        if i == '.-':
            result.append('A')
        elif i == '-...':
            result.append('B')
        elif i == '-.-.':
            result.append('C')
        elif i == '-..':
            result.append('D')
        elif i == '.':
            result.append('E')
        elif i == '..-.':
            result.append('F')
        elif i == '--.':
            result.append('G')
        elif i == '....':
            result.append('H')
        elif i == '..':
            result.append('I')
        elif i == '.---':
            result.append('J')
        elif i == '-.-':
            result.append('K')
        elif i == '.-..':
            result.append('L')
        elif i == '--':
            result.append('M')
        elif i == '-.':
            result.append('N')
        elif i == '---':
            result.append('O')
        elif i == '.--.':
            result.append('P')
        elif i == '--.-':
            result.append('Q')
        elif i == '.-.':
            result.append('R')
        elif i == '...':
            result.append('S')
        elif i == '-':
            result.append('T')
        elif i == '..-':
            result.append('U')
        elif i == '...-':
            result.append('V')
        elif i == '.--':
            result.append('W')
        elif i == '	-..-':
            result.append('X')
        elif i == '-.--':
            result.append('Y')
        elif i == '--..':
            result.append('Z')
    result="".join(result)
    print(result,end=' ')
    result=[]
'''

'''#<정규식 1>다음 중 정규식 a[.]{3,}b과 매치되는 문자열은 무엇일까?
[.] -> 문자 '.' 그대로를 표현
{n,} --> 앞 문자 n번이상, 무한
'''

'''#<정규식 2>  소문자로 시작하는 문자, m.start  --> 시작하는 index 위치 end 끝나는 위치
import re
p = re.compile("[a-z]+")
m = p.search("5 python")
print(m.start() + m.end())
'''

'''#<정규식 3>
import re

s = """
park 010-9999-9988
kim 010-9909-7789
lee 010-8789-7768
"""

pat = re.compile("(\d{3}[-]\d{4})[-]\d{4}")
result = pat.sub("\g<1>-####", s)

print(result)
'''

''' #<정규식 4>
import re

pat = re.compile(".*[@].*[.](?=com$|net$).*$")

print (pat.match("pahkey@gmail.com"))
print (pat.match("kim@daum.net"))
print (pat.match("lee@myhome.co.kr"))
'''
'''#소트
from operator import itemgetter

students = [
    ("홍길동", 22),
    ("김철수", 32),
    ("박영희", 17),
]

students = sorted(students, key=itemgetter(1))

print(students)
'''

'''#[문제15] 시저 암호화

words = input("단어 입력 \n")
n = int(input(" n 값 입력 \n"))
result=""
words_list = list(words)

for i in words_list:
    c=ord(i)
    c+=n
    if c > 90:
        c =64 +(c - 90)
    result += chr(c)
print(result)
'''
