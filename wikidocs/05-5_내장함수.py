# wikidocs 05-5 내장함수
# https://wikidocs.net/32
'''
abs
all
any
chr
dir
divmod *
enumerate **
eval *
filter **
hex
id
input *
int
isinstance
len *
list *
map **
max
min
oct
open *
ord
pow *
range *
round
sorted * sort함수는 그 리스트자체를 정렬, sorted는 새로운 리스트를 반환.
str
tuple
type
zip
'''

#1
abc = ['a','b','c']
dict_abc = {}
for i, a in enumerate(abc):
    dict_abc[i] = a
print(dict_abc)

#2
a = int("0xea", 16)
print(a)

#3
print(list(map(lambda x: x*3, [1,2,3,4])))

#4
a = [-8, 2, 7, 5, -3, 5, 0, 1]
print(min(a) + max(a))

#5
print(round(17/3, 4))

#6
a = [1,2,3,4]
b = ['a','b','c','d']
c = list(zip(a,b))
print(c)
