#wikidocs 02-5

#1
a = {'name':'홍길동','birth':1128,'age':30}
print("1.",a)
#2
a = dict()
print(a)

print("2. list cannot be key")

#3
a = {'A':90, 'B':80, 'C':70}
print("3.",a['B'])
del a['B']
print(a)

#4
a = {'A':90, 'B':80}
print("4.",a.get('C',70))

#5
a = {'A':90, 'B':80, 'C':70}
a_min = min(a.values())
print("5.", a_min)

#6
a = {'A':90, 'B':80, 'C':70}
a_list = list(a.items())
print("6.",a_list)
