#wikidocs 02-3

#1
a = ['Life', 'is', 'too', 'short', 'you', 'need', 'python']
print("1.",a[a.index("you")], a[a.index("too")])

#2
b = ['Life', 'is', 'too', 'short']
print("2.",a[0],a[1],a[2],a[3])

#3
a = [1, 2, 3]
len_a = len(a)
print("3.",len_a)

#4
a = [1,2,3]
b = [4,5]
a.append(b)
# a_extend = a.extend(b)
# print("4. a_append: ", a_append)
# print("a_extend: ", a_extend)
print("4. append adding list, extend adding elements")
print(a)
a = [1,2,3]
b = [4,5]
a.extend(b)
print(a)

#5
a = [1,3,5,4,2]
a.sort()
a.reverse()
print("5.",a)

#6
a = [1,2,3,4,5]
a.remove(2)
a.remove(4)
print("6.",a)
