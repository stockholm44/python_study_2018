#wikidocs

#1
a = [1,2,3]
b = [1,2,3]
print(a is b, "->They have different memory")

#2
a = [1,2,3]
b = a
print(a is b, "->They share same memory")

#3
a = b = [1,2,3]
a[1] = 4
print(b, "-> They hace same memory")

#4
a = [1,2,3]
b = a[:]
print(a is b,"-> a[:] means b is only copied element of a")

#5
a = [1,2,3]
b = a[:]
a[1] = 4
print(a)
print(b,"-> it will be [1,2,3], becuz a[1] = 4 can only changing a's element")

#6
print("6")
a = [1,2,3]
a = a + [4,5]
print("6. a + [4,5] : ",a)
a = [1,2,3]
a.extend([4,5])
print("a.extend([4,5]): ",a)

#7
a = [1,[2,3],4]
b = a[:]
a[1][0]= 5 # ->[1,[5,3],4]
print(b,"-> I dont know. The answer is '[:] is swallow copy.' ")
