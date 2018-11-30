#wikidocs 02-6

#1
aa = set(['a','b','c'])
print("1.", aa)

#2
a = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
a_set = set(a)
print("2.",a_set)

#3
s1 = set(['a', 'b', 'c', 'd', 'e'])
s2 = set(['c', 'd', 'e', 'f', 'g'])
print("3.",s1-s2)
print("or",s1.difference(s2))

#4
a = set([])
print("4. Empty Set: ",a,"Type of A is ",type(a))

#5
a = {'a','b','c'}
b = {'d','e','f'}
print("5. a add b :", a|b)
print("or",a.union(b))
