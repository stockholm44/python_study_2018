#Wikidocs 02-2 Practice

#1
print('1. "Jump to python" Solve the practice')

#2
print("2. Life is short\nYou need Python")

#3
print("3."," "*24,"PYTHON")

#4
civ_num = "881120-1068234"
civ_list=civ_num.split("-")
print("4.\n1)",civ_list[0], "\n2)",civ_list[1])

#5
pin = "881120-1068234"
pin_split = pin.split("-")
sex = pin_split[1][0]
print("5.",sex)

#6
a = "1980M1120"
a_change = a[4]+a[0:4]+a[5:]
print("6.",a_change)

#7
cc = "PYTHON"
cc_format_add = "%30s" %cc
print("7.",cc_format_add)

#8
dd = "Life is too short, you need python"
dd_short_index = dd.find("short")
print("8.",dd_short_index)

#9
ee = "a:b:c:d"
ee_replace = ee.replace(":", "#")
print("9.", ee_replace)
