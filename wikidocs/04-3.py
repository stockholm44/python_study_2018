# # wikidocs 04-3 파일 읽고 쓰기.
#
# #1
# # f1 = open("test.txt", 'w')
# # f1.write("Life is too short")
# # f1.close()
# # f2 = open("test.txt", 'r')
# # print(f2.read())
#
# #2
# f=open("test.txt", 'a')
# a = input("저장할 내용을 입력하세요:")
# a = a + "\n"
# f.write(a)
# f.close()
#
# #3
# list = []
# f= open("C:/djangocym/study_2018/wikidocs/abc.txt", 'r')
# while 1:
#     line = f.readline()
#     if not line: break
#     list.append(line)
# f.close
#
# list.reverse()
# print(list)
# f =open("C:/djangocym/study_2018/wikidocs/abc.txt", 'w')
# for i in list:
#     data = i
#     f.write(data)
# f.close()

#4
list = []
f = open("C:/djangocym/study_2018/wikidocs/test.txt", 'r')
lines = f.readlines()
print(type(lines))
print(lines)
for line in lines:
    # print(line)
    if "java" in line:
        line = line.replace("java", "python")
    list.append(line)
f.close

f = open("C:/djangocym/study_2018/wikidocs/test.txt", 'w')
for line in list:
    f.write(line)
f.close

# #5
# sum = 0
# f=open("C:/djangocym/study_2018/wikidocs/sample.txt", 'r')
# lines=f.readlines()
# for line in lines:
#     sum = sum + int(line)
# avg = sum/len(lines)
# f.close()
# f=open("C:/djangocym/study_2018/wikidocs/result.txt", 'w')
# sum_char = "Sum is " + str(sum) + "\n"
# avg_char = "Average is " + str(avg)
# f.write(sum_char)
# f.write(avg_char)
# f.close()
