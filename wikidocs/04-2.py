# wikidocs 04-2

# #1
# input1 = input("첫번째 숫자를 입력하세요:")
# input2 = input("두번째 숫자를 입력하세요:")
#
# total = int(input1) + int(input2)
# print("두 수의 합은 %d 입니다" % total)
#
#2
input3 = input("숫자들을 입력하세요:")
input_list = input3.split(",")
sum = 0
for i in input_list:
    sum = sum + int(i)
print(sum)

#3
print("you" "need" "python")
print("you"+"need"+"python")
print("you", "need", "python")
print("".join(["you", "need", "python"]))

#4
gugudan_num = int(input("구구단을 출력할 숫자를 입력하세요(2~9):"))
for i in range(1,10):
    print(gugudan_num * i, end=' ')
