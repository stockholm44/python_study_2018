# 1부터 10,000까지 8이라는 숫자가 총 몇번 나오는가?
#
# 8이 포함되어 있는 숫자의 갯수를 카운팅 하는 것이 아니라 8이라는 숫자를 모두 카운팅 해야 한다.
# (※ 예를들어 8808은 3, 8888은 4로 카운팅 해야 함)

num_list = list(range(10001))
num_8_count = 0
for num in num_list:
    num_str_list = list(str(num))
    for n in num_str_list:
        if n == '8':
            num_8_count += 1

print(num_8_count)
