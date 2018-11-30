'''
http://codingdojang.com/scode/473?answer_mode=hide

디지털 시계에 하루동안(00:00~23:59) 3이 표시되는 시간을 초로 환산하면 총 몇 초(second) 일까요?

디지털 시계는 하루동안 다음과 같이 시:분(00:00~23:59)으로 표시됩니다.

00:00 (60초간 표시됨)
00:01
00:02
...
23:59
'''


import time


# 총합을 구할 변수 초기화
sum_include_3 = 0

# UTC 0에 00:00~23:59를 위한 시간의 범위를 반복시킴.
for i in range(0, 24 * 60 - 1):
    b = time.localtime(0 + 60 * i)
    a = time.strftime('%H:%M', b) # XX:XX형태로 출력
    c = list(str(a)) # 그 출력의 각각 글자를 리스트로 저장

    if '3' in c:
        # 해당시간을 초로 환산
        time_second = 60
        # print(b, time_second)
        sum_include_3 += time_second
print("The sum of times which includes 3 is -->", sum_include_3)
