# wikidocs 외장함수
# https://wikidocs.net/33
'''
sys
명령 행에서 인수 전달하기 - sys.argv
강제로 스크립트 종료하기 - sys.exit
자신이 만든 모듈 불러와 사용하기 - sys.path
pickle  * 쓸모있나...?
os
내 시스템의 환경 변수값을 알고 싶을 때 - os.environ
디렉터리 위치 변경하기 - os.chdir
디렉터리 위치 리턴받기 - os.getcwd
시스템 명령어 호출하기 - os.system
실행한 시스템 명령어의 결과값 리턴받기 - os.popen
기타 유용한 os 관련 함수
shutil
파일 복사하기 - shutil.copy(src, dst)
glob
디렉터리에 있는 파일들을 리스트로 만들기 - glob(pathname)
tempfile
time * 타임쪽은 마스터하장.
time.time
time.localtime
time.asctime
time.ctime
time.strftime
time.sleep
calendar
calendar.weekday
calendar.monthrange
random
webbrowser
namedtuple *
defaultdict
threading * 멀티 스레드를 가능하게 하는 모듈.
'''
#
# a = ("Kll", 25, "Programmer")
# b = ("Chulsu", 32, "Manager")
# c = ("Young", 41, "Designer")
#
# for person in [a, b, c]:
#     print("Name:%s" % person[0])
#     print("age:%s" % person[1])
#     print("Job:%s" % person[2])
#
# class Person:
#     def __init__(self, name, age, job):
#         self.name=name
#         self.age=age
#         self.job=job
#
# a = Person('Hong',25,"Programmer")
# b = Person('Yoo',33,'Engineer')
#
# for person in [a,b]:
#     print('Name:%s' % person.name)
#     print('Age:%d' % person.age)
#     print('Job:%s' % person.job)

# from collections import namedtuple
# Person = namedtuple("Person", ["name","age","job"])
#
# a = Person("Hong", 25, "Programmer")
# print(a)
# for person in [a]:
#     print("name: %s" % person.name)

# import threading
# import time
# def say(msg):
#     while True:
#         time.sleep(1)
#         print(msg)
#
# for msg in ['you','need','python']:
#     t = threading.Thread(target=say, args=(msg,))
#     t.daemon = True
#     t.start()
#
# for i in range(100):
#     time.sleep(0.1)
#     print(i)


#1 sys.args
import sys
# print(sys.argv[1:])
sum = 0
for i in sys.argv[1:]:
    sum += int(i)
print(sum)

#1 class로.
import sys
class Sum:
    def __init__(self, *arg):
        self.arg = arg
    def sum_arg(self, arg):
        sum = 0
        for i in arg[1:]:
            sum += int(i)
        return sum

a = Sum()
print(a.sum_arg(sys.argv))

#2 os
import os
os.chdir("C:\djangocym\study_2018\wikidocs")
os.system("dir")
f = os.popen("dir")
print(f.read())

#3 glob
import glob
a = glob.glob("C:/djangocym/study_2018/wikidocs/*.py")
print(a)

#4 time
import time
now = time.time()
print(now)
now_str = time.strftime('%Y/%m/%d %X', time.localtime(time.time()))
print(now_str)

#5 random
import random

def rand_lotto(data):
    number = random.randint(0, len(data)-1)
    return data.pop(number)

data = list(range(1, 46))
for i in range(6):
    print(rand_lotto(data))

#6 namedtuple
from collections import namedtuple
Student = namedtuple("Student" ,["name", "score"])
a = Student(name="홍길동", score=20)
print(a)
print(a.name)
print(a.score)
