# wikidocs Sort.

# 객체로 이루어진 리스트를 소트
# class Student:
#     def __init__(self, name, age, grade):
#         self.name = name
#         self.age = age
#         self.grade = grade
#
#     def __repr__(self):
#         return repr((self.name, self.age, self.grade))
#
# # student = Student('jane',22,'A')
# # print(student.name)
# # print(student.age)
# # print(student)
#
# student_objects = [
#     Student('jane',22,'A'),
#     Student('dave',32,'B'),
#     Student('sally',17,'A'),
# ]
# print(student_objects)


# operator 모듈
# # itemgetter사용
# from operator import itemgetter
#
# students = [
#     ("jane", 22, 'A'),
#     ("dave", 32, 'B'),
#     ("sally", 17, 'B'),
# ]
# # print(students)
# result = sorted(students, key = itemgetter(1))

# attrgetter
from operator import attrgetter

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def __repr__(self):
        return repr((self.name, self.age, self.grade))

student_objects = [
    Student("jane", 22, 'A'),
    Student("dave", 32, 'B'),
    Student("sally", 17, 'B'),
]
# print(students)
result = sorted(student_objects, key =attrgetter('age'))
result2 = sorted(result, key=attrgetter('grade'))
print(result2)
