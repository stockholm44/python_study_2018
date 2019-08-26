# 27. 공개속성 보다는 비공개 속성을 사용하자
class MyObject(object):
    def __init__(self):
        self.public_field = 5
        self.__private_field = 10

    def get_private_field(self):
        return self.__private_field

foo = MyObject()
assert foo.public_field == 5
assert foo.get_private_field() == 10

# foo.__private_field

class MyOtherObject(object):
    def __init__(self):
        self.__private_field = 71

    @classmethod
    def get_private_field_of_instance(cls, instance):
        return instance.__private_field

bar = MyOtherObject()
assert MyOtherObject.get_private_field_of_instance(bar) == 71

class MyParentObject(object):
    def __init__(self):
        self.__private_field = 71

class MyChildObject(MyParentObject):
    def get_private_field(self):
        return self.__private_field

baz = MyChildObject()
# baz.get_private_field()

assert baz._MyParentObject__private_field ==71
print(baz.__dict__)


# 잘못된 접근방식
class MyClass(object):
    def __init__(self, value):
        self.__value = value

    def get_value(self):
        return str(self.__value)

foo = MyClass(5)
assert foo.get_value() == '5'


class MyIntegerSubclass(MyClass):
    def get_value(self):
        return int(self._MyClass__value)

foo = MyIntegerSubclass(5)
assert foo.get_value() == 5

class MyBaseClass(object):
    def __init__(self, value):
        self.__value = value

    def get_value(self):
        return str(self.__value)

class MyClass(MyBaseClass):
    def __init__(self, value):
        # 사용자가 객체에 전달한 값을 저장한다.
        # 문자열로 강제할 수 있는 값이어야 하며,
        # 객체에 할당하고 나면 불변으로 취급해야 한다.
        self.__value = value
    def get_value(self):
        return str(self.__value)

class MyIntegerSubclass(MyClass):
    def get_value(self):
        return int(self.__MyClass__value)

# foo = MyIntegerSubclass(5)
# foo.get_value()



class ApiClass(object):
    def __init__(self):
        self._value =  5

    def get(self):
        return self._value

class Child(ApiClass):
    def __init__(self):
        super().__init__()
        self._value = 'hello' # 충돌

a = Child()
print(a.get(), 'and', a._value, 'should be different')

#속성이름안겹치게하자
class ApiClass(object):
    def __init__(self):
        self.__value = 5

    def get(self):
        return self.__value

class Child(ApiClass):
    def __init__(self):
        super().__init__()
        self._value = 'hello' # OK

a = Child()
print(a.get(), 'and', a._value, 'are different')
