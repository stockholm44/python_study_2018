class OldResistor(object):
    def __init__(self,ohms):
        self._ohms = ohms

    def get_ohms(self):
        return self._ohms

    def set_ohms(self, ohms):
        self._ohms = ohms

r0 = OldResistor(50e3)
print('Before: %5r' % r0.get_ohms())
r0.set_ohms(10e3)
print('After: %5r' % r0.get_ohms())

r0.set_ohms(r0.get_ohms() + 5e3)
print('Finally: %5r', r0.get_ohms())

# 일반속성으로 구현시
class Resistor(object):
    def __init__(self, ohms):
        self.ohms = ohms
        self.voltage = 0
        self.current = 0
r1 = Resistor(50e3)
print(r1.ohms)
r1.ohms = 10e3
print(r1.ohms)

# 속성설정시 특별한 동작이 일어나야하면 @property 데코레이터사용
class VoltageResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)
        self._voltage = 0

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, voltage):
        self._voltage = voltage
        self.current = self._voltage / self.ohms

r2 = VoltageResistance(1e3)
print('Before : %5r amps' % r2.current)
r2.voltage = 10
print('After : %5r amps' % r2.current)

# 프로퍼티에 setter 설정시 타입체크하고 값검증가능.

class BoundedResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)

    @property
    def ohms(self):
        return self._ohms

    @ohms.setter
    def ohms(self, ohms):
        if ohms <= 0:
            raise ValueError('%f ohms must be >0' % ohms)
        self._ohms = ohms

r3 = BoundedResistance(1e3)
# r3.ohms = 0

# BoundedResistance(-5)

# 부모클래스의 속성을 불변으로 만드든데도 @property 사용가능
class FixedResistance(Resistor):
    @property
    def ohms(self):
        return self._ohms
    @ohms.setter
    def ohms(self, ohms):
        if hasattr(self, '_ohms'):
            raise AttributeError("Can't set attribute")
        self._ohms = ohms

r4 = FixedResistance(1e3)
print(r4.ohms)
# r4.ohms = 2e3
print(r4.ohms)

class MysteriousResistor(Resistor):
    @property
    def ohms(self):
        self.voltage = self._ohms * self.current
        return self._ohms
r7 = MysteriousResistor(10)
print(r7.ohms, r7.voltage, r7.current)
r7.current = 0.01
print('Before: %5r' % r7.voltage)
r7.ohms
print('After: %5r' % r7.voltage)
