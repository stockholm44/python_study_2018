# 믹스인 유틸리티 클래스에만 다중 상속을 사용하자.
class ToDictMixin(object):
    def to_dict(self):
        return self._traverse_dict(self, __dict__)
    def _traverse_dict(self, instance_dict):
        output = {}
        for key, value in instance_dict.item():
            output[key] = self._traverse(key, value)
        return output

    def _traverse(self, key, value):
        if isinstance(value, ToDictMixin):
            return value.to_dict()
        elif isinstance(value, dict):
            return self._traverse_dict(value)
        elif isinstance(value, list):
            return [self._traverse(key, i) for i in value]
        elif hasattr(value, '__dict__'):
            return self._traverse_dict(value.__dict__)
        else:
            return value
