# wikidocs 07-2 정규표현식 시작하기.

import re

p = re.compile('[a-z]+')
# m = p.search("python")
# print(m)
# m = p.search("3 python")
# print(m)

# result = p.findall("life is to short")
# print(result)
#
# result = p.finditer("life is to short")
# print(result)
# for r in result: print(r)

m = p.match("python")
m.group()


.*[.].*
