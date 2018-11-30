'''
wikidocs
탭을 4개의 공백으로 바꾸자..
'''

import sys

if len(sys.argv) == 3:
    src = sys.argv[1]
    dst = sys.argv[2]

    f = open(src)
    tab_content = f.read()
    f.close()

    f = open(dst, 'w')
    space_content = tab_content.replace('\t', " " * 4)
    f.write(space_content)
    f.close()   
