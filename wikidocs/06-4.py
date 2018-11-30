# wikidocs 메모장 만들기
# python 06-2.py -a sadasdasjdska
import sys

option = sys.argv[1]

if option == '-a':
    memo = sys.argv[2]
    f = open("memo.txt", "a")
    f.write(memo)
    print(memo, "was added")
    f.write('\n')
    f.close()
elif option == '-v':
    f = open('memo.txt')
    memo = f.read()
    f.close()
    print(memo)
