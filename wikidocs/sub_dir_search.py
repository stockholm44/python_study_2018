'''
sub_dir_search.py
하위 디렉토리에서 python 파일만 검색하는 프로그램.
'''
import os

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if os.path.isdir(full_filename):
            search(full_filename)
        else:
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.py':
                print(full_filename)

search("C:\djangocym\ssdsa.")
