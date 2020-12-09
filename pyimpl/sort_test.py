from inspect import getmembers, isfunction
from timeit import default_timer as timer
import random
import sorting

def testAll():
    methods = [x for x in getmembers(sorting) if isfunction(x[1])]
    for sort in methods:
        #Order(sort[1])
        Time(sort[1])


def Order(f):
    a = [9, 3, 4, 4, 5, 2, 4, 9, 6, 0, 11, 7, 1, 8, 1, 10]
    b = [0,1,1,2,3,4,4,4,5,6,7,8,9,9,10,11]
    print(f)
    assert f(a) == b

def Time(f):
    n = 10000
    x = [i for i in range(n)]
    random.shuffle(x)
    start = timer()
    f(x)
    end = timer()
    result = end-start
    print(f"Function {f} ran in {result} seconds with n = {n}")



if __name__ == "__main__":
    testAll()