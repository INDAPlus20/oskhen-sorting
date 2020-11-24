from inspect import getmembers, isfunction
import sorting

def testAll():
    methods = [x for x in getmembers(sorting) if isfunction(x[1])]
    for sort in methods:
        Order(sort[1])


def Order(f):
    a = [9, 3, 4, 4, 5, 2, 4, 9, 6, 0, 11, 7, 1, 8, 1, 10]
    b = [0,1,1,2,3,4,4,4,5,6,7,8,9,9,10,11]
    print(f)
    assert f(a) == b


if __name__ == "__main__":
    testAll()