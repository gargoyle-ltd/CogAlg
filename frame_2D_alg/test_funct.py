import numpy.ma as ma
import numpy as np

a = [[0, 7, 7, 6, 5, 0, 9, 0], [0, 7, 7, 6, 5, 0, 9, 0], [0, 7, 7, 6, 5, 0, 9, 0]]

b = ma.array([1, 2, 3, 4, 0, 0, 0, 4], mask=[0, 0, 0, 0, 1, 1, 1, 0])
print(type(b), b)

for i in a, b:
    if  isinstance(i, np.ndarray):
        print(True, type(i), i)
    else:
        print('No mask', type(i), i)

an = np.array([[1, 2, 3, 4, 5],
               [5, 4, 3, 2, 1],
               [1, 1, 1, 1, 1],
               [1, 2, 3, 4, 5],
               [5, 4, 3, 2, 1]])

print(an)
for i in an:
    new_list = list(map(int, i))
    print(new_list)