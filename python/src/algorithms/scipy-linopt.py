from scipy.optimize import linprog

# https://realpython.com/linear-programming-python/#linear-programming-python-implementation

# maximise z = x + 2y
# st
# 2x + y <= 20
# -4x + 5y <= 10
# -x + 2y >= -2
# -x + 5y = 15
# x >= 0
# y >= 0

# Convert to standard form...
# minimize -z
# Ax <= b
# Bx = c
# x >= 0 (via bounds)

# 2x + y <= 20
# -4x + 5y <= 10
# x - 2y <= 2
# -x + 5y = 15

obj = [-1, -2]

A = [[ 2,  1],
     [-4,  5],
     [ 1, -2]]
b = [20,
     10,
     2]

B = [[-1, 5]]

c =[15]

bnd = [(0, float("inf")),
       (0, float("inf"))]

# opt = linprog(c=obj, A_ub=A, b_ub=b, A_eq=B, b_eq=c, bounds=bnd, method="revised simplex")
# opt = linprog(c=obj, A_ub=A, b_ub=b, A_eq=B, b_eq=c, bounds=bnd, method="interior-point")
opt = linprog(c=obj, A_ub=A, b_ub=b, A_eq=B, b_eq=c, bounds=bnd, method="simplex")
print(opt)


