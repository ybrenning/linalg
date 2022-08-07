from __future__ import annotations
import numpy as np

numeric = int | float | complex | np.number


class Vector:   
    def __init__(self, *args: numeric) -> None:
        if len(args) <= 1:
            raise ValueError("Vector must be initialized with at least 2 dimensions")
        else:
            self.elems = args

    def __str__(self) -> str:
        return str(self.elems)

    def __repr__(self) -> str:
        return f"Vector({repr(self.elems)})"

    def __len__(self) -> int:
        return len(self.elems)

    def __iter__(self):
        return self.elems.__iter__()

    def __add__(self, __o: object) -> Vector:
        if isinstance(__o, self.__class__):
            return Vector(*(x + y for x, y in zip(self, __o)))
        elif isinstance(__o, numeric):
            return Vector(*(x + __o for x in self))
        else:
            raise TypeError("Only add vectors with numerics or other vectors")

    def __radd__(self, __o: object) -> Vector:
        return self.__add__(__o)

    def __sub__(self, __o: object):
        if not isinstance(__o, self.__class__) and not np.isinstance(__o, numeric):
            raise TypeError("Only subtract vectors with numerics or other vectors")
        else:
            return self.__add__(__o * (-1))

    def __rsub__(self, __o: object):
        return self.__sub__(__o)

    def __mul__(self, __o: object) -> Vector | numeric:
        if isinstance(__o, self.__class__):
            return self.scprod(__o)
        elif isinstance(__o, numeric):
            return self.scmult(__o)
        else:
            raise TypeError("Only multiply vectors with numerics or other vectors")

    def __rmul__(self, __o: object) -> Vector | numeric:
        return self.__mul__(__o)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.elems == __o.elems
        else:
            raise TypeError("Comparators must both be of type Vector")

    def scmult(self, alpha: numeric) -> Vector:
        return Vector(*(x * alpha for x in self))

    def scprod(self, w: Vector) -> numeric:
        if self.__len__() != len(w):
            raise ValueError("Vectors must be of equal dimension")

        return sum(x * y for x, y in zip(self, w))


class Matrix:
    def __init__(self, *args: list[numeric]) -> None:
        if len(args) == 0:
            raise ValueError("Missing matrix initializer")
        for arg in args:
            if len(args[0]) != len(arg):
                raise ValueError("Matrix rows must be of equal length")

        self.elems = args

    def __str__(self) -> str:
        return ", \n".join((str(x) for x in self.elems))

    def __repr__(self) -> str:
        return f"Matrix({repr(self.elems)})"                        

    def __mul__(self, __o: object) -> Matrix:
        if isinstance(__o, self.__class__):
            cols_fst, rows_fst = len(self.elems[0]), len(self.elems)
            cols_snd, rows_snd = len(__o.elems[0]), len(__o.elems)
            
            if cols_fst == rows_snd:
                res = [[0 for _ in range(0, cols_snd)] for _ in range(0, rows_fst)]
                
                for i in range(0, rows_fst):
                    for j in range(0, cols_snd):
                        for k in range(0, cols_fst):
                            res[i][j] += self.elems[i][k] * __o.elems[k][j]
                
                return Matrix(*res)
            else:
                raise ValueError("Number of cols in first matrix must be equal to number of rows in second matrix")
        else:
            raise TypeError("Matrix multiplication not yet implemented for calculations other than Matrix * Matrix")

    def det(self) -> numeric:
        if len(self.elems) != len(self.elems[0]):
            raise ValueError("Determinant can only be calculated for a square Matrix of size nxn")

        if len(self.elems) == 2:
            return self.__det2d()
        elif len(self.elems) == 3:
            return self.__det3d()

    def __det2d(self) -> numeric:
        return self.elems[0][0] * self.elems[1][1] - self.elems[0][1] * self.elems[1][0]

    def __det3d(self) -> numeric:
        return (
            self.elems[0][0] * self.elems[1][1] * self.elems[2][2]
            + self.elems[0][1] * self.elems[1][2] * self.elems[2][0]
            + self.elems[0][2] * self.elems[1][0] * self.elems[2][1]
            - self.elems[0][2] * self.elems[1][1] * self.elems[2][0]
            - self.elems[0][1] * self.elems[1][0] * self.elems[2][2]
            - self.elems[0][0] * self.elems[1][2] * self.elems[2][1]
        )

def main() -> None:
    # Basic boilerplate code

    v = Vector(3, 2, 8)
    w = Vector(3, 2, 2)
    print(v)
    print(v - w)
    print(repr(v))

    x = Vector(3, 2, 8)
    print(x == v)

    a = Matrix([1, 2], [0, 1], [4, 0])
    print(a)
    print(repr(a))
    # print(a.det())
    
    b = Matrix([3, 2, 1], [1, 0, 2])
    print(a * b)


if __name__ == "__main__":
    main()
