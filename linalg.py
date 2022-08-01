from __future__ import annotations
import numpy as np

numeric = int | float | complex | np.number

class Vector:
    def __init__(self, *args: tuple[numeric]) -> None:
        if len(args) <= 1:
            raise ValueError("Vector must be initialized with at least 2 dimensions")
        else:
            self.elems = args
    
    def __str__(self) -> str:
        return str(self.elems)
    
    def __len__(self) -> int:
        return len(self.elems)
    
    def __iter__(self):
        return self.elems.__iter__()
    
    def __add__(self, other: Vector | numeric) -> Vector:
        if isinstance(other, self.__class__):
            return Vector(*(x + y for x, y in zip(self, other)))
        elif isinstance(other, numeric):
            return Vector(*(x + other for x in self))
        else:
            raise TypeError("Only add vectors with numerics or other vectors")
    
    def __radd__(self, other: Vector | numeric) -> Vector:
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, self.__class__) and not np.isinstance(other, numeric):
            raise TypeError("Only subtract vectors with numerics or other vectors")
        return self.__add__(other * (-1))
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other) -> Vector | numeric:
        if isinstance(other, self.__class__):
            return self.scprod(other)
        elif isinstance(other, numeric):
            return self.scmult(other)
        else:
            raise TypeError("Only multiply vectors with numerics or other vectors")
    
    def __rmul__(self, other: Vector | numeric) -> Vector | numeric:
        return self.__mul__(other)
    
    def scmult(self, alpha: numeric) -> Vector:
        return Vector(*(x * alpha for x in self))
    
    def scprod(self, w: Vector) -> numeric:
        if self.__len__() != len(w):
            raise ValueError("Vectors must be of equal dimension")
            
        return sum(x * y for x, y in zip(self, w))


def main():
    # Basic boilerplate code
    
    v = Vector(3,2,8)
    w = Vector(3,2,2)
    print(v-w)
    

if __name__ == "__main__":
    main()
