from __future__ import annotations

numeric = int | float | complex


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

    def __add__(self, other: Vector | numeric) -> Vector:
        if isinstance(other, self.__class__):
            return Vector(*(x + y for x, y in zip(self, other)))
        elif isinstance(other, numeric):
            return Vector(*(x + other for x in self))
        else:
            raise TypeError("Only add vectors with numerics or other vectors")

    def __radd__(self, other: Vector | numeric) -> Vector:
        return self.__add__(other)

    def __sub__(self, other: Vector | numeric) -> Vector:
        if not isinstance(other, self.__class__) and not isinstance(other, numeric):
            raise TypeError("Only subtract vectors with numerics or other vectors")
        else:
            return self.__add__(other * (-1))

    def __rsub__(self, other: Vector | numeric) -> Vector:
        return self.__sub__(other)

    def __mul__(self, other: Vector | numeric) -> Vector | numeric:
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


def main():
    # Basic boilerplate code

    v = Vector(3, 2, 8)
    w = Vector(3, 2, 2)
    print(v - w)
    print(repr(v))

    a = Matrix(
        [1, 2, 3], 
        [4, 5, 6], 
        [7, 8, 9]
    )
    
    print(a)
    print(repr(a))


if __name__ == "__main__":
    main()
