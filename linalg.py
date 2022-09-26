from __future__ import annotations
import math

from typing import Any, Iterator

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

    def __iter__(self) -> Iterator[numeric]:
        return self.elems.__iter__()

    def __add__(self, __o: object) -> Vector:
        if isinstance(__o, self.__class__):
            if len(__o) != self.__len__():
                raise ValueError("Vectors must be of equal dimension")
            return Vector(*(x + y for x, y in zip(self, __o)))
        elif isinstance(__o, numeric):
            return Vector(*(x + __o for x in self))
        else:
            raise TypeError("Only add vectors with numerics or other vectors")

    def __radd__(self, __o: object) -> Vector:
        return self.__add__(__o)

    def __sub__(self, __o: object) -> Vector:
        if not isinstance(__o, self.__class__) and not isinstance(__o, numeric):
            raise TypeError("Only subtract vectors with numerics or other vectors")
        else:
            return self.__add__(__o * (-1))

    def __mul__(self, __o: object) -> Vector | numeric:
        if isinstance(__o, self.__class__):
            return self.scprod(__o)
        elif isinstance(__o, numeric):
            return self.scmul(__o)
        else:
            raise TypeError("Only multiply vectors with numerics or other vectors")

    def __rmul__(self, __o: object) -> Vector | numeric:
        return self.__mul__(__o)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.elems == __o.elems
        else:
            raise TypeError("Comparators must both be of type Vector")

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def scmul(self, alpha: numeric) -> Vector:
        return Vector(*(x * alpha for x in self))

    def scprod(self, w: Vector) -> numeric:
        if self.__len__() != len(w):
            raise ValueError("Vectors must be of equal dimension")

        return sum(x * y for x, y in zip(self, w))

    def magnitude(self) -> numeric:
        mag = 0
        for i in range(0, self.__len__()):
            mag += self.elems[i] ** 2

        assert isinstance(mag, int) or isinstance(mag, float)
        return math.sqrt(mag)

    def norm(self) -> Vector:
        """Norm of the vector (https://en.wikipedia.org/wiki/Norm_(mathematics))

        Returns:
            Normed vector
        """

        return self.scmul(1 / self.magnitude())

    def orth_proj(self, w: Vector) -> Vector:
        """Orthogonal projection (https://en.wikipedia.org/wiki/Vector_projection)

        Args:
            w: Vector to be projected onto

        Returns:
            Projected vector
        """

        return self.scmul((self.scprod(w) / self.scprod(self)))


def orthogonalize(*args: Vector) -> tuple[Vector, ...]:
    """Orthogonalize a given set of vectors using the
    [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

    Args:
        *args: Set of linearly independent vectors

    Returns:
        An orthogonal set containing that spans the same k-dimensional subspace as the given set.
    """

    if len(args) < 2:
        raise ValueError("Please provide at least 2 vectors")

    vecs = [args[0]]
    for i in range(1, len(args)):
        vecs.append(args[i] - args[i - 1].orth_proj(args[i]))

    return tuple(vecs)


def orthonormalize(*args: Vector) -> tuple[Vector, ...]:
    """Orthonormalize a given set of vectors using the
    [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

    Args:
        *args: Set of linearly independent vectors

    Returns:
        An orthonormalized set containing that spans the same k-dimensional subspace as the given set.
    """

    if len(args) < 2:
        raise ValueError("Please provide at least 2 vectors")

    return tuple([v.norm() for v in orthogonalize(*args)])


class Matrix:
    def __init__(self, *args: list[numeric]) -> None:
        if len(args) == 0 or len(args[0]) == 0:
            raise ValueError("Missing matrix initializer")
        for arg in args:
            if len(args[0]) != len(arg):
                raise ValueError("Matrix rows must be of equal length")

        self.elems = args

    def __str__(self) -> str:
        return ", \n".join((str(x) for x in self.elems))

    def __repr__(self) -> str:
        return f"Matrix({repr(self.elems)})"

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and __o.elems == self.elems

    def __mul__(self, __o: object) -> Matrix:
        if isinstance(__o, self.__class__):
            cols_fst, rows_fst = len(self.elems[0]), len(self.elems)
            cols_snd, rows_snd = len(__o.elems[0]), len(__o.elems)

            if cols_fst == rows_snd:
                res: list[list[numeric]] = [
                    [0 for _ in range(0, cols_snd)] for _ in range(0, rows_fst)
                ]

                for i in range(0, rows_fst):
                    for j in range(0, cols_snd):
                        for k in range(0, cols_fst):
                            res[i][j] += self.elems[i][k] * __o.elems[k][j]

                return Matrix(*res)
            else:
                raise ValueError(
                    "Number of cols in first matrix must be equal to number of rows in second matrix"
                )
        elif isinstance(__o, numeric):
            args = [[x * __o for x in row] for row in self.elems]
            assert all(all(isinstance(x, numeric) for x in row) for row in args)
            return Matrix(*args)  # type: ignore
        else:
            raise TypeError(
                "Matrices can only be multiplied by other matrices or numerics."
            )

    def det(self) -> numeric:
        if len(self.elems) != len(self.elems[0]):
            raise ValueError(
                "Determinant can only be calculated for a square Matrix of size nxn"
            )

        if len(self.elems) == 2:
            return self.__det2d()
        elif len(self.elems) == 3:
            return self.__det3d()
        else:
            raise ValueError(
                "Determinant calculation not yet implemented for matrices larger than 3x3"
            )

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

    def inverse(self) -> Matrix:
        if self._isinvertable():
            if len(self.elems) == 2:
                pass
        else:
            raise ValueError("Matrix is not invertable")

    def _isinvertable(self) -> bool:
        return len(self.elems) == len(self.elems[0]) and self.det() != 0

    def transpose(self) -> Matrix:
        return Matrix(*map(list, zip(*self.elems)))


import unittest


class TestVector(unittest.TestCase):
    def test_vector_init(self) -> None:
        self.assertRaises(
            ValueError,
            Vector,
        )
        self.assertRaises(ValueError, Vector, 1)

    def test_vector_add(self) -> None:
        self.assertEqual(Vector(1, 2, 3) + Vector(1, 2, 3), Vector(2, 4, 6))
        self.assertEqual(
            Vector(1.5, 2.5, 3.5) + Vector(0.6, 1.2, 1.8), Vector(2.1, 3.7, 5.3)
        )
        self.assertEqual(
            Vector(3 + 4j, 2 + 3j, 7 + 1j) + Vector(6 + 1j, 5 + 2j, 1 + 3j),
            Vector(9 + 5j, 7 + 5j, 8 + 4j),
        )

        self.assertEqual(Vector(1, 2, 3) + 1, Vector(2, 3, 4))
        self.assertEqual(Vector(1.5, 2.5, 3.5) + 0.5, Vector(2.0, 3.0, 4.0))
        self.assertEqual(
            Vector(3 + 4j, 2 + 3j, 7 + 1j) + (5 + 2j), Vector(8 + 6j, 7 + 5j, 12 + 3j)
        )

        self.assertEqual(Vector(1, 2, 3) + 0.5, Vector(1.5, 2.5, 3.5))
        self.assertEqual(Vector(1, 2, 3) + (3 + 2j), Vector(4 + 2j, 5 + 2j, 6 + 2j))

        self.assertRaises(ValueError, Vector(1, 2, 3).__add__, Vector(1, 2))
        self.assertRaises(ValueError, Vector(1, 2).__add__, Vector(1, 2, 3))

        self.assertRaises(TypeError, Vector(1, 2, 3).__add__, "String")
        self.assertRaises(TypeError, Vector(1, 2, 3).__add__, "c")
        self.assertRaises(TypeError, Vector(1, 2, 3).__add__, [1, 2, 3])
        self.assertRaises(TypeError, Vector(1, 2, 3).__add__, (1, 2, 3))
        self.assertRaises(
            TypeError, Vector(1, 2, 3).__add__, {"one": 1, "two": 2, "three": 3}
        )

    def test_vector_radd(self) -> None:
        self.assertEqual(1 + Vector(1, 2, 3), Vector(1, 2, 3) + 1)
        self.assertEqual(0.5 + Vector(1, 2, 3), Vector(1, 2, 3) + 0.5)
        self.assertEqual((2 + 3j) + Vector(1, 2, 3), Vector(1, 2, 3) + (2 + 3j))

        self.assertRaises(TypeError, Vector(1, 2, 3).__radd__, "String")
        self.assertRaises(TypeError, Vector(1, 2, 3).__radd__, "c")
        self.assertRaises(TypeError, Vector(1, 2, 3).__radd__, [1, 2, 3])
        self.assertRaises(TypeError, Vector(1, 2, 3).__radd__, (1, 2, 3))
        self.assertRaises(
            TypeError, Vector(1, 2, 3).__radd__, {"one": 1, "two": 2, "three": 3}
        )

    def test_vector_sub(self) -> None:
        self.assertEqual(Vector(1, 2, 3) - Vector(1, 2, 3), Vector(0, 0, 0))
        self.assertEqual(Vector(1.5, 2.5, 3.5) - Vector(0.5, 0.5, 0.5), Vector(1, 2, 3))
        self.assertEqual(
            Vector(2 + 3j, 5 + 4j, 8 + 2j) - Vector(1 + 2j, 2 + 2j, 3 + 4j),
            Vector(1 + 1j, 3 + 2j, 5 - 2j),
        )

        self.assertEqual(Vector(1, 2, 3) - 1, Vector(0, 1, 2))
        self.assertEqual(Vector(1.5, 2.5, 3.5) - 1.5, Vector(0, 1, 2))
        self.assertEqual(
            Vector(2 + 3j, 5 + 4j, 8 + 2j) - (1 + 2j), Vector(1 + 1j, 4 + 2j, 7 + 0j)
        )

        self.assertRaises(ValueError, Vector(1, 2, 3).__sub__, Vector(1, 2))
        self.assertRaises(ValueError, Vector(1, 2).__sub__, Vector(1, 2, 3))

        self.assertRaises(TypeError, Vector(1, 2, 3).__sub__, "String")
        self.assertRaises(TypeError, Vector(1, 2, 3).__sub__, "c")
        self.assertRaises(TypeError, Vector(1, 2, 3).__sub__, [1, 2, 3])
        self.assertRaises(TypeError, Vector(1, 2, 3).__sub__, (1, 2, 3))
        self.assertRaises(
            TypeError, Vector(1, 2, 3).__sub__, {"one": 1, "two": 2, "three": 3}
        )

    def test_scmul(self) -> None:
        # Two-dimensional
        v = Vector(1, 2)
        double_v = v.scmul(2)
        for index, value in enumerate(double_v):
            self.assertEqual(value, v.elems[index] * 2)

        # Three-dimensional
        v = Vector(1, 2, 3)
        double_v = v.scmul(2)
        for index, value in enumerate(double_v):
            self.assertEqual(value, v.elems[index] * 2)

        negative_v = v.scmul(-1)
        for index, value in enumerate(negative_v):
            self.assertEqual(value, v.elems[index] * -1)

        decimal_v = v.scmul(0.5)
        for index, value in enumerate(decimal_v):
            self.assertEqual(value, v.elems[index] / 2)

        zero_v = v.scmul(0)
        for index, value in enumerate(zero_v):
            self.assertEqual(value, 0)

    def test_scprod(self) -> None:
        # Two-dimensional
        self.assertEqual(Vector(1, 2).scprod(Vector(3, 4)), 11)

        # Three-dimensional
        self.assertEqual(Vector(1, 2, 3).scprod(Vector(-7, 8, 9)), 36)
        self.assertEqual(Vector(1, 0, 0).scprod(Vector(0, 0, 1)), 0)

    def test_magnitude(self) -> None:
        pass

    def test_norm(self) -> None:
        pass

    def orth_proj(self) -> None:
        pass

    def orthogonolize(self) -> None:
        pass

    def orthonormalize(self) -> None:
        pass


class TestMatrix(unittest.TestCase):
    def test_matrix_init(self) -> None:
        self.assertRaises(
            ValueError,
            Matrix,
        )
        self.assertRaises(ValueError, Matrix, [])
        self.assertRaises(ValueError, Matrix, [1, 2], [3])
        self.assertRaises(ValueError, Matrix, [1], [2, 3])
        self.assertRaises(ValueError, Matrix, [1, 2], [])

        _ = Matrix([1, 2], [3, 4])
        _ = Matrix([1, 2, 3], [4, 5, 6])
        _ = Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9])
        _ = Matrix([1, 2, 3])
        _ = Matrix([1], [2], [3])

    def test_matrix_mul(self) -> None:
        # 2x2 Matrices with scalars
        self.assertEqual(Matrix([1, 2], [3, 4]) * 2, Matrix([2, 4], [6, 8]))
        self.assertEqual(
            Matrix([0.5, 1.5], [2.5, 3.5]) * 2, Matrix([1.0, 3.0], [5.0, 7.0])
        )

        # 3x3 Matrices with scalars
        self.assertEqual(
            Matrix([1, 2, 3], [4, 5, 6]) * 2, Matrix([2, 4, 6], [8, 10, 12])
        )
        self.assertEqual(
            Matrix([0.5, 1.5, 2.5], [3.5, 4.5, 5.5]) * 2,
            Matrix([1.0, 3.0, 5.0], [7.0, 9.0, 11.0]),
        )

        # Matrix * Matrix
        self.assertEqual(
            Matrix([3, 2, 1], [1, 0, 2]) * Matrix([1, 2], [0, 1], [4, 0]),
            Matrix([7, 8], [9, 2]),
        )
        self.assertEqual(
            Matrix([3, 5, -1], [4, -8, 2]) * Matrix([0, 3, 1], [6, 5, 0], [2, -7, 3]),
            Matrix([28, 41, 0], [-44, -42, 10]),
        )

        self.assertRaises(
            ValueError,
            Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9]).__mul__,
            Matrix([1, 2], [3, 4]),
        )
        self.assertRaises(TypeError, Matrix([1, 2], [3, 4]).__mul__, "String")
        self.assertRaises(TypeError, Matrix([1, 2], [3, 4]).__mul__, [1, 2, 3])
        self.assertRaises(TypeError, Matrix([1, 2], [3, 4]).__mul__, (1, 2, 3))
        self.assertRaises(
            TypeError, Matrix([1, 2], [3, 4]).__mul__, {"one": 1, "two": 2, "three": 3}
        )

    def test_det(self) -> None:
        self.assertEqual(Matrix([3, 7], [1, -4]).det(), -19)

        self.assertEqual(Matrix([0, 1, 2], [3, 2, 1], [1, 1, 0]).det(), 3)

        # Not nxn (square) matrix
        self.assertRaises(ValueError, Matrix([1, 2, 3], [4, 5, 6]).det)

    def test_inverse(self) -> None:
        pass

    def test_isinvertable(self) -> None:
        self.assertTrue(Matrix([1, 2], [3, 4])._isinvertable())
        self.assertFalse(Matrix([1, 2, 3], [4, 5, 6])._isinvertable())
        self.assertFalse(Matrix([1, 2], [2, 4])._isinvertable())

    def test_transpose(self) -> None:
        self.assertEqual(Matrix([1, 2], [3, 4]).transpose(), Matrix([1, 3], [2, 4]))
        self.assertEqual(
            Matrix([1, 2, 3], [4, 5, 6]).transpose(), Matrix([1, 4], [2, 5], [3, 6])
        )


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
