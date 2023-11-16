"""Basic implementations of Linear Algebra concepts in Python.

This library is always a work in progress and is by no means guaranteed to be complete.
"""

from __future__ import annotations

__author__ = "Yannick Brenning"

import math
from typing import Iterator

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
        """Calculates the [magnitude](https://en.wikipedia.org/wiki/Magnitude_(mathematics)#Euclidean_vector_space)
        of the current vector instance

        Returns:
            Magnitude ||v|| of vector v
        """
        mag = 0
        for i in range(0, self.__len__()):
            mag += self.elems[i] ** 2

        assert isinstance(mag, int) or isinstance(mag, float)
        return math.sqrt(mag)

    def norm(self) -> Vector:
        """Norm of the vector v where ||v|| = 1

        See also: https://en.wikipedia.org/wiki/Norm_(mathematics)


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
    """Orthogonalize a given set of vectors using the \
    [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process).

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
    """Orthonormalize a given set of vectors using the \
    [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process).

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
        """Calculates the [determinant](https://en.wikipedia.org/wiki/Determinant) of a square matrix.

        Returns:
            determinant as a numeric scalar value
        """

        if len(self.elems) != len(self.elems[0]):
            raise ValueError(
                "Determinant can only be calculated for a square Matrix of size nxn"
            )

        return Matrix.__det_rec(self)

    @staticmethod
    def __det2d(matrix: Matrix) -> numeric:
        return (
            matrix.elems[0][0] * matrix.elems[1][1]
            - matrix.elems[0][1] * matrix.elems[1][0]
        )

    @staticmethod
    def __det_rec(matrix: Matrix) -> numeric:
        """Recursive implementation of the \
        [Laplace Expansion](https://en.wikipedia.org/wiki/Laplace_expansion).
        
        We use the simple determinant formula for 2-dimensional matrices as the base case and use cofactor
        expansion to calculate the determinant for higher dimensional matrices.

        Note: 
            This algorithm quickly becomes inefficent for large matrices.

        Args:
            matrix: Matrix or submatrix of which to calculate the determinant

        Returns:
            numeric: Determinant of given matrix
        """

        if len(matrix.elems) == 1:
            return matrix.elems[0][0]
        elif len(matrix.elems) == 2:
            return Matrix.__det2d(matrix)
        else:
            return sum(
                (-1) ** col * x * Matrix.__det_rec(Matrix.minor(matrix, 0, col))
                for col, x in enumerate(matrix.elems[0])
            )

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        return Matrix(*map(list, zip(*matrix.elems)))

    @staticmethod
    def minor(matrix: Matrix, row: int, col: int) -> Matrix:
        """Creates the first minor submatrix of an nxn matrix using a given row and column to remove.
        See also: [Minor (linear algebra)](https://en.wikipedia.org/wiki/Minor_(linear_algebra)#First_minors)

        Args:
            matrix: Matrix to be split
            row: Row to be removed
            col: Column to be removed

        Returns:
            Submatrix of size (n-1)x(n-1) formed by removing the row and column
        """

        if len(matrix.elems) != len(matrix.elems[0]):
            raise ValueError(
                "Minor can only be calculated for a square Matrix of size nxn"
            )

        args = [
            row[:col] + row[col + 1 :]
            for row in (matrix.elems[:row] + matrix.elems[row + 1 :])
        ]

        return Matrix(*args)

    @staticmethod
    def cofactor(matrix: Matrix) -> Matrix:
        """Calculates the cofactor matrix of a given matrix using the minors.
        See also: [Cofactor Matrix](https://en.wikipedia.org/wiki/Minor_(linear_algebra)#Inverse_of_a_matrix)

        Args:
            matrix: Matrix of which to determine the cofactors

        Returns:
            Matrix of cofactors
        """

        minor = [
            [
                Matrix.__det_rec(Matrix.minor(matrix, row, col))
                for col in range(0, len(matrix.elems))
            ]
            for row in range(0, len(matrix.elems))
        ]

        cofactor = [
            [(-1) ** (row + col) * x for col, x in enumerate(minor[row])]
            for row in range(0, len(minor))
        ]
        return Matrix(*cofactor)

    def adjungate(self) -> Matrix:
        """Calculates the adjungate matrix of the current instance which corresponds to the
        transpose of its cofactor matrix.

        See also: [Adjungate Matrix](https://en.wikipedia.org/wiki/Adjugate_matrix)

        Returns:
            Adjungate of the matrix
        """

        if self.elems == 1 == self.elems[0]:
            return Matrix([1])
        else:
            return Matrix.transpose(Matrix.cofactor(self))

    def inverse(self) -> Matrix:
        """Calculates the inverse matrix B of the current nxn matrix A such that
        AB = BA = I, where I is the nxn [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).

        Returns:
            Matrix to be reversed
        """

        if self._isinvertable():
            return self.adjungate() * (1 / self.det())
        else:
            raise ValueError("Matrix is not invertable")

    def _isinvertable(self) -> bool:
        """Determines whether or not the current matrix instance is invertable.

        Returns:
            True if reversible, False otherwise
        """

        return len(self.elems) == len(self.elems[0]) and self.det() != 0


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
        self.assertEqual(Vector(1, 0, 0).magnitude(), 1)
        self.assertEqual(Vector(0, 1, 0).magnitude(), 1)
        self.assertEqual(Vector(0, 0, 1).magnitude(), 1)

        # Magnitude of opposite vector should be identical
        self.assertEqual(Vector(-1, 0, 0).magnitude(), Vector(1, 0, 0).magnitude())

        self.assertEqual(Vector(0, 0, 2).magnitude(), 2)
        self.assertEqual(Vector(1, 2, 2).magnitude(), 3)
        self.assertEqual(Vector(4, 0, 0).magnitude(), 4)
        self.assertEqual(Vector(4, 3, 0).magnitude(), 5)

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

    def test_transpose(self) -> None:
        self.assertEqual(
            Matrix.transpose(Matrix([1, 2], [3, 4])), Matrix([1, 3], [2, 4])
        )
        self.assertEqual(
            Matrix.transpose(Matrix([1, 2, 3], [4, 5, 6])),
            Matrix([1, 4], [2, 5], [3, 6]),
        )

        self.assertEqual(
            Matrix.transpose(Matrix([0.5, 1.5, 2.5], [3.5, 4.5, 5.5])),
            Matrix([0.5, 3.5], [1.5, 4.5], [2.5, 5.5]),
        )

    def test_minor(self) -> None:
        self.assertEqual(
            Matrix.minor(Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9]), 0, 0),
            Matrix([5, 6], [8, 9]),
        )
        self.assertEqual(
            Matrix.minor(Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9]), 0, 1),
            Matrix([4, 6], [7, 9]),
        )
        self.assertEqual(
            Matrix.minor(Matrix([1, 2, 3], [4, 5, 6], [7, 8, 9]), 1, 1),
            Matrix([1, 3], [7, 9]),
        )
        self.assertEqual(
            Matrix.minor(
                Matrix([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]),
                0,
                0,
            ),
            Matrix([6, 7, 8], [10, 11, 12], [14, 15, 16]),
        )

        self.assertRaises(
            ValueError, Matrix.minor, Matrix([1, 2], [3, 4], [5, 6]), 0, 0
        )
        self.assertRaises(ValueError, Matrix.minor, Matrix([1, 2, 3], [4, 5, 6]), 0, 0)

    def test_cofactor(self) -> None:
        self.assertEqual(
            Matrix.cofactor((Matrix([-3, 2, -5], [-1, 0, -2], [3, -4, 1]))),
            Matrix([-8, -5, 4], [18, 12, -6], [-4, -1, 2]),
        )

    def test_adjungate(self) -> None:
        self.assertEqual(
            Matrix([-3, 2, -5], [-1, 0, -2], [3, -4, 1]).adjungate(),
            Matrix([-8, 18, -4], [-5, 12, -1], [4, -6, 2]),
        )

    def test_inverse(self) -> None:
        self.assertTrue(Matrix([2, 1], [6, 4]).inverse(), Matrix([2, -0.5], [-3, 1]))
        self.assertTrue(Matrix([-1, 1.5], [1, -1]).inverse(), Matrix([2, 3], [2, 2]))

    def test_isinvertable(self) -> None:
        self.assertTrue(Matrix([1, 2], [3, 4])._isinvertable())
        self.assertFalse(Matrix([1, 2, 3], [4, 5, 6])._isinvertable())
        self.assertFalse(Matrix([1, 2], [2, 4])._isinvertable())


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
