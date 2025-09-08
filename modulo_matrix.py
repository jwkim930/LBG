from typing import Self
import numpy as np


class ModMatrix:
    """
    A matrix where entries are from the ring Z/nZ.
    Binary operations require the same modulus.
    """
    def __init__(self, size: tuple[int, int], n: int):
        """
        Initializes a modulo matrix object.
        The entries are all zero when initialized.

        :param size: The size of the matrix, (row, col).
        :param n: The modulus for this matrix. Must be at least 2.
        """
        if len(size) != 2 or size[0] < 1 or size[1] < 1:
            raise ValueError(f"matrix size must be two positive integers but was given {size}")
        if n < 2:
            raise ValueError(f"the modulus must be at least 2, was given {n}")
        self._size = size
        self._n = n
        self._array: np.typing.NDArray[np.int64] = np.zeros(size, dtype=np.int64)

    @classmethod
    def matrix(cls, content: list[list[int]] | np.typing.NDArray[np.int_], n: int) -> Self:
        """
        Create a modulo matrix object from a 2D list or NumPy array.

        :param content: The array of integers containing the initial values.
                        Cannot be empty and must be rectangular.
                        The values are copied, not reassigned.
        :param n: The modulus for this matrix. Must be at least 2.
        :return: The initialized ModMatrix object.
        """
        if isinstance(content, list):
            invalid = False
            nrow = -1
            for row in content:
                if nrow != -1 and (nrow == 0 or nrow != len(row)):
                    invalid = True
                    break
                nrow = len(row)
            if invalid:
                raise ValueError("invalid content size: either empty or non-rectangular")
            size = (len(content), nrow)
        elif isinstance(content, np.ndarray):
            if content.size == 0:
                raise ValueError("invalid content size: either empty or non-rectangular")
            size = content.shape

        result = cls(size, n)
        result._array += np.array(content) if isinstance(content, list) else content
        result._array %= n
        return result

    @property
    def size(self) -> tuple[int, int]:
        """
        The dimension of the matrix, (row, col).
        """
        return self._size

    @property
    def n(self) -> int:
        """
        The modulus of the modulo matrix.
        """
        return self._n

    def __str__(self) -> str:
        return str(self._array)

    def __repr__(self) -> str:
        return f"matrix({str(self._array)}, {self._n})"

    def __mul__(self, other) -> Self:
        if isinstance(other, int):
            # scalar multiplication
            return self.matrix(self._array * other, self._n)
        # matrix multiplication below
        # make sure the subclass constructor is used
        if issubclass(self.__class__, other.__class__):
            if self._n != other._n:
                raise ValueError("binary operation requires the same modulus")
            return self.matrix(self._array @ other._array, self._n)
        elif issubclass(other.__class__, self.__class__):
            if self._n != other._n:
                raise ValueError("binary operation requires the same modulus")
            return other.matrix(self._array @ other._array, self._n)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other) -> Self:
        if isinstance(other, self.__class__):
            if self._n != other._n:
                raise ValueError("binary operation requires the same modulus")
            return self.matrix(self._array + other._array, self._n)
        else:
            return NotImplemented

    def __neg__(self) -> Self:
        return self.matrix(-self._array, self._n)

    def __sub__(self, other) -> Self:
        if isinstance(other, self.__class__):
            return self + (-other)
        else:
            return NotImplemented

    def __getitem__(self, key: tuple[int, int]) -> int:
        if len(key) != 2:
            raise IndexError(f"{self.__class__.__name__} requires index with two integers")
        return int(self._array[key])

    def __setitem__(self, key: tuple[int, int], value: int):
        if len(key) != 2:
            raise IndexError(f"{self.__class__.__name__} requires index with two integers")
        self._array[key] = value % self._n

    def get_row(self, i: int):
        """
        Returns the row at the index.
        The returned row is still a 2D array as a row.

        :param i: The index of the row.
        :return: The row matrix at the index.
        """
        return self.matrix(self._array[[i], :], self._n)

    def get_column(self, i: int):
        """
        Returns the column at the index.
        The returned column is still a 2D array as a column.

        :param i: The index of the column.
        :return: The column matrix at the index.
        """
        return self.matrix(self._array[:, [i]], self._n)

    def hstack(self, other: "ModMatrix") -> Self:
        """
        Returns a matrix created by concatenating two matrices horizontally.
        :param other: The matrix to come to the right of this matrix.
                      Must have the same modulus and the same number of rows.
        :return: The concatenated matrix.
        """
        if self._n != other._n:
            raise ValueError("matrix moduli do not agree")
        if self._size[0] != other._size[0]:
            raise ValueError("the matrices have different number of rows")
        # make sure to use the subclass constructor
        if issubclass(self.__class__, other.__class__):
            return self.__class__.matrix(np.hstack((self._array, other._array)), self._n)
        else:
            return other.__class__.matrix(np.hstack((self._array, other._array)), self._n)

    def transpose(self) -> Self:
        """
        Returns a copy of the matrix that is transposed with the same modulus.
        :return: The transposed matrix.
        """
        return self.__class__.matrix(self._array.transpose(), self._n)

class PrimeModMatrix(ModMatrix):
    """
    ModMatrix but with a prime modulus.
    Supports division by an integer.
    """
    def __init__(self, size: tuple[int, int], n: int):
        """
        Initializes a modulo matrix object with a prime modulus.
        The entries are all zero when initialized.

        :param size: The size of the matrix, (row, col).
        :param n: The modulus for this matrix. Must be at least 2 and a prime number.
                  Must be less than 100 for implementation reasons.
        """
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        if n not in primes:
            raise ValueError("non-prime modulus or modulus over 100")
        super().__init__(size, n)

    def __truediv__(self, other) -> Self:
        if isinstance(other, int):
            # invert other by extended Euclidean algorithm
            a = other   # dividend
            b = self._n   # divisor
            x0, x1 = 1, 0   # last two coefficients for a
            y0, y1 = 0, 1   # last two coefficient for b
            while b > 0:
                q, a, b = a // b, b, a % b
                x0, x1 = x1, x0 - q * x1
                y0, y1 = y1, y0 - q * y1
            assert a == 1, f"Euclidean algorithm says other ({other}) is not coprime to n ({self.n}), which shouldn't happen"

            # multiply by the inverse
            return self * x0
        else:
            return NotImplemented

    def row_reduced(self) -> Self:
        """
        Returns a copy of the matrix reduced to reduced row echelon form.

        :return: The row-reduced matrix.
        """
        result = self.matrix(self._array, self._n)
        # Gaussian elimination
        def row_swap(i: int, j: int):
            temp = np.copy(result._array[i])
            result._array[i] = result._array[j]
            result._array[j] = temp
        def row_mult(i: int, x: int):
            result._array[i] *= x
            result._array[i] %= result._n
        def row_add(add_to_index: int, add_index: int, x: int = 1):
            result._array[add_to_index] += result._array[add_index] * x
            result._array[add_to_index] %= result._n

        m, n = result._size
        row = 0
        pivots = []   # track pivot positions
        for col in range(n):
            # find pivot row at or below 'row'
            pivot = None
            for r in range(row, m):
                if result[r, col] != 0:
                    pivot = r
                    break

            if pivot is None:
                continue  # no pivot in this column

            # bring pivot row up
            if pivot != row:
                row_swap(pivot, row)

            # normalize pivot row
            factor = (PrimeModMatrix.matrix([[1]], result._n) / result[row, col])[0, 0]
            row_mult(row, factor)

            # eliminate entries below pivot
            for r in range(row + 1, m):
                if result[r, col] != 0:
                    factor = (-(PrimeModMatrix.matrix([[result[r, col]]], result._n) / result[row, col]))[0, 0]
                    row_add(r, row, factor)
            pivots.append((row, col))
            row += 1
            if row == m:
                break

        # backward elimination
        for row, col in reversed(pivots):
            for r in range(row):
                if result[r, col] != 0:
                    factor = -result[r, col] % result._n
                    row_add(r, row, factor)
        return result
