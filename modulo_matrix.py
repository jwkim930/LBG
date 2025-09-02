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
    def matrix(cls, content: list[list[int]] | np.typing.NDArray[np.int_], n: int) -> "ModMatrix":
        """
        Create a modulo matrix object from a 2D list or NumPy array.

        :param content: The array of integers containing the initial values.
                        Cannot be empty and must be rectangular.
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
    def size(self):
        """
        The dimension of the matrix, (row, col).
        """
        return self._size

    @property
    def n(self):
        """
        The modulus of the modulo matrix.
        """
        return self._n

    def __str__(self):
        return str(self._array)

    def __repr__(self):
        return f"matrix({str(self._array)}, {self._n})"

    def __mul__(self, other):
        if isinstance(other, int):
            return self.matrix(self._array * other, self._n)
        if isinstance(other, self.__class__):
            if self._n != other._n:
                raise ValueError("binary operation requires the same modulus")
            # implement later if needed
            return NotImplemented
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self._n != other._n:
                raise ValueError("binary operation requires the same modulus")
            return self.matrix(self._array + other._array, self._n)
        else:
            return NotImplemented

    def __neg__(self):
        return self.matrix(-self._array, self._n)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self + (-other)
        else:
            return NotImplemented
