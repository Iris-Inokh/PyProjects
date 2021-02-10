"""
Provide LU decomposition for Tridiagonal-Matrices.
A - Tridiagonal Matrix, for all |i-j|>1 a_i_j = 0
L - Lower bidiagonal Matrix
U - Upper diagonal Matrix
"""
import numpy as np


class LU:
    def __init__(self):
        pass

    def get_tridiag_decomp(A):
        self.M = A.copy()
        L, U = __tridiag_decompose(self.M)

    def __tridiag_decompose(A):
        return L, U