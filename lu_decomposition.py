"""
Provide LU decomposition for Tridiagonal-Matrices nxn.
A - Tridiagonal Matrix, for all |i-j|>1 a_i_j = 0
L - Lower bidiagonal Matrix
U - Upper diagonal Matrix
"""
import numpy as np


class LU_decomp:
    def __init__(self):
        pass

    def get_tridiag_decomp(self, A):
        L, U = self.__tridiag_decompose(A.copy())
        return L, U
    
    def __is_tridiag(self,A):
        n = A.shape[0]
        for i in range (n-2):
            for j in range(i+2, n):
                if A[i][j] != 0 or A[j][i] != 0:
                    return False
        return True

    def __tridiag_decompose(self,A):
        self.A = A
        # Check if matrix is square
        try:
            self.m, self.n = self.A.shape
        except ValueError:
            raise ValueError(f"Matrix A with shape {self.A.shape} must have exactly two dimensions.")
        if self.m != self.n:
            raise ValueError(f"Decomposition works only with square Matrices. A is {self.A.shape} Matrix")
        elif not(self.__is_tridiag(self.A)):
            raise ValueError(f"Given Matrix is not tridiagonal!")
            
        ### start decomposition
        L = np.identity(self.n)
        U = np.identity(self.n)
        a = self.A[0][0]
        U[0][0] = a

        for i in range(self.n-1):
            U[i][i+1] = self.A[i][i+1]
            L[i+1][i] = self.A[i+1][i]/U[i][i]
            U[i+1][i+1] = self.A[i+1][i+1] - (L[i+1][i]*U[i][i+1])

        return L, U


if __name__ == "__main__":
    # test Matrix 01:
    A = np.array([ \
            [-2, -3, 0], \
            [-2, -1, -1], \
            [0, 1, -1]] \
            , dtype=float)

    """ 
    CHECK -> inf
    #test Matrix 02
    B = np.array([ \
            [1, -2, 0, 0], \
            [-2, 4, 1, 0], \
            [0, -4, 2, -1], \
            [0, 0, 2, 3] ]\
            ,dtype=float)  
    """  
    my_decomp = LU_decomp()                    #create class instance
    L, U = my_decomp.get_tridiag_decomp(A)
    ### Check if A=LU
    a_test = np.allclose(A, L@U)
    print(f"L*U = A? -> {a_test}")
    # b_test = np.allclose(B, Q@R)
    # print(f"Q*R = A? -> {b_test}")