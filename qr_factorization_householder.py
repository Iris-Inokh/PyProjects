import numpy as np

class QR:
    def __init__(self):
        pass
    
    def get_decomp(self, A):
        return self.__householder(A.copy())

    def __householder(self, R):
        """
        Decompose Matrix A in to Q and R Matrices using Housholder transformations.
        Q - Orthogonal Matrix
        R - Upper Triangilar Matrix
        Method takes m x n Matrix with m>=n as input and returns Q and R Matrices 
        """
        self.R = R
        try:
            self.m, self.n = self.R.shape
        except ValueError:
            raise ValueError("Matrix A with shape {0!s} must have exactly two dimensions.".format(self.R.shape))
        if self.n > self.m:
            raise ValueError("QR factorization only works with arrays of shape (m,n) if m >= n.".format(self.R.shape))

        #n - number of vectors
        #m - number of dimentions
        self.Q_final = np.identity(self.m)
        for i in range(self.n):
            a_norm = np.linalg.norm(np.transpose(self.R)[i][i:])    #norm of a given vector
            e1 = np.zeros(self.m-i) 
            e1[0] = 1.                                              #construct basis vector e1 
            vec_1 = np.transpose(self.R)[i][i:]                     #a1 vector in matrix A    
            vec_1.shape = (np.size(vec_1), 1)                       #reshape to (, 1)
            vec_2 = np.sign(vec_1[0][0])*a_norm * e1                #vector with shape (m-i, )
            vec_2.shape = (np.size(vec_2), 1)                       #reshape to (, 1)
            v1 = np.add(vec_1, vec_2)     

            prod_vec = np.outer(np.transpose(v1),v1)                #v1 * v1t -> matrix
            prod_vec2 = np.transpose(v1).dot(v1)                    #1d array
            k = 2/prod_vec2
            Q1 = np.subtract(np.identity(self.m-i), k*prod_vec)     #reflexion
            Q_full = np.identity(self.m)
            Q_full[i:Q1.shape[0]+i, i:Q1.shape[1]+i] = Q1           #expand Q matrix to m dimentions with identity matrix
            self.Q_final = self.Q_final.dot(Q_full)
            self.R = Q_full.dot(self.R)
    
        return self.Q_final, self.R


if __name__ == "__main__":
    # test Matrix 01:
    A = np.array([ \
            [-2, -2, -2], \
            [-2, -1, -1], \
            [1, 0, -1]] \
            , dtype=float)

    #test Matrix 02
    B = np.array([ \
            [1, -2, -1], \
            [2, 0, 1], \
            [2, -4, 2], \
            [4, 0, 0] ]\
            ,dtype=float)  
    
    my_qr = QR()                    #create class instance
    Q, R = my_qr.get_decomp(A)      #compute decomposition
    #Q, R = my_qr.get_decomp(B)
    
    ### Check Q and R matricies
    q_test = np.allclose(Q.T@Q, np.identity(Q.shape[0]))
    print(f"Q orthogonal? -> {q_test}")
    r_test = np.allclose(R, np.triu(R))             #returns upper triangle of array
    print(f"R upper triangilar? -> {r_test}")
    a_test = np.allclose(A, Q@R)
    print(f"Q*R = A? -> {a_test}")
    # b_test = np.allclose(B, Q@R)
    # print(f"Q*R = A? -> {b_test}")

