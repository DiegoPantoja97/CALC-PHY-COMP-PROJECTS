#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np

"""Segment 1 - Matrix and Vector Basics """


# Block 1: Defining Vectors and Matrices
print("Block 1: Defining Vectors and Matrices")
v1 = np.array([1, 2, 3])  # Example row vector
v2 = np.array([4, 5, 6])  # Another row vector
v3 = np.array([[1], # Column vector
               [2], 
               [3]])  
print(f"Row Vector v1: {v1}")
print(f"Row Vector v2: {v2}")
print(f"Column Vector v3:\n{v3}")

A = np.array([[1 ,2 ,3], # Example matrix with complex values
              [4, 5, 6], 
              [7, 8, 9]])  
print(f"Matrix A:\n{A}")

A = np.array([[1 + 2j, 2 - 1j, 3], # Example matrix with complex values
              [4, 5 + 1j, 6], 
              [7, 8, 9]])  
print(f"Matrix A:\n{A}")







# Block 2: Slicing and Indexing
print("\n\n\nBlock 2: Slicing and Indexing")
print(f"A[0, 0]: {A[0, 0]}")  # Single element
print(f"A[1, :]: {A[1, :]}")  # Second row
print(f"A[:, 1]: {A[:, 1]}")  # Second column
print(f"A[0:2, 0:2]:\n {A[0:2, 0:2]}")  # Top-left 2x2 submatrix








# Block 3: Derived Matrix Representations
print("\n\n\nBlock 3: Derived Matrix Representations")
transpose_A = A.T
print(f"Transpose of A:\n{transpose_A}")

diagonal_A = np.diag(A)
print(f"Diagonal of A: {diagonal_A}")

triu_A = np.triu(A)
print(f"Upper Triangular Part of A:\n{triu_A}")

tril_A = np.tril(A)
print(f"Lower Triangular Part of A:\n{tril_A}")

conjugate_A = np.conj(A)
print(f"Conjugate of A:\n{conjugate_A}")

transpose_conjugate_A = A.conj().T  # Hermitian transpose
print(f"Transpose Conjugate (Hermitian) of A:\n{transpose_conjugate_A}")

if np.linalg.det(A) != 0:
    inverse_A = np.linalg.inv(A)
    print(f"Inverse of A:\n{inverse_A}")
else:
    print("Matrix A is singular, so it does not have an inverse.")







# Block 4: Vector Operations
print("\nBlock 4: Vector Operations")
dot_product = np.dot(v1, v2)
print(f"Dot Product (v1 . v2): {dot_product}")

cross_product = np.cross(v1, v2)
print(f"Cross Product (v1 x v2): {cross_product}")

outer_product = np.outer(v1, v2)
print(f"Outer Product:\n{outer_product}")

magnitude_v1 = np.linalg.norm(v1)
print(f"Magnitude of v1: {magnitude_v1}")

addition = v1 + v2
print(f"Addition (v1 + v2): {addition}")







# Block 5: Matrix Creation and Initialization
print("\n\n\nBlock 5: Matrix Creation and Initialization")

zeros_matrix = np.zeros((3, 3))
print(f"Zero Matrix:\n{zeros_matrix}")

ones_matrix = np.ones((3, 3))
print(f"Ones Matrix:\n{ones_matrix}")

full_matrix = np.full((3, 3), 7)
print(f"Full Matrix (filled with 7):\n{full_matrix}")

np.fill_diagonal(full_matrix, 5)
print(f"Full Matrix with Diagonal Replaced by 5:\n{full_matrix}")

""" Random Filling """

random_matrix = np.random.rand(3, 3)
print(f"Random Matrix:\n{random_matrix}")

random_uniform_custom = np.random.uniform(-10, 10, (3, 3))
print(f"Uniform Random Matrix [-10, 10):\n{random_uniform_custom}")

random_normal = np.random.randn(3, 3)
print(f"Standard Normal Random Matrix:\n{random_normal}")

random_integers = np.random.randint(1, 10, (3, 3))
print(f"Random Integer Matrix [1, 10):\n{random_integers}")

random_binomial = np.random.binomial(10, 0.5, (3, 3))  # n=10 trials, p=0.5 success probability
print(f"Binomial Random Matrix:\n{random_binomial}")

random_exponential = np.random.exponential(scale=2, size=(3, 3))
print(f"Exponential Random Matrix (scale=2):\n{random_exponential}")

random_poisson = np.random.poisson(lam=3, size=(3, 3))  # lam = expected number of events
print(f"Poisson Random Matrix (lambda=3):\n{random_poisson}")

mean = [0, 0]  # Mean vector
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
random_multivariate = np.random.multivariate_normal(mean, cov, size=5)
print(f"Multivariate Normal Random Samples:\n{random_multivariate}")

#Selects Random elements from an array
elements = np.array([1, 2, 3, 4, 5])
random_choice = np.random.choice(elements, size=(3, 3))
print(f"Random Choice Matrix:\n{random_choice}")



identity_matrix = np.eye(4)
print(f"Identity Matrix (4x4):\n{identity_matrix}")





# Block 6: Stacking and Concatenation
print("\n\n\nBlock 6: Stacking and Concatenation")
stacked_vector = np.hstack((v1, v2))  # Horizontal stacking
print(f"Horizontally Stacked Vectors: {stacked_vector}")

stacked_column_vector = np.hstack((v3, v3))  # Horizontal stacking of column vectors
print(f"Horizontally Stacked Column Vectors:\n{stacked_column_vector}")


block_matrix = np.block([[A, A], # Block matrix
                         [A, A]])  
print(f"Block Matrix:\n{block_matrix}")


# Define two 2D arrays
a = np.array([[1, 2], 
              [3, 4]])
b = np.array([[5, 6], 
              [7, 8]])

# Concatenate along rows (axis=0)
result_rows = np.concatenate((a, b), axis=0)
print("Concatenation Along Rows (axis=0):\n", result_rows)

# Concatenate along columns (axis=1)
result_columns = np.concatenate((a, b), axis=1)
print("Concatenation Along Columns (axis=1):\n", result_columns)



# Block 7: Matrix Operations


print("\n\n\nBlock 7: Matrix Operations")

# Define matrices A and B
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

B = np.array([[1, 0, 1], 
              [0, 1, 0], 
              [1, 5, -2]])

# Matrix Multiplication using @ operator
C = A @ B
print(f"Matrix Multiplication (A @ B):\n{C}")

# Check if matrices commute: A @ B != B @ A
D = B @ A
print(f"Matrix Multiplication (B @ A):\n{D}")
commutes = np.allclose(C, D)
print(f"Do A and B commute? {'Yes' if commutes else 'No'}")

# Commutator [A, B] = A @ B - B @ A
commutator = A @ B - B @ A
print(f"Commutator [A, B]:\n{commutator}")

# Anti-Commutator {A, B} = A @ B + B @ A
anti_commutator = A @ B + B @ A
print(f"Anti-Commutator {A, B}:\n{anti_commutator}")

# Determinant of A
determinant_A = np.linalg.det(A)
print(f"Determinant of A: {determinant_A:.2f}")

# Cofactor Matrix of A
if determinant_A != 0:
    cofactor_matrix = np.linalg.inv(A).T * determinant_A
    print(f"Cofactor Matrix of A:\n{cofactor_matrix}")
else:
    print("Cofactor Matrix: A is singular, so it doesn't have an inverse.")

# Have matrix act on a vector
v = np.array([1, 2, 3])  # Define a vector
result_A_v = A @ v
print(f"Result of A acting on vector v:\n{result_A_v}")

result_B_v = B @ v
print(f"Result of B acting on vector v:\n{result_B_v}")

""" NumPy allows 𝐴@𝑣 where A is 3×3 v is a 1D 1x3 vector because it implicitly treats the 1D array as a column vector """


# Block 8: Matrix Power Series

def matrix_power_series(A, func="exp", terms=10):
    """
    Computes the power series expansion of a matrix for common functions.
    
    :param A: The input matrix (NumPy array).
    :param func: The function to approximate ("exp", "sin", or "cos").
    :param terms: The number of terms in the power series.
    :return: The approximated matrix.
    """
    n = A.shape[0]  # Size of the matrix
    result = np.zeros_like(A, dtype=np.float64)  # Initialize result matrix
    current_term = np.eye(n)  # Start with the identity matrix
    
    for k in range(terms):
        if func == "exp":
            coeff = 1 / np.math.factorial(k)
        elif func == "sin":
            coeff = (-1)**((k - 1) // 2) / np.math.factorial(k) if k % 2 == 1 else 0
        elif func == "cos":
            coeff = (-1)**(k // 2) / np.math.factorial(k) if k % 2 == 0 else 0
        else:
            raise ValueError("Unsupported function. Choose 'exp', 'sin', or 'cos'.")
        
        # Add current term to the result
        result += coeff * current_term
        
        # Update current_term for next iteration
        current_term = current_term @ A
    
    return result

# Example Usage
if __name__ == "__main__":
    # Define a matrix
    A = np.array([[0, 1], 
                  [-1, 0]], dtype=np.float64)

    # Compute exp(A)
    exp_A = matrix_power_series(A, func="exp", terms=20)
    print("Matrix Exponential (exp(A)):")
    print(exp_A)

    """# Compute sin(A)
    sin_A = matrix_power_series(A, func="sin", terms=20)
    print("\nMatrix Sine (sin(A)):")
    print(sin_A)

    # Compute cos(A)
    cos_A = matrix_power_series(A, func="cos", terms=20)
    print("\nMatrix Cosine (cos(A)):")
    print(cos_A) """






# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



"""Segment 2 - Linear Transformations """





# Helper function to plot vectors in R2
def plot_vectors_r2(vectors, labels=None, colors=None, title="Vectors in R2"):
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    for i, vec in enumerate(vectors):
        color = colors[i] if colors else 'blue'
        label = labels[i] if labels else f"Vector {i+1}"
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color, label=label)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Helper function to plot vectors in R3
def plot_vectors_r3(vectors, labels=None, colors=None, title="Vectors in R3"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, vec in enumerate(vectors):
        color = colors[i] if colors else 'blue'
        label = labels[i] if labels else f"Vector {i+1}"
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, label=label)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Predefined linear transformations in R2
def rotation_matrix_2d(theta):
    """Rotation matrix for R2."""
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def scaling_matrix_2d(sx, sy):
    """Scaling matrix for R2."""
    return np.array([[sx, 0], 
                     [0, sy]])

def shear_matrix_2d(kx, ky):
    """Shear matrix for R2."""
    return np.array([[1, kx], 
                     [ky, 1]])

def reflection_matrix_2d(angle):
    """Reflection about a line making angle with X-axis in R2."""
    return np.array([[np.cos(2 * angle), np.sin(2 * angle)], 
                     [np.sin(2 * angle), -np.cos(2 * angle)]])

# Predefined linear transformations in R3
def rotation_matrix_3d(axis, theta):
    """Rotation matrix for R3 about a given axis."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], 
                         [0, c, -s], 
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], 
                         [0, 1, 0], 
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], 
                         [s, c, 0], 
                         [0, 0, 1]])

def scaling_matrix_3d(sx, sy, sz):
    """Scaling matrix for R3."""
    return np.array([[sx, 0, 0], 
                     [0, sy, 0], 
                     [0, 0, sz]])

def shear_matrix_3d(kxy, kxz, kyx, kyz, kzx, kzy):
    """Shear matrix for R3."""
    return np.array([[1, kxy, kxz], 
                     [kyx, 1, kyz], 
                     [kzx, kzy, 1]])

def reflection_matrix_3d(axis):
    """Reflection matrix for R3 about a coordinate plane."""
    if axis == 'xy':
        return np.array([[1, 0, 0], 
                         [0, 1, 0], 
                         [0, 0, -1]])
    elif axis == 'xz':
        return np.array([[1, 0, 0], 
                         [0, -1, 0], 
                         [0, 0, 1]])
    elif axis == 'yz':
        return np.array([[-1, 0, 0], 
                         [0, 1, 0], 
                         [0, 0, 1]])

def rotation_about_point_matrix(x0, y0, theta):
    """
    Construct a transformation matrix for rotating about a point (x0, y0).
    """
    # Translation to origin
    T_neg = np.array([[1, 0, -x0],
                      [0, 1, -y0],
                      [0, 0, 1]])
    
    # Rotation about origin
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    
    # Translation back to the point
    T_pos = np.array([[1, 0, x0],
                      [0, 1, y0],
                      [0, 0, 1]])
    
    # Combined transformation matrix
    M = T_pos @ R @ T_neg
    return M

# Projection Matrix onto a Line in R2
def projection_matrix_onto_line_r2(m):
    """
    Returns the projection matrix for projecting onto a line y = mx in R2.
    
    :param m: Slope of the line y = mx.
    :return: The projection matrix.
    """
    return (1 / (1 + m**2)) * np.array([[1, m],
                                        [m, m**2]])

# Projection Matrix onto a Plane in R3
def projection_matrix_onto_plane_r3(normal):
    """
    Returns the projection matrix for projecting onto a plane with a given normal vector in R3.
    
    :param normal: The normal vector of the plane as a NumPy array [nx, ny, nz].
    :return: The projection matrix.
    """
    normal = np.array(normal)
    normal_norm_sq = np.dot(normal, normal)  # ||n||^2
    return np.eye(3) - (np.outer(normal, normal) / normal_norm_sq)


def define_linear_transformation():
    """
    Defines a generic linear transformation matrix directly using NumPy array.
    """
    # Example: Replace with your own values
    matrix = np.array([
        [2, 0],  # First row of the matrix
        [0, 3]   # Second row of the matrix
    ], dtype=float)
    
    print("\nYour transformation matrix is:")
    print(matrix)
    return matrix

# Example usage
if __name__ == "__main__":
    # R2 Example
    vector_r2 = np.array([3, 4])
    rotation = rotation_matrix_2d(np.radians(90))  # 45-degree rotation
    transformed_r2 = rotation @ vector_r2

    print(f"Original Vector (R2): {vector_r2}")
    print(f"Transformed Vector (R2): {transformed_r2}")

    plot_vectors_r2([vector_r2, transformed_r2], labels=["Original", "Rotated"], colors=["blue", "red"])

    # R3 Example
    vector_r3 = np.array([1, 2, 3])
    scaling = scaling_matrix_3d(2, 0.5, 1)  # Scaling by different factors
    transformed_r3 = scaling @ vector_r3

    print(f"Original Vector (R3): {vector_r3}")
    print(f"Transformed Vector (R3): {transformed_r3}")

    plot_vectors_r3([vector_r3, transformed_r3], labels=["Original", "Scaled"], colors=["blue", "green"])


# In[29]:


"""Segment 3 - Matrix Classification Check """


def classify_matrix(matrix):
    """
    Classifies a matrix based on various properties.
    
    :param matrix: Input matrix (NumPy array).
    :return: List of all properties the matrix satisfies.
    """
    properties = []

    # Basic checks
    if np.isreal(matrix).all():
        properties.append("Real")
    if not np.isreal(matrix).all():
        properties.append("Imaginary")

    # Orthogonal: A @ A.T = I
    if np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0])):
        properties.append("Orthogonal")

    # Unitary: A† @ A = I
    if np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0])):
        properties.append("Unitary")

    # Symmetric: A.T = A
    if np.allclose(matrix, matrix.T):
        properties.append("Symmetric")

    # Hermitian: A† = A
    if np.allclose(matrix, matrix.conj().T):
        properties.append("Hermitian")

    # Anti-symmetric: A.T = -A
    if np.allclose(matrix.T, -matrix):
        properties.append("Anti-Symmetric")

    # Anti-Hermitian: A† = -A
    if np.allclose(matrix.conj().T, -matrix):
        properties.append("Anti-Hermitian")

    # Involutory: A @ A = I
    if np.allclose(matrix @ matrix, np.eye(matrix.shape[0])):
        properties.append("Involutory")

    # Diagonal: All non-diagonal elements are zero
    if np.allclose(matrix, np.diag(np.diag(matrix))):
        properties.append("Diagonal")

    # Upper triangular: All elements below diagonal are zero
    if np.allclose(matrix, np.triu(matrix)):
        properties.append("Upper Triangular")

    # Lower triangular: All elements above diagonal are zero
    if np.allclose(matrix, np.tril(matrix)):
        properties.append("Lower Triangular")

    # Nilpotent: A^k = 0 for some k
    try:
        k = 1
        power = matrix
        while k <= matrix.shape[0]:
            power = power @ matrix
            if np.allclose(power, 0):
                properties.append("Nilpotent")
                break
            k += 1
    except:
        pass
    # Singular: Determinant is zero
    if np.isclose(np.linalg.det(matrix), 0):
        properties.append("Singular")

    return properties

# Example Usage
if __name__ == "__main__":
    # Define a Matrix
    
    matrix = np.array([
        [1, 0, 1-1j],
        [0, 1, 0],
        [1+1j, 0, 1]
        ], dtype=complex)

    print("Matrix:")
    print(matrix)

    # Classify the matrix
    classifications = classify_matrix(matrix)
    print("\nClassifications:")
    for prop in classifications:
        print(f"- {prop}")


# In[31]:


"""Segment 4 - Eigenvalues, Eigenvectors, Diagnolization/Spectral Decomposition """


import numpy as np
import matplotlib.pyplot as plt

def plot_vectors_r2(vectors, colors, labels, title="Vectors in R2"):
    """
    Plots a set of vectors in R2.
    
    :param vectors: List of 2D vectors to plot.
    :param colors: List of colors for the vectors.
    :param labels: List of labels for the vectors.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    for i, vec in enumerate(vectors):
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Main Program
if __name__ == "__main__":
    # Define a sample matrix
    matrix = np.array([
        [2, 1],
        [1, 2]
    ], dtype=float)

    print("Matrix:")
    print(matrix)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors (P):")
    print(eigenvectors)

    # Compute P^-1
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    print("\nMatrix P (Eigenvectors):")
    print(P)
    print("\nMatrix P^-1 (Inverse of P):")
    print(P_inv)

    # Construct the diagonal matrix D
    D = np.diag(eigenvalues)
    print("\nDiagonal Matrix D (Eigenvalues):")
    print(D)

    # Verify A = P D P^-1
    reconstructed_A = P @ D @ P_inv
    print("\nReconstructed Matrix A (P D P^-1):")
    print(reconstructed_A)

    # Verify P^-1 A P = D
    transformed_D = P_inv @ matrix @ P
    print("\nTransformed Matrix (P^-1 A P):")
    print(transformed_D)

    # Transform eigenvectors using the matrix
    transformed_vectors = matrix @ eigenvectors

    # Plot original and transformed eigenvectors
    vectors_to_plot = []
    colors = []
    labels = []

    for i in range(eigenvectors.shape[1]):
        vectors_to_plot.append(eigenvectors[:, i])  # Original eigenvector
        colors.append("blue")
        labels.append(f"Eigenvector {i+1} (Original)")

        vectors_to_plot.append(transformed_vectors[:, i])  # Transformed vector
        colors.append("red")
        labels.append(f"Eigenvector {i+1} (Transformed)")

    plot_vectors_r2(vectors_to_plot, colors, labels, title="Eigenvectors Before and After Transformation")


# In[45]:


""" Segment 6 Matrix Decomposition """



import numpy as np
from scipy.linalg import lu, svd, qr, eig

# LU Decomposition
def lu_decomposition(matrix):
    P, L, U = lu(matrix)
    return P, L, U

# QR Decomposition
def qr_decomposition(matrix):
    Q, R = qr(matrix)
    return Q, R

# Singular Value Decomposition (SVD)
def svd_decomposition(matrix):
    U, Sigma, Vh = svd(matrix)
    return U, Sigma, Vh

# Eigenvalue Decomposition
def eigen_decomposition(matrix):
    eigenvalues, eigenvectors = eig(matrix)
    return eigenvalues, eigenvectors

# Main Program
if __name__ == "__main__":
    # Define a sample matrix
    matrix = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    print("Matrix A:")
    print(matrix)

    # LU Decomposition
    print("\nLU Decomposition:")
    P, L, U = lu_decomposition(matrix)
    print(f"P (Permutation Matrix):\n{P}")
    print(f"L (Lower Triangular Matrix):\n{L}")
    print(f"U (Upper Triangular Matrix):\n{U}")
    reconstructed_lu = P @ L @ U
    print(f"Reconstructed Matrix A (P L U):\n{reconstructed_lu}")

    # QR Decomposition
    print("\nQR Decomposition:")
    Q, R = qr_decomposition(matrix)
    print(f"Q (Orthogonal Matrix):\n{Q}")
    print(f"R (Upper Triangular Matrix):\n{R}")
    reconstructed_qr = Q @ R
    print(f"Reconstructed Matrix A (Q R):\n{reconstructed_qr}")

    # SVD Decomposition
    print("\nSingular Value Decomposition (SVD):")
    U, Sigma, Vh = svd_decomposition(matrix)
    print(f"U (Left Singular Vectors):\n{U}")
    print(f"Sigma (Singular Values):\n{Sigma}")
    print(f"Vh (Right Singular Vectors):\n{Vh}")
    Sigma_matrix = np.diag(Sigma)
    reconstructed_svd = U @ Sigma_matrix @ Vh
    print(f"Reconstructed Matrix A (U Sigma Vh):\n{reconstructed_svd}")

    # Eigenvalue Decomposition
    print("\nEigenvalue Decomposition:")
    eigenvalues, eigenvectors = eigen_decomposition(matrix)
    print(f"Eigenvalues:\n{eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    reconstructed_eigen = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    print(f"Reconstructed Matrix A (Eigenvectors Eigenvalues Eigenvectors^-1):\n{reconstructed_eigen}")


# In[43]:


""" Segment 6 Matrix Decomposition """


import numpy as np
from scipy.linalg import lu, svd, qr, eig

# LU Decomposition
def lu_decomposition(matrix):
    """
    Perform LU decomposition.
    :param matrix: The input matrix.
    :return: (P, L, U) where P is the permutation matrix,
             L is the lower triangular matrix, and U is the upper triangular matrix.
    """
    P, L, U = lu(matrix)
    return P, L, U

# QR Decomposition
def qr_decomposition(matrix):
    """
    Perform QR decomposition.
    :param matrix: The input matrix.
    :return: (Q, R) where Q is an orthogonal matrix and R is an upper triangular matrix.
    """
    Q, R = qr(matrix)
    return Q, R

# Singular Value Decomposition (SVD)
def svd_decomposition(matrix):
    """
    Perform Singular Value Decomposition.
    :param matrix: The input matrix.
    :return: (U, Sigma, Vh) where U and Vh are orthogonal matrices,
             and Sigma is the diagonal matrix of singular values.
    """
    U, Sigma, Vh = svd(matrix)
    return U, Sigma, Vh

# Eigenvalue Decomposition
def eigen_decomposition(matrix):
    """
    Perform Eigenvalue Decomposition.
    :param matrix: The input matrix.
    :return: (eigenvalues, eigenvectors).
    """
    eigenvalues, eigenvectors = eig(matrix)
    return eigenvalues, eigenvectors

# Example Usage
if __name__ == "__main__":
    # Define a sample matrix
    matrix = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)

    print("Matrix:")
    print(matrix)

    # LU Decomposition
    print("\nLU Decomposition:")
    P, L, U = lu_decomposition(matrix)
    print(f"P (Permutation Matrix):\n{P}")
    print(f"L (Lower Triangular Matrix):\n{L}")
    print(f"U (Upper Triangular Matrix):\n{U}")

    # QR Decomposition
    print("\nQR Decomposition:")
    Q, R = qr_decomposition(matrix)
    print(f"Q (Orthogonal Matrix):\n{Q}")
    print(f"R (Upper Triangular Matrix):\n{R}")

    # SVD Decomposition
    print("\nSingular Value Decomposition (SVD):")
    U, Sigma, Vh = svd_decomposition(matrix)
    print(f"U (Left Singular Vectors):\n{U}")
    print(f"Sigma (Singular Values):\n{Sigma}")
    print(f"Vh (Right Singular Vectors):\n{Vh}")

    # Eigenvalue Decomposition
    print("\nEigenvalue Decomposition:")
    eigenvalues, eigenvectors = eigen_decomposition(matrix)
    print(f"Eigenvalues:\n{eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")


# In[ ]:




