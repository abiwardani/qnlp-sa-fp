import numpy as np
from sympy.simplify.fu import fu, L, TR9, TR10i, TR11
from sympy import factor, sin, cos, powsimp, exp
from sympy import re, im, I, E, Abs, S, conjugate
from sympy import symbols, Function, lambdify, simplify, preorder_traversal, Float, latex, pprint
from IPython.display import display

class QuantumState:
    def __init__(self, n_qubits, post_select=[]):
        self.n_qubits = n_qubits
        self.post_select = post_select
        self.states = [id(n_qubits)]
    
    def Compose(self, ops=[]):
        if len(ops) == 0:
            ops = [id(1) for _ in range(self.n_qubits)]
        
        self.states.append(transform_vect(mtp2(ops), self.states[-1]))

    def Measure(self, state=""):
        if state == "":
            state = "0"*n_qubits
        
        raw = braket(idx_to_vect(state), self.states[-1])#.expand(trig=True).expand()

        braket_ = simplify(raw, maxn=17, chop=True)
        temp = braket_
        for a in preorder_traversal(temp):
            if isinstance(a, Float):
                braket_ = braket_.subs(a, round(a, 15))

        p = simplify(re(braket_)*re(braket_)+im(braket_)*im(braket_))

        return p

def tensor_product(a, b):
    # definition of tensor product
    # A otimes B => a*B for a in A

    if ("__len__" not in dir(a)):
        return a*b
    elif ("__len__" not in dir(b)):
        return b*a
    else:
        res = [tensor_product(a[i//len(b)], b[i%len(b)]) for i in range(len(a)*len(b))]
        return np.array(res)

def transform_vect(M, v):
    # transform vector state v by operation matrix M
    # only applies transformation if dimensions are valid

    if (len(M) == len(v) and len(M[0]) == len(v)):
        return matmul(M, v)
    else:
        return v

def matmul(a, b):
    # matrix multiplication

    return np.matmul(a, b)

def diag_ones(dim):
    # make diagonal ones matrix (multiplicative identity matrix)

    return np.array([[int(j == i) for j in range(dim)] for i in range(dim)], dtype='object')

def id(n):
    # make identity matrix of n-qubit state

    return diag_ones(2**n)

def rz(theta):
    # Pauli-Z rotation matrix

    return np.array([[exp(-I*theta/2), 0], [0, exp(I*theta/2)]])

def rx(theta):
    # X-rotation matrix

    return np.array([[cos(theta/2), -I*sin(theta/2)], [-I*sin(theta/2), cos(theta/2)]])

def ry(theta):
    # Y-rotation matrix

    return np.array([[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]])

def x():
    # Pauli-X matrix

    return np.array([[0, 1], [1, 0]])

def y():
    # Pauli-Y matrix

    return np.array([[0, -1j], [1j, 0]])

def z():
    # Pauli-Z matrix

    return np.array([[1, 0], [0, -1]])

def cnot(direction=1):
    # CNOT matrix
    # 1 0 0 0
    # 0 1 0 0
    # 0 0 0 1
    # 0 0 1 0

    return cgate_0_1(x(), direction)

def cgate_0_1(M, direction=1):
    # controlled gate where the first m qubits
    # indicate the control register, and the last
    # m qubits indicate the target register

    base = diag_ones(2*len(M))

    if direction == 1:
        k = len(M)
    else:
        k = 0

    for i in range(len(M)):
        for j in range(len(M[0])):
            base[i+k][j+k] = M[i][j]

    return np.array(base)

def cgate(M, c, t, n=None):
    # control gate as defined by its effect
    # constructs matrix where if qubit c is |0>,
    # cgate performs I on qubit c and I on qubit t
    # but if qubit c is |1>,
    # cgate performs I on qubit c and M on qubit t

    if n is None:
        n = max(c, t)+1

    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])

    proj0x0 = np.outer(ket_0, dagger(ket_0))
    proj1x1 = np.outer(ket_1, dagger(ket_1))

    proj0s = [id(1) for _ in range(n)]
    proj0s[c] = proj0x0
    proj0s[t] = id(1)

    proj1s = [id(1) for _ in range(n)]
    proj1s[c] = proj1x1
    proj1s[t] = M

    return np.array(mtp2(proj0s)+mtp2(proj1s))

def dagger(M):
    # dagger operation

    return M.transpose()

def conjugate(M):
    # symbolic conjugation of matrix M

    res = [[i for i in row] for row in M]

    for i in range(len(M)):
        for j in range(len(M)):
            if type(res[i][j]) == type(1+1j):
                res[i][j] = np.conjugate(res[i][j])
            if type(res[i][j] == e):
                res[i][j] = e(-res[i][j].args())
            elif type(-res[i][j] == e):
                res[i][j] = -e((-res[i][j]).args())

    return res

def idx_to_vect(bit_str):
    # converts bit string of Z-basis states into multiqubit state
    # uses big Endian
    # e.g. "0010" into |0100>

    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])

    kets = [ket_0 if b == "0" else ket_1 for b in bit_str]

    return multitensor_product(kets)

def braket(sv1, sv2):
    # braket operator
    # braket(a, b) = <a|b>
    
    sv1_dagger = dagger(sv1)

    res = matmul(sv1_dagger, sv2)

    return res

def multitensor_product(mats):
    # multitensor product
    # A otimes B otimes C otimes D
    # = A otimes (B otimes (C otimes D))
    # uses big Endian

    res = None

    for mat in mats:
        if res is None:
            res = mat
        else:
            res = tensor_product(res, mat)

    return np.array(res)

def mtp2(mats):
    # multitensor product
    # D otimes C otimes B otimes A
    # = D otimes (C otimes (B otimes A))
    # uses little Endian

    res = None
    tempmat = None

    for mat in mats[::-1]:
        if res is None:
            if tempmat is None:
                tempmat = mat
            else:
                res = tensor_product(mat, tempmat)
        else:
            res = tensor_product(mat, res)

    return res

def idx_to_vect2(bit_str):
    # converts bit string in Z-basis to multiqubit state
    # uses little Endian
    # e.g. "0010" into |0010>

    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])

    kets = [ket_0 if b == "0" else ket_1 for b in bit_str]

    return mtp2(kets)

ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])
H = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])