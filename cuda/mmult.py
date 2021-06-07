import os
import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def mmult(m1, m2, m3):
    if m1.dtype != np.float32 or m2.dtype != np.float32 or m3.dtype != np.float32:
        raise TypeError("mmult expects ndarrays with a dtype of float32.")
    if m1.shape[0] != m3.shape[0] or m1.shape[1] != m2.shape[0] or m2.shape[1] != m3.shape[1]:
        raise ValueError("ndarray dims don't allow for a matrix multiplication\nm1 shape:" + str(m1.shape) + "\nm2 shape:" + str(m2.shape) + "\nm3 shape:" + str(m3.shape))
    if not m1.data.contiguous or not m2.data.contiguous or not m3.data.contiguous:
        raise TypeError("mmult expects contiguous ndarrays")
    d_m1 = cuda.mem_alloc(m1.nbytes)
    d_m2 = cuda.mem_alloc(m2.nbytes)
    d_m3 = cuda.mem_alloc(m3.nbytes)
    cuda.memcpy_htod(d_m1, m1)
    cuda.memcpy_htod(d_m2, m2)
    block = (m3.shape[1], m3.shape[0], 1)
    threads = (1, 1, 1)
    mmult_func.prepared_call(block, threads, d_m1, d_m2, d_m3, m1.shape[0], m1.shape[1], m2.shape[1])
    cuda.memcpy_dtoh(m3, d_m3)


def compare_matrices(m1, m2):
    if m1.shape[0] != m2.shape[0]:
        return False
    if m1.shape[1] != m2.shape[1]:
        return False
    matrices_same = True
    for r in range(0, m1.shape[0]):
        for c in range(0, m1.shape[1]):
            if m1[r, c] != m2[r, c]:
                matrices_same = False
                break
    return matrices_same


# In order for cuda to load correctly, nvcc needs to be in a directory in the PATH variable. This guarantees the
# directory of nvcc will be in PATH
os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda-11.3/bin"
mmult_kernel = SourceModule("""
    __global__ void mmult(float *m1, float *m2, float *m3, int m1_height, int m1_width, int m2_width){
        int r = blockIdx.y;
        int c = blockIdx.x;
        int sum = 0;
        for(int i = 0; i < m1_width; i++){
            float val1 = *(m1 + m1_width * r + i);
            float val2 = *(m2 + m2_width * i + c);
            sum += val1 * val2;
        }
        *(m3 + m2_width * r + c) = sum;
    }
""")
mmult_func = mmult_kernel.get_function("mmult")
mmult_func.prepare(["P", "P", "P", "i", "i", "i"])


if __name__ == "__main__":
    matrix1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    matrix2 = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
    matrix3 = np.zeros((2, 2), dtype=np.float32)
    correct_matrix3 = np.array([[6, 6], [15, 15]], dtype=np.float32)
    try:
        mmult(matrix1, matrix2, matrix3)
    except TypeError:
        print("Test 1 passed")

    matrix1 = np.array(matrix1, dtype=np.float32)
    matrix3 = np.zeros((2, 3), dtype=np.float32)
    try:
        mmult(matrix1, matrix2, matrix3)
    except ValueError:
        print("Test 2 passed")

    matrix3 = np.zeros((2, 2), dtype=np.float32, order='C')
    mmult(matrix1, matrix2, matrix3)

    if compare_matrices(matrix3, correct_matrix3):
        print("Test 3 passed")
    else:
        print("Test 3 failed\nExpected:\n" + str(correct_matrix3) + "\nActual:\n" + str(matrix3))

    matrix1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    matrix2 = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32)
    correct_matrix3 = np.array([[6, 12], [15, 30]], dtype=np.float32)
    mmult(matrix1, matrix2, matrix3)
    if compare_matrices(matrix3, correct_matrix3):
        print("Test 4 passed")
    else:
        print("Test 4 failed\nExpected:\n" + str(correct_matrix3) + "\nActual:\n" + str(matrix3))

