#include "lab1.h"
#include "utils.h"

void multiplyOnSingleCpu(int rows1, int internalDim, int cols2, int** first, int** second, int** result);
void multiplyOnMultipleCpus(int rows1, int internalDim, int cols2, int** first, int** second, int** result);
void multiplyOnGpu(int rows1, int internalDim, int cols2, int* first, int* second, int* result);

void matrixByMatrixMult() {
    cout << "Initialization" << endl;
    auto rows1 = 1000;
    auto internalDim = 3000;
    auto cols2 = 500;
    cout << "dimentions: (";
    cout << rows1 << "x" << internalDim << ") by (";
    cout << internalDim << "x" << cols2 << ")" << endl << endl;
    auto firstMatrix = generateIntMatrix(rows1, internalDim);
    auto secondMatrix = generateIntMatrix(internalDim, cols2);
    auto result = createIntMatrix(rows1, cols2);
    auto timer = Timer();

    cout << "Single CPU starts" << endl;
    timer.Start();
    multiplyOnSingleCpu(rows1, internalDim, cols2, firstMatrix, secondMatrix, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "CPUs in parallel starts" << endl;
    timer.Start();
    multiplyOnMultipleCpus(rows1, internalDim, cols2, firstMatrix, secondMatrix, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "GPU starts" << endl;
    auto firstMatrixAsArray = matrixToArray(rows1, internalDim, firstMatrix);
    auto secondtMatrixAsArray = matrixToArray(internalDim, cols2, secondMatrix);
    auto resultAsArray = new int[rows1 * cols2];
    timer.Start();
    multiplyOnGpu(rows1, internalDim, cols2, firstMatrixAsArray, secondtMatrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void multiplyOnSingleCpu(int rows1, int internalDim, int cols2, int** first, int** second, int** result) {
    for (int i = 0; i < rows1; i++)
        for (int j = 0; j < cols2; j++)
            for (int k = 0; k < internalDim; k++)
                result[i][j] = first[i][k] * second[k][j];
}

void multiplyOnMultipleCpus(int rows1, int internalDim, int cols2, int** first, int** second, int** result) {
    #pragma omp parallel for
    for (int i = 0; i < rows1; i++)
        for (int j = 0; j < cols2; j++)
            for (int k = 0; k < internalDim; k++)
                result[i][j] = first[i][k] * second[k][j];
}

void multiplyOnGpu(int rows1, int internalDim, int cols2, int* first, int* second, int* result) {
    array_view<const int, 2> m1(rows1, internalDim, first);
    array_view<const int, 2> m2(internalDim, cols2, second);
    array_view<int, 2> res(rows1, cols2, result);
    res.discard_data();
    parallel_for_each(res.extent, [=](index<2> idx) restrict(amp) { 
        for (int i = 0; i < internalDim; i++)
            res[idx] = m1(idx[0], i) * m2(i, idx[1]);
    });
    res.synchronize();
}
