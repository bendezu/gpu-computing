#include "lab1.h"
#include "utils.h"

void multiplyOnSingleCpu(int rows, int cols, int** matrix, int num, int** result);
void multiplyOnMultipleCpus(int rows, int cols, int** matrix, int num, int** result);
void multiplyOnGpu(int rows, int cols, int* matrix, int num, int* result);

void matrixByNumMult() {
    cout << "Initialization" << endl;
    auto rows = 10000;
    auto cols = 5000;
    cout << "dimensions: " << rows << "x" << cols << endl << endl;
    auto num = 2;
    auto matrix = generateIntMatrix(rows, cols);
    auto result = createIntMatrix(rows, cols);
    auto timer = Timer();

    cout << "Single CPU starts" << endl;
    timer.Start();
    multiplyOnSingleCpu(rows, cols, matrix, num, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "CPUs in parallel starts" << endl;
    timer.Start();
    multiplyOnMultipleCpus(rows, cols, matrix, num, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "GPU starts" << endl;
    auto matrixAsArray = matrixToArray(rows, cols, matrix);
    auto resultAsArray = new int[rows * cols];
    timer.Start();
    multiplyOnGpu(rows, cols, matrixAsArray, num, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void multiplyOnSingleCpu(int rows, int cols, int** matrix, int num, int** result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = num * matrix[i][j];
        }
    }
}

void multiplyOnMultipleCpus(int rows, int cols, int** matrix, int num, int** result) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = num * matrix[i][j];
        }
    }
}

void multiplyOnGpu(int rows, int cols, int* matrix, int num, int* result) {
    array_view<const int, 2> array2d(rows, cols, matrix);
    array_view<int, 2> product(rows, cols, result);
    product.discard_data();
    parallel_for_each(product.extent, [=](index<2> idx) restrict(amp) { product[idx] = num * array2d[idx]; });
    product.synchronize();
}
