#include "lab1.h"
#include "utils.h"

void transposeOnSingleCpu(int rows, int cols, int** matrix, int** result);
void transposeOnMultipleCpus(int rows, int cols, int** matrix, int** result);

void matrixTransposition() {
    cout << "Initialization" << endl;
    auto rows = 10000;
    auto cols = 5000;
    cout << "dimensions: " << rows << "x" << cols << endl << endl;
    auto matrix = generateIntMatrix(rows, cols);
    auto result = createIntMatrix(cols, rows);
    auto timer = Timer();

    cout << "Single CPU starts" << endl;
    timer.Start();
    transposeOnSingleCpu(rows, cols, matrix, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "CPUs in parallel starts" << endl;
    timer.Start();
    transposeOnMultipleCpus(rows, cols, matrix, result);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "GPU starts" << endl;
    auto matrixAsArray = matrixToArray(rows, cols, matrix);
    auto resultAsArray = new int[cols * rows];
    timer.Start();
    transposeOnGpu(rows, cols, matrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void transposeOnSingleCpu(int rows, int cols, int** matrix, int** result) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j][i] = matrix[i][j];
}

void transposeOnMultipleCpus(int rows, int cols, int** matrix, int** result) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j][i] = matrix[i][j];
}

void transposeOnGpu(int rows, int cols, int* matrix, int* result) {
    array_view<const int, 2> m(rows, cols, matrix);
    array_view<int, 2> res(rows, cols, result);
    res.discard_data();
    parallel_for_each(res.extent, [=](index<2> idx) restrict(amp) { res[idx] = m(idx[1], idx[0]); });
    res.synchronize();
}
