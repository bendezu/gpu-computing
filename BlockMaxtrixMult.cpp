#include "lab2.h"
#include "utils.h"
#include "lab1.h"

static const int tileSize = 32;
static const int rows1 = tileSize * 30;
static const int internalDim = tileSize * 100;
static const int cols2 = tileSize * 20;
void tiledMultiplication(int* first, int* second, int* result);
void sharedMultiplication(int* first, int* second, int* result);

void blockMatrixMult() {
    cout << "Initialization" << endl;
    cout << "dimensions: (";
    cout << rows1 << "x" << internalDim << ") by (";
    cout << internalDim << "x" << cols2 << ")" << endl << endl;
    auto timer = Timer();
    auto firstMatrixAsArray = generateIntArray(rows1 * internalDim);
    auto secondMatrixAsArray = generateIntArray(internalDim * cols2);
    auto resultAsArray = new int[rows1 * cols2];

    cout << "Regular multiplication starts" << endl;
    timer.Start();
    multiplyOnGpu(rows1, internalDim, cols2, firstMatrixAsArray, secondMatrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Tiled multiplication starts" << endl;
    timer.Start();
    tiledMultiplication(firstMatrixAsArray, secondMatrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Shared multiplication starts" << endl;
    timer.Start();
    sharedMultiplication(firstMatrixAsArray, secondMatrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void tiledMultiplication(int* first, int* second, int* result) {
    array_view<const int, 2> m1(rows1, internalDim, first);
    array_view<const int, 2> m2(internalDim, cols2, second);
    array_view<int, 2> res(rows1, cols2, result);
    res.discard_data();
    parallel_for_each(
        res.extent.tile<tileSize, tileSize>(),
        [=](tiled_index<tileSize, tileSize> tidx) restrict(amp) {
            int sum = 0;
            for (int i = 0; i < internalDim; i++)
                sum += m1(tidx.global[0], i) * m2(i, tidx.global[1]);
            res[tidx.global] = sum;
        });
    res.synchronize();
}

void sharedMultiplication(int* first, int* second, int* result) {
    array_view<const int, 2> m1(rows1, internalDim, first);
    array_view<const int, 2> m2(internalDim, cols2, second);
    array_view<int, 2> res(rows1, cols2, result);
    res.discard_data();
    parallel_for_each(
        res.extent.tile<tileSize, tileSize>(),
        [=](tiled_index<tileSize, tileSize> tidx) restrict(amp) {
            int row = tidx.local[0];
            int col = tidx.local[1];
            tile_static int local1[tileSize][tileSize];
            tile_static int local2[tileSize][tileSize];
            int sum = 0;
            for (int i = 0; i < internalDim; i += tileSize) {
                local1[row][col] = m1(tidx.global[0], col + i);
                local2[row][col] = m2(row + i, tidx.global[1]);
                tidx.barrier.wait();

                for (int k = 0; k < tileSize; k++)
                    sum += local1[row][k] * local2[k][col];
                tidx.barrier.wait();

            }
            res[tidx.global] = sum;
        });
    res.synchronize();
}
