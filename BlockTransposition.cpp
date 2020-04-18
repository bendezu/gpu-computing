#include "lab2.h"
#include "utils.h"
#include "lab1.h"

static const int tileSize = 32;
static const int rows = tileSize * 300;
static const int cols = tileSize * 200;
void blockTransposeOnGpu(int* matrix, int* result);

void blockTransposition() {
    cout << "Initialization" << endl;
    cout << "dimensions: " << rows << "x" << cols << endl << endl;
    auto timer = Timer();
    auto matrixAsArray = generateIntArray(rows * cols);
    auto resultAsArray = new int[cols * rows];

    cout << "Regular transposition starts" << endl;
    timer.Start();
    transposeOnGpu(rows, cols, matrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Block transposition starts" << endl;
    timer.Start();
    blockTransposeOnGpu(matrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void blockTransposeOnGpu(int* matrix, int* result) {
    array_view<const int, 2> inData(rows, cols, matrix);
    array_view<int, 2> outData(rows, cols, result);
    outData.discard_data();
    parallel_for_each(
        inData.extent.tile<tileSize, tileSize>(),
        [=](tiled_index<tileSize, tileSize> tidx) restrict(amp) {
            
            tile_static int localData[tileSize][tileSize];
            localData[tidx.local[1]][tidx.local[0]] = inData[tidx.global];

            tidx.barrier.wait();
            index<2> outIdx(index<2>(tidx.tile_origin[1], tidx.tile_origin[0]) + tidx.local);
            outData[outIdx] = localData[tidx.local[0]][tidx.local[1]];
        });
    outData.synchronize();
}
