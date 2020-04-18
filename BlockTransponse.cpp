#include "lab2.h"
#include "utils.h"
#include "lab1.h"

static const int tileSize = 32;
static const int rows = tileSize * 300;
static const int cols = tileSize * 200;
void blockTransponseOnGpu(int* matrix, int* result);

void blockTransponse() {
    cout << "Initialization" << endl;
    cout << "dimentions: " << rows << "x" << cols << endl << endl;
    auto timer = Timer();
    auto matrixAsArray = generateIntArray(rows * cols);
    auto resultAsArray = new int[cols * rows];

    cout << "Regular transponse starts" << endl;
    timer.Start();
    transponseOnGpu(rows, cols, matrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Block transponse starts" << endl;
    timer.Start();
    blockTransponseOnGpu(matrixAsArray, resultAsArray);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void blockTransponseOnGpu(int* matrix, int* result) {
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
