#include "lab3.h"
#include "utils.h"

static const int initialStride = 3;
static const int window = 2;
static const int tileSize = 32;
static const int size = std::pow(tileSize, 5);// 2 * window * initialStride;

void cpuReduction(int* vector);
void strideReduction(int* vector);
void windowStrideReduction(int* vector);
void tiledReduction(int* vector);

void vectorSum() {
    cout << "Initialization" << endl;
    cout << "initialStride: " << initialStride << endl;
    cout << "window: " << window << endl;
    cout << "tileSize: " << tileSize << endl;
    cout << "size: " << size << endl << endl;
    auto vector = generateIntArray(size);
    auto timer = Timer();

    cout << "CPU reduction starts" << endl;
    timer.Start();
    cpuReduction(vector);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Stride reduction starts" << endl;
    timer.Start();
    //strideReduction(vector);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Window stride reduction starts" << endl;
    timer.Start();
    //windowStrideReduction(vector);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Tiled reduction starts" << endl;
    timer.Start();
    tiledReduction(vector);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void cpuReduction(int* vector) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += vector[i];
    cout << "CPU result: " << sum << endl;
}

void strideReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    extent<1> e(size / initialStride);
    for (int stride = initialStride - 1; stride >= 0; stride--)
    {
        parallel_for_each(e, [=](index<1> idx) restrict(amp) {
            int origin = idx[0] * initialStride;
            if (stride != 0)
                array[origin] += array[origin + stride];
            else if (idx[0] != 0)
                atomic_fetch_add(&array[0], array[origin]);
        });
        array.synchronize();
    }
    cout << "GPU result: " << array[0] << endl;
}

void windowStrideReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    extent<1> e(size / initialStride / window);
    for (int stride = initialStride - 1; stride >= 0; stride--)
    {
        parallel_for_each(e, [=](index<1> idx) restrict(amp) {
            int origin = idx[0] * window * initialStride;
            if (stride != 0) {
                for (int i = window - 1; i >= 0; i--) {
                    int windowOrigin = origin + i * initialStride;
                    array[origin] += array[windowOrigin + stride];
                }
            } else {
                for (int i = window - 1; i >= 0; i--) {
                    int windowOrigin = origin + i * initialStride;
                    if (i != 0)
                        atomic_fetch_add(&array[origin], array[windowOrigin]);
                    else if (idx[0] != 0)
                        atomic_fetch_add(&array[0], array[origin]);
                }
            }
        });
        array.synchronize();
    }
    cout << "GPU result: " << array[0] << endl;
}

void tiledReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    parallel_for_each(
        array.extent.tile<tileSize>(), 
        [=](tiled_index<tileSize> tidx) restrict(amp) {
            tile_static int local[tileSize];
            int localIdx = tidx.local[0];
            //copy
            local[localIdx] = array[tidx.global];
            tidx.barrier.wait();
            //local sum
            for (int step = 1; step < tileSize; step *= 2) { // step: 1 2 4 ...
                bool workingThread = localIdx % (2 * step) == 0;
                if (workingThread) {
                    local[localIdx] += local[localIdx + step];
                }
            }
            //tidx.barrier.wait();
            if (localIdx == 0) array[tidx.global] = local[localIdx];
        });
    array.synchronize();

    int numberOfLocalSums = size / tileSize;
    int step = tileSize;
    while (numberOfLocalSums > tileSize) {
        extent<1> e(numberOfLocalSums);
        parallel_for_each(e.tile<tileSize>(), [=](tiled_index<tileSize> tidx) restrict(amp) {
            tile_static int local[tileSize];
            int localIdx = tidx.local[0];
            //copy
            local[localIdx] = array[step * tidx.global];
            tidx.barrier.wait();
            //local sum
            for (int localStep = 1; localStep < tileSize; localStep *= 2) {
                bool workingThread = localIdx % (2 * localStep) == 0;
                if (workingThread) {
                    local[localIdx] += local[localIdx + localStep];
                }
            }
            //tidx.barrier.wait();
            if (localIdx == 0) array[step * tidx.global] = local[localIdx];
        });
        array.synchronize();
        step *= tileSize;
        numberOfLocalSums = std::ceil(numberOfLocalSums / (float)tileSize);
    }

    int sum = 0;
    for (int i = 0; i < numberOfLocalSums; i++) {
        sum += array[i * step];
    }
    cout << "GPU result: " << sum << endl;
}
