#include "lab3.h"
#include "utils.h"

static const int ITERATIONS = 5;

static const int WINDOW = 2048;
static const int TILE_SIZE = 1024;
static const int size = std::pow(2, 22);

int cpuReduction(int* vector);
int strideReduction(int* vector);
int windowStrideReduction(int* vector);
int tiledReduction(int* vector);
int windowTileReduction(int* vector);

void launchExperiment(char* title, std::function<int()> calculation) {
    auto timer = Timer();
    float avg = 0;
    cout << title << " starts" << endl;
    for (int i = 0; i <= ITERATIONS; i++) {
        timer.Start();
        int result = calculation();
        timer.Stop();
        if (i == 0) {
            cout << "checksum: " << result << endl;
            continue;
        }
        auto elapsed = timer.Elapsed();
        cout << "iteration " << i << ": " << elapsed << "ms" << endl;
        avg += elapsed / ITERATIONS;
    }
    cout << "Average is " << avg << "ms" << endl << endl;
}

void vectorSum() {
    cout << "Initialization" << endl;
    cout << "window: " << WINDOW << endl;
    cout << "tileSize: " << TILE_SIZE << endl;
    cout << "size: " << size << endl << endl;
    auto vector = generateIntArray(size);

    launchExperiment("CPU reduction", [=]() { return cpuReduction(vector); });
    //launchExperiment("Stride reduction", [=]() { return strideReduction(vector); });
    //launchExperiment("Window stride reduction", [=]() { return windowStrideReduction(vector); });
    //launchExperiment("Tiled reduction", [=]() { return tiledReduction(vector); });
    launchExperiment("Window Tile reduction", [=]() { return windowTileReduction(vector); });
}

int cpuReduction(int* vector) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += vector[i];
    return sum;
}

int strideReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    for (int stride = size / 2; stride >= 1; stride/=2)
    {
        extent<1> e(stride);
        parallel_for_each(e, [=](index<1> idx) restrict(amp) {
                array[idx[0]] += array[idx[0] + stride];
        });
        array.synchronize();
    }
    return array[0];
}

int windowStrideReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    int _window = WINDOW;
    for (int stride = size / WINDOW; stride >= 1; stride /= WINDOW)
    {
        extent<1> e(stride);
        parallel_for_each(e, [=](index<1> idx) restrict(amp) {
            for (int i = 1; i < _window; i++) {
                array[idx[0]] += array[idx[0] + i * stride];
            }
        });
        array.synchronize();
    }
    return array[0];
}

int tiledReduction(int* vector) {
    array_view<int, 1> array(size, vector);
    parallel_for_each(
        array.extent.tile<TILE_SIZE>(),
        [=](tiled_index<TILE_SIZE> tidx) restrict(amp) {
            tile_static int local[TILE_SIZE];
            int localIdx = tidx.local[0];
            //copy
            local[localIdx] = array[tidx.global];
            tidx.barrier.wait();
            //local sum
            for (int step = 1; step < TILE_SIZE; step *= 2) { // step: 1 2 4 ...
                bool workingThread = localIdx % (2 * step) == 0;
                if (workingThread) {
                    local[localIdx] += local[localIdx + step];
                }
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[tidx.global] = local[localIdx];
        });
    array.synchronize();
    // дополнительные запуски на gpu (для наглядности не стал объединять с предыдущем запуском)
    int numberOfLocalSums = size / TILE_SIZE;
    int step = TILE_SIZE;
    while (numberOfLocalSums > TILE_SIZE) {
        extent<1> e(numberOfLocalSums);
        parallel_for_each(e.tile<TILE_SIZE>(), [=](tiled_index<TILE_SIZE> tidx) restrict(amp) {
            tile_static int local[TILE_SIZE];
            int localIdx = tidx.local[0];
            //copy
            local[localIdx] = array[step * tidx.global];
            tidx.barrier.wait();
            //local sum
            for (int localStep = 1; localStep < TILE_SIZE; localStep *= 2) {
                bool workingThread = localIdx % (2 * localStep) == 0;
                if (workingThread) {
                    local[localIdx] += local[localIdx + localStep];
                }
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[step * tidx.global] = local[localIdx];
        });
        array.synchronize();
        step *= TILE_SIZE;
        numberOfLocalSums = std::ceil(numberOfLocalSums / (float)TILE_SIZE);
    }
    // sum on cpu
    int sum = 0;
    for (int i = 0; i < numberOfLocalSums; i++) {
        sum += array[i * step];
    }
    return sum;
}

int windowTileReduction(int* vector) {
    int _window = WINDOW;
    array_view<int, 1> array(size, vector);
    extent<1> e(size / WINDOW);
    parallel_for_each(
        e.tile<TILE_SIZE>(),
        [=](tiled_index<TILE_SIZE> tidx) restrict(amp) {
            tile_static int local[TILE_SIZE];
            int localIdx = tidx.local[0];
            int globalIdx = tidx.global[0] * _window;
            //copy
            int localSum = array[globalIdx];
            for (int i = 1; i < _window; i++) {
                localSum += array[globalIdx + i];
            }
            local[localIdx] = localSum;
            tidx.barrier.wait();
            //local tiled sum
            for (int step = TILE_SIZE / 2; step > 0; step /= 2) {
                if (localIdx < step)
                    local[localIdx] += local[localIdx + step];
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[globalIdx] = local[localIdx];
        });
    array.synchronize();
    int numberOfLocalSums = size / TILE_SIZE / _window;
    // sum on cpu
    int sum = 0;
    for (int i = 0; i < numberOfLocalSums; i++) {
        sum += array[i * TILE_SIZE * _window];
    }
    return sum;
}
