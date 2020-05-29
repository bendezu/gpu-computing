#include "lab3.h"
#include "utils.h"

static const int ITERATIONS = 5;

static const int WINDOW = std::pow(2, 10);
static const int tileSize = 1024;
static const int size = std::pow(tileSize, 1) * WINDOW;

int cpuReduction(int* vector);
int strideReduction(int* vector);
int windowStrideReduction(int* vector);
int tiledReduction(int* vector);
int fixedTiledReduction(int* vector);

void vectorSum() {
    cout << "Initialization" << endl;
    cout << "window: " << WINDOW << endl;
    cout << "tileSize: " << tileSize << endl;
    cout << "size: " << size << endl << endl;
    auto vector = generateIntArray(size);
    auto timer = Timer();
    float avg = 0;

    cout << "CPU reduction starts" << endl;
    for (int i = 0; i <= ITERATIONS; i++) {
        timer.Start();
        int result = cpuReduction(vector);
        timer.Stop();
        if (i == 0) { 
            cout << "checksum: " << result << endl; 
            continue; 
        }
        auto elapsed = timer.Elapsed();
        cout << "iteration " << i << ": " << elapsed << "ms" << endl;
        avg += elapsed / ITERATIONS;
    }
    cout << "CPU average is " << avg << "ms" << endl << endl;

    cout << "Stride reduction starts" << endl;
    avg = 0;
    for (int i = 0; i <= ITERATIONS; i++) {
        timer.Start();
        int result = 0;// strideReduction(vector);
        timer.Stop();
        if (i == 0) {
            cout << "checksum: " << result << endl;
            continue;
        }
        auto elapsed = timer.Elapsed();
        cout << "iteration " << i << ": " << elapsed << "ms" << endl;
        avg += elapsed / ITERATIONS;
    }
    cout << "GPU average is " << avg << "ms" << endl << endl;

    cout << "Window stride reduction starts" << endl;
    avg = 0;
    for (int i = 0; i <= ITERATIONS; i++) {
        timer.Start();
        int result = 0;// windowStrideReduction(vector);
        timer.Stop();
        if (i == 0) {
            cout << "checksum: " << result << endl;
            continue;
        }
        auto elapsed = timer.Elapsed();
        cout << "iteration " << i << ": " << elapsed << "ms" << endl;
        avg += elapsed / ITERATIONS;
    }
    cout << "GPU average is " << avg << "ms" << endl << endl;

    cout << "Tiled reduction starts" << endl;
    avg = 0;
    for (int i = 0; i <= ITERATIONS; i++) {
        timer.Start();
        int result = tiledReduction(vector);
        timer.Stop();
        if (i == 0) {
            cout << "checksum: " << result << endl;
            continue;
        }
        auto elapsed = timer.Elapsed();
        cout << "iteration " << i << ": " << elapsed << "ms" << endl;
        avg += elapsed / ITERATIONS;
    }
    cout << "GPU average is " << avg << "ms" << endl << endl;
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
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[tidx.global] = local[localIdx];
        });
    array.synchronize();
    // дополнительные запуски на gpu (для наглядности не стал объединять с предыдущем запуском)
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
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[step * tidx.global] = local[localIdx];
        });
        array.synchronize();
        step *= tileSize;
        numberOfLocalSums = std::ceil(numberOfLocalSums / (float)tileSize);
    }
    // sum on cpu
    int sum = 0;
    for (int i = 0; i < numberOfLocalSums; i++) {
        sum += array[i * step];
    }
    return sum;
}

int fixedTiledReduction(int* vector) {
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
            for (int step = tileSize / 2; step > 0; step /= 2) {
                if (localIdx < step)
                    local[localIdx] += local[localIdx + step];
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[tidx.global] = local[localIdx];
        });
    array.synchronize();
    // дополнительные запуски на gpu (для наглядности не стал объединять с предыдущем запуском)
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
            for (int localStep = tileSize / 2; localStep > 0; localStep /= 2) {
                if (localIdx < localStep)
                    local[localIdx] += local[localIdx + localStep];
                tidx.barrier.wait();
            }
            if (localIdx == 0) array[step * tidx.global] = local[localIdx];
            });
        array.synchronize();
        step *= tileSize;
        numberOfLocalSums = std::ceil(numberOfLocalSums / (float)tileSize);
    }
    // sum on cpu
    int sum = 0;
    for (int i = 0; i < numberOfLocalSums; i++) {
        sum += array[i * step];
    }
    return sum;
}
