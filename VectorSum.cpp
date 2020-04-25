#include "lab3.h"
#include "utils.h"

static const int initialStride = 6;
static const int window = 30;
static const int size = 100000 * window * initialStride;

void cpuReduction(int* vector);
void strideReduction(int* vector);
void windowStrideReduction(int* vector);

void vectorSum() {
    cout << "Initialization" << endl;
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
    strideReduction(vector);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "Window stride reduction starts" << endl;
    timer.Start();
    windowStrideReduction(vector);
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