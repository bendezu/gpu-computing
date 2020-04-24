#include "lab3.h"
#include "utils.h"

static const int size = 10;

void reduceVector(int* vector);

void vectorSum() {
    cout << "Initialization" << endl;
    cout << "size: " << size << endl << endl;
    auto vector = generateIntArray(size);
    auto timer = Timer();

    printArray(vector, size);
    int sum = 0;
    for (size_t i = 0; i < size; i++) sum += vector[i];
    cout << "CPU result: " << sum << endl;

    cout << "Regular reduction starts" << endl;
    timer.Start();
    reduceVector(vector);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void reduceVector(int* vector) {
    int s = 0;
    array_view<const int, 1> array(size, vector);
    array_view<int, 1> sum(1, &s);
    parallel_for_each(array.extent, [=](index<1> idx) restrict(amp) {
        atomic_fetch_add(&sum[0], array[idx]);
    });
    array.synchronize();
    cout << "GPU result: " << sum[0] << endl;
}