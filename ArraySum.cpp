#include "lab1.h"
#include "utils.h"

void sumOnSingleCpu(int n, int* pA, int* pB, int* pC);
void sumOnMultipleCpus(int n, int* pA, int* pB, int* pC);
void sumOnGpu(int n, int* pA, int* pB, int* pC);

void arraySum() {
    cout << "Initialization" << endl;
    auto size = 100000000;
    cout << "size: " << size << endl << endl;
    auto pA = generateIntArray(size);
    auto pB = generateIntArray(size);
    auto pC = new int[size];
    auto timer = Timer();

    cout << "Single CPU starts" << endl;
    timer.Start();
    sumOnSingleCpu(size, pA, pB, pC);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "CPUs in parallel starts" << endl;
    timer.Start();
    sumOnMultipleCpus(size, pA, pB, pC);
    timer.Stop();
    cout << "CPU done in " << timer.Elapsed() << "ms" << endl << endl;

    cout << "GPU starts" << endl;
    timer.Start();
    sumOnGpu(size, pA, pB, pC);
    timer.Stop();
    cout << "GPU done in " << timer.Elapsed() << "ms" << endl << endl;
}

void sumOnSingleCpu(int n, int* pA, int* pB, int* pC) {
    for (int i = 0; i < n; i++) {
        pC[i] = pA[i] + pB[i];
    }
}

void sumOnMultipleCpus(int n, int* pA, int* pB, int* pC) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        pC[i] = pA[i] + pB[i];
    }
}

void sumOnGpu(int n, int* pA, int* pB, int* pC) {
    array_view<const int, 1> a(n, pA);
    array_view<const int, 1> b(n, pB);
    array_view<int, 1> sum(n, pC);
    sum.discard_data();
    parallel_for_each(sum.extent, [=](index<1> idx) restrict(amp) { sum[idx] = a[idx] + b[idx]; });
    sum.synchronize();
}