#include <iostream>
#include <amp.h>

using namespace std;
using namespace concurrency;

void AddArrays(int n, int *pA, int *pB, int *pC);

int main() {
    auto size(10);
    auto pA = new int[size];
    auto pB = new int[size];
    auto pC = new int[size];
    for (int i = 0; i < size; ++i) {
        pA[i] = i;
        pB[i] = i;
    }

    AddArrays(10, pA, pB, pC);

    for (int i = 0; i < size; ++i) {
        cout << pC[i] << " ";
    }

    return 0;
}

void AddArrays(int n, int *pA, int *pB, int *pC) {
    array_view<const int, 1> a(n, pA);
    array_view<const int, 1> b(n, pB);
    array_view<int, 1> sum(n, pC);
    sum.discard_data();
    parallel_for_each(sum.extent, [=](index<1> idx) restrict(amp) { sum[idx] = a[idx] + b[idx]; });
    sum.synchronize();
}