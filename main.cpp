#pragma once
#include "lab1.h"

void lab1();

int main() {
    lab1();
    return 0;
}

void lab1() {
    cout << "GPU INFO" << endl;
    gpusInfo();

    cout << "ARRAY SUM" << endl;
    arraySum();

    cout << "MATRIX BY NUMBER MULTIPLICATION" << endl;
    matrixByNumMult();

    cout << "MATRIX TRANSPONSE" << endl;
    matrixTransponse();

    cout << "MATRIX BY MATRIX MULTIPLICATION" << endl;
    matrixByMatrixMult();
}
