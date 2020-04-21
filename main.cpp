#pragma once
#include "lab1.h"
#include "lab2.h"

void lab1();
void lab2();

int main() {
    lab2();
    return 0;
}

void lab1() {
    cout << "GPU INFO" << endl;
    gpusInfo();

    cout << "ARRAY SUM" << endl;
    arraySum();

    cout << "MATRIX BY NUMBER MULTIPLICATION" << endl;
    matrixByNumMult();

    cout << "MATRIX TRANSPOSITION" << endl;
    matrixTransposition();

    cout << "MATRIX BY MATRIX MULTIPLICATION" << endl;
    matrixByMatrixMult();
}

void lab2() {
    cout << "BLOCK MATRIX TRANSPOSITION" << endl;
    blockTransposition();

    cout << "BLOCK MATRIX MULTIPLICATION" << endl;
    blockMatrixMult();
}