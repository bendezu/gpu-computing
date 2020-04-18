#include "stdafx.h"
#pragma once

void gpusInfo();

void arraySum();

void matrixByNumMult();

void matrixTransponse();
void transponseOnGpu(int rows, int cols, int* matrix, int* result);

void matrixByMatrixMult();
void multiplyOnGpu(int rows1, int internalDim, int cols2, int* first, int* second, int* result);