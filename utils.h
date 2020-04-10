#pragma once
#include "stdafx.h"
#include <ppl.h>

int* generateIntArray(int size);

void printArray(int* array, int size);

int** createIntMatrix(int rows, int cols);

void deleteMatrix(int rows, int** matrix);

int** generateIntMatrix(int rows, int cols);

int* matrixToArray(int rows, int cols, int** matrix);