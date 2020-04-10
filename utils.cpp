#include "utils.h"

int* generateIntArray(int size)
{
	auto array = new int[size];
	for (size_t i = 0; i < size; i++) {
		array[i] = rand();
	}
	return array;
}

void printArray(int* array, int size){
	for (size_t i = 0; i < size; i++) {
		cout << array[i] << " ";
	}
	cout << endl;
}

int** createIntMatrix(int rows, int cols) {
	int** matrix = new int* [rows];
	for (int i = 0; i < rows; i++)
		matrix[i] = new int[cols];
	return matrix;
}

void deleteMatrix(int rows, int** matrix) {
	for (int i = 0; i < rows; i++) delete[] matrix[i];
	delete[] matrix;
}

int** generateIntMatrix(int rows, int cols) {
	int** matrix = new int* [rows];
	for (int i = 0; i < rows; i++) {
		int* row = new int[cols];
		for (int j = 0; j < cols; j++) row[j] = rand();
		matrix[i] = row;
	}
	return matrix;
}

int* matrixToArray(int rows, int cols, int** matrix) {
	int* array = new int[rows * cols];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) 
			array[i * cols + j] = matrix[i][j];
	return array;
}
