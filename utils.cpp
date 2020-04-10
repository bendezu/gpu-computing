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
