#include "cuda.h"

#include <iostream>

using namespace std;

int *firstMatrix, *secondMatrix;

int main(int argc, char* argv)
{
	int l1, m1;
	int l2, m2;

	cout << "Enter the dimension of the first matrix" << endl;
	cin >> l1 >> m1;

	cout << "Enter the dimension of the second matrix" << endl;
	cin >> l2 >> m2;

	cout << "Enter the elements of the first matrix\n\nRow major Order:" << endl;
	firstMatrix = new int[l1*m1];
	for (int i = 0; i < l1*m1; i++)
		cin >> firstMatrix[i];

	cout << "Enter the elements of the second matrix\n\nRow major Order:" << endl;
	secondMatrix = new int[l2*m2];
	for (int i = 0; i < l2*m2; i++)
		cin >> secondMatrix[i];


	delete[] firstMatrix;
	delete[] secondMatrix;
	return 0;
}