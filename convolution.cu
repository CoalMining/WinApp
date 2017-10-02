#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;


__global__ 
void convolution2D(int* firstMatrix, int* secondMatrix, int* resultMatrix, int l1, int m1, int l2, int m2);

int *firstMatrix, *kernelMatrix, *resultMatrix;

int main(int argc, char* argv)
{
	int l1, m1;
	int l2, m2;

	cout << "Enter the dimension of the first matrix" << endl;
	cin >> l1 >> m1;
	
	cout << "Enter the elements of the first matrix\n\nRow major Order:" << endl;
	firstMatrix = new int[l1*m1];
	resultMatrix = new int[l1*m1];
	for (int i = 0; i < l1*m1; i++)
		cin >> firstMatrix[i];

	cout << "Enter the dimension of the kernel matrix" << endl;
	cin >> l2 >> m2;

	cout << "Enter the elements of the kernel matrix\n\nRow major Order:" << endl;
	kernelMatrix = new int[l2*m2];
	for (int i = 0; i < l2*m2; i++)
		cin >> kernelMatrix[i];

	int *d_firstMatrix, *d_kernelMatrix, *d_resultMatrix;
	if (cudaMalloc((void**)&d_firstMatrix, l1*m1 * sizeof(int)) != cudaSuccess)
	{
		cout << "Error in allocating firstmatrix on device" << endl;
		return -1;
	}
	if (cudaMalloc((void**)&d_resultMatrix, l1*m1 * sizeof(int)) != cudaSuccess)
	{
		cout << "Error in allocating result matrix on device" << endl;
		return -1;
	}
	if (cudaMalloc((void**)&d_kernelMatrix, l2*m2 * sizeof(int)) != cudaSuccess)
	{
		cout << "Error in allocating kernel matrix on device" << endl;
		return -1;
	}

	if (cudaMemcpy(d_firstMatrix, firstMatrix, sizeof(int)*l1*m1, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Error in copying data for first matrix to device" << endl;
		return -1;
	}

	if (cudaMemcpy(d_kernelMatrix, kernelMatrix, sizeof(int)*l2*m2, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Error in copying data for second matrix to device" << endl;
		return -1;
	}

	//following section launches kernel
	convolution2D << < 1,dim3(m1,l1,1)>> > (d_firstMatrix,d_kernelMatrix, d_resultMatrix,l1,m1,l2,m2);
	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		cout << "Error in synchronizing device" << endl;
		return -1;
	}

	if (cudaMemcpy(resultMatrix,d_resultMatrix,sizeof(int)*l1*m1,cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		cout << "Error in copying data back from device to host" << endl;
		return -1;
	}

	cout << "The elements in row major order are as follows" << endl;
	for (int i = 0; i < l1*m1; i++)
	{
		cout << resultMatrix[i] << " ";
	}

	cudaFree(d_firstMatrix);
	cudaFree(d_kernelMatrix);
	cudaFree(d_resultMatrix);

	delete[] firstMatrix;
	delete[] kernelMatrix;
	delete[] resultMatrix;


	return 0;
}


__global__ 
void convolution2D(int* firstMatrix, int* kernelMatrix, int* resultMatrix, int l1, int m1, int l2, int m2)
{
	//x is horizontal, y is vertical in Matrix
	//kernel call should be accordingly
	int tIdX = threadIdx.x + blockDim.x*blockIdx.x;
	int tIdY = threadIdx.y + blockDim.y*blockIdx.y;

	//l2 is no of rows so yShift comes from l2
	int yShift = l2 / 2;	//floor of l2/2
	int xShift = m2 / 2;	//floor of m2/2
	
	//starting index of element for the first matrix
	int yStart = tIdY - yShift;
	int xStart = tIdX - xShift;

	//private to each thread
	int tempRes = 0;

	for (int i = 0; i < l2 ; i++)
	{
		for (int j = 0; j < m2 ; j++)
		{
			//these loops are supposed to go through all the elememts of  the kernel matrix, so the loop

			//the element from the first matrix tthat are multiplied with the elements of the kernel matrix can be found by following:
			//	-> the X and Y index of the thread gives the center element
			//	-> subtract the xShift from X dim and yShift from Y dim to get the element of the first matrix to be multiplied with the first element of the kernel matrix
			//							(only if the index after shifting is greater than 0)
			//	-> since the shifting should go such that the elments of the kernel is mutliplied to corresponding element of first matrix add i and j
			if ((yStart + i) >= 0 && (xStart + j) >= 0 && (yStart + i)<l1 && (xStart + j)<m1)
			{
				tempRes += (kernelMatrix[i*m2 + j] * firstMatrix[(yStart + i)*m1 + (xStart + j)]);
			}
		}
	}
	resultMatrix[m1*tIdY + tIdX] = tempRes;
}