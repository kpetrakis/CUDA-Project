
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

// MSVC defines this in winsock2.h!?
typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
#define NI 9000
#define NJ 9000


//__launch_bounds__(1024, 6)
__global__ void  Convolution(double *a, double *b, int ni, int nj)
{
	int Col = blockDim.x*blockIdx.x + threadIdx.x;
	int	Row = blockDim.y*blockIdx.y + threadIdx.y;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((Row>0)&& (Col>0)&& (Row < ni-1) && (Col < nj-1)) {
		
		b[Row*nj + Col] = c11 * a[(Row - 1)*nj + (Col - 1)] + c12 * a[(Row + 0)*nj + (Col - 1)] + c13 * a[(Row + 1)*nj + (Col - 1)]
			+ c21 * a[(Row - 1)*nj + (Col + 0)] + c22 * a[(Row + 0)*nj + (Col + 0)] + c23 * a[(Row + 1)*nj + (Col + 0)]
			+ c31 * a[(Row - 1)*nj + (Col + 1)] + c32 * a[(Row + 0)*nj + (Col + 1)] + c33 * a[(Row + 1)*nj + (Col + 1)];
	
	}
}

void init(double* A)
{
	int i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand() / RAND_MAX;
		}
	}
}


int main()
{
	const int size = NI*NJ;
	double *A;
	double *B;
	struct timeval	cpu_start, cpu_end;

	A = (double*)malloc(size * sizeof(double));
	B = (double*)malloc(size * sizeof(double));

	init(A);


	double *dev_a = 0;
	double *dev_b = 0;
	cudaError_t cudaStatus;




	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, A, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, B, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}

	unsigned int BLOCK_SIZE_PER_DIM = 16;

	unsigned int numBlocksX = (NJ - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (NI - 1) / BLOCK_SIZE_PER_DIM + 1;
	//printf("%d\n", numBlocksX);
	//printf("%d\n", numBlocksY);

	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);
	dim3 dimGrid(numBlocksX, numBlocksY, 1);

	gettimeofday(&cpu_start, NULL);
	// Launch a kernel on the GPU with one thread for each element.
	Convolution << <dimGrid, dimBlock>> >(dev_a, dev_b, NI, NJ);
	//gettimeofday(&cpu_end, NULL);
	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec)*1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		system("pause");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		system("pause");
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(B, dev_b, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec)*1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);
Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda failed!");
		return 1;
	}
	
	/*for (int i = 0; i < NI ; i++) {
		for (int j = 0; j < NJ; j++) {
			printf("%lf ", B[i*NJ + j]);
		}
		printf("\n\n");
	}*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
	 fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(A);
	free(B);
	system("pause");
	return 0;
}




