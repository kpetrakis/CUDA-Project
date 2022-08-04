
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <io.h>
#include <assert.h>
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

#define M 1024
#define N 1024

#define BLOCK_SIZE 16

#define FLOAT_N 3214212.01

void init_arrays(double* data)
{
	int i, j;

	for (i = 1; i < (M+1); i++) {
		for (j = 1; j < (N+1); j++) {
			data[i*(N+1)  + j] = ((double)i*j) / M;
		}
	}
}

__global__ void covariance(double *data, double *mean, double *symmat, int m, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	unsigned int Col = bx*blockDim.x + tx;
	unsigned int Row = by*blockDim.y + ty;
	
	__shared__ double data_s1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double data_s2[BLOCK_SIZE][BLOCK_SIZE];

	double float_n = 3214212.01;
	while (Col < n) {
		double result = 0;
		for (int i = 0; i < m; i++)
			result += data[i*n + Col];
		mean[Col] = result/float_n;
		Col += gridDim.x*blockDim.x;
	}
	while (Col < n) {
		for (int i = 0; i < m; i++)
			data[i*n + Col] -= mean[Col];
		Col += gridDim.x*blockDim.x;
	}

	double Pvalue = 0.0;
	
	for (int l = 0; l <((n+BLOCK_SIZE-1)/ BLOCK_SIZE); l++) {
		// Colaborative loading of tiles into shared memory
		data_s1[ty][tx] = data[Row * n + l * BLOCK_SIZE + tx];
		data_s2[ty][tx] = data[Col + (l * BLOCK_SIZE + ty) * m];
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++) {
			if ((l*BLOCK_SIZE + k) > m) break;
			Pvalue += data_s1[ty][k] * data_s2[k][tx];

		}
		  __syncthreads();
		
	}
	
	symmat[Row*n + Col] = Pvalue;

   
}

int main(int argc, char *argv[])
{
	double		*data;
	double		*symmat;
	double		*mean;
	struct timeval	cpu_start, cpu_end;

	data = (double*)malloc((M + 1)*(N + 1) * sizeof(double));
	symmat = (double*)malloc((M + 1)*(N + 1) * sizeof(double));
	mean = (double*)malloc((M+1) * sizeof(double));

	init_arrays(data);

	double *dev_data = 0;
	double *dev_mean = 0;
	double *dev_symmat = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_data, (M+1)*(N+1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_symmat, (M + 1)*(N + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mean, N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data, data, (M + 1)*(N + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}

	
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gettimeofday(&cpu_start, NULL);
	covariance << <dimGrid, dimBlock >> >(dev_data, dev_mean, dev_symmat, M, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		system("pause");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		system("pause");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(symmat, dev_symmat, (M + 1)*(N + 1) *sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

Error:
	cudaFree(dev_data);
	cudaFree(dev_mean);
	cudaFree(dev_symmat);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda failed!");
		return 1;
	}

	/*for (int i = 0; i < M ; i++) {
		for (int j = 0; j < N; j++) {
			printf("%lf ", symmat[i*N+j]);
		}
			printf("\n");
	}*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(data);
	free(symmat);
	free(mean);
	system("pause");
	return 0;
}


