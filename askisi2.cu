#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>
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

#define NX 4000
#define NY 4000
#define BLOCK_SIZE 256

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(double *x, double *A)
{
	int i, j;

	for (i = 0; i < NX; i++) {
		//x[i] = i * M_PI;
		for (j = 0; j < NY; j++) {
			A[i*NY + j] = ((double)i*(j)) / NX;
		}
	}
	for (j = 0; j<NY; j++) {
		x[j] = j*M_PI;
	}

}

__global__ void trans_norm_vector(double* A, double* x, double* tmp, double *y, int nx, int ny)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ double x_s[BLOCK_SIZE];
	__shared__ double tmp_s[BLOCK_SIZE];
		double sum = 0.0;
		double res = 0.0;

	for (int m = 0; m < ((ny + BLOCK_SIZE - 1) / BLOCK_SIZE); m++) {
		if ((m*BLOCK_SIZE + threadIdx.x) < ny) x_s[threadIdx.x] = x[threadIdx.x + m*BLOCK_SIZE];
		else x_s[threadIdx.x] = 0.f;
		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; e++) {
			//To A*x ginetai kata stiles-pio grigoro
			sum += A[tid + (e + BLOCK_SIZE*m)*nx] * x_s[e];
		}
		__syncthreads();
	}
	if (tid < nx) tmp[tid] = sum;

	for (int m = 0; m < ((ny + BLOCK_SIZE - 1) / BLOCK_SIZE); m++) {
		if ((m*BLOCK_SIZE + threadIdx.x) < ny) tmp_s[threadIdx.x] = tmp[threadIdx.x + m*BLOCK_SIZE];
		else tmp_s[threadIdx.x] = 0.f;
		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; e++) {
			res += A[tid + (e + BLOCK_SIZE*m)*nx] * tmp_s[e];
		}
		__syncthreads();
	}
	if (tid < nx) y[tid] = res;
	
	

}

int main(int argc, char *argv[])
{
	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	struct timeval	cpu_start, cpu_end;

	A = (double*)malloc(NX*NY * sizeof(double));
	x = (double*)malloc(NY * sizeof(double));
	y = (double*)malloc(NY * sizeof(double));
	tmp = (double*)malloc(NX * sizeof(double));

	init_array(x, A);

	double *dev_a = 0;
	double *dev_x = 0;
	double *dev_y = 0;
	double *dev_tmp = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, NX*NY * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, NY * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tmp, NX * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, NX * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		system("pause");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, A, NX*NY * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_x, x, NY * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}


	//trans_norm_vector(A, x, y, tmp);
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((NX+BLOCK_SIZE-1)/BLOCK_SIZE);
	gettimeofday(&cpu_start, NULL);
	trans_norm_vector << <dimGrid, dimBlock >> >(dev_a, dev_x, dev_tmp, dev_y, NX, NY);
	

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
	cudaStatus = cudaMemcpy(y, dev_y, NX * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		system("pause");
		goto Error;
	}
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);
Error:
	cudaFree(dev_a);
	cudaFree(dev_x);
	cudaFree(dev_tmp);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda failed!");
		return 1;
	}
	/*for (int i = 0; i < NX ; i++) {
	printf("%lf \n", y[i]);
	}*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(A);
	free(x);
	free(y);
	free(tmp);
	system("pause");
	return 0;
}
