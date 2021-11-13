#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <vector> //for vector  
#include <algorithm> //for generate
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <windows.h> //winapi header  

using namespace std;

// method used while debugging for printing an array
__global__ void printArray(int* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

void debugArray(char a[], int* arr, int n) {
    printf("DEBUGGING %s\n", a);
    printArray << <1, 1 >> > (arr, n);
}

// creating a histogram 
__global__ void createHistogram(int* a, int* h, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int pos = a[tid];
    atomicAdd(&h[pos], 1);

}

// function created to get the prefix-sum of an array using the inclusive scan method
void prefix_sum_on_gpu(int* data, int* output, int size) {
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
    // Allocate temporary storage for inclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
    //printf("Successfully prefixed sum");
}
int* cubsort(int* d_x, int N)
{
    // Declare, allocate, and initialize device-accessible pointers for sorting data
    int  num_items = N;          // e.g., 7
    int* d_keys_in = d_x;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int* d_keys_out;        // e.g., [        ...        ]
    cudaMalloc(&d_keys_out, sizeof(int) * N);
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    // d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
    return d_keys_out;
}
void test_cubsort(int* d_x, int N) {
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int* d_y = cubsort(d_x, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Took %f milliseconds using cubsort\n", milliseconds);
    cudaFree(d_y);
}

int main()
{
    //N = number of elements
    //M = maximum element + 1 (histogram size)
    int N, M;
    cin >> N >> M;
    float milliseconds = 0;


    size_t bytesN = sizeof(int) * N;
    size_t bytesM = sizeof(int) * M;

    vector<int> x(N), s(N);
    generate(x.begin(), x.end(), [M]() {return rand() % M; });

    int* d_x, * d_a, * d_ap, * d_b, * d_bp;

    /*------------------------START OF THE GPU COMPUTATION--------------------------*/

    // x is input arr
    // d_x is copy of x on gpu
    cudaMalloc(&d_x, bytesN);
    cudaMemcpy(d_x, x.data(), bytesN, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;
    int numBlocks2 = (M + numThreads - 1) / numThreads;

    cudaMalloc(&d_a, bytesM);
    //d_a is histogram of d_x 
    createHistogram << <numThreads, numBlocks >> > (d_x, d_a, N);

    cudaMalloc(&d_ap, bytesM);
    //d_ap is prefix sum of d_a
    prefix_sum_on_gpu(d_a, d_ap, M);

    cudaMalloc(&d_b, bytesN);
    //d_b is histogram of d_ap
    createHistogram << <numThreads, numBlocks2 >> > (d_ap, d_b, M);

    cudaMalloc(&d_bp, bytesN);
    //d_bp is the prefix-sum of db
    prefix_sum_on_gpu(d_b + 1, d_bp + 1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Took %f milliseconds using HP\n", milliseconds);
    test_cubsort(d_x, N);

    cudaMemcpy(s.data(), d_bp, bytesN, cudaMemcpyDeviceToHost);

    /*------------------------END OF THE GPU COMPUTATION--------------------------*/

    // x is the initial array and s is the sorted array.
    /*for (auto& element : x) cout << element << " ";
    cout << endl;

    for (auto& element : s) cout << element << " ";
    cout << endl;*/
    return 0;
}