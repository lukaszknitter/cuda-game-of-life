#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
// #define PRINT // use with ROW_ELEMENTS < 50

// Cells definitions
#define ALIVE 35 // #
#define DEAD 32 // (space)
#define BORDER 66 // B

// Kernel definition
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 1

// Universe definition
#define STEPS 10
#define ROW_ELEMENTS 1000 // 3000 1x1 = 42s
#define ROW_WITH_BORDER_ELEMENTS (ROW_ELEMENTS + 2)

__device__ int getNeighboursCount(const char *universe, int i) {
    int count = 0;
    int startIndex = i - ROW_WITH_BORDER_ELEMENTS - 1;
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            int currentIndex = startIndex + k;
            if (currentIndex != i && // not itself
                universe[currentIndex] == ALIVE) {
                count++;
            }
        }
        startIndex += ROW_WITH_BORDER_ELEMENTS;
    }
    return count;
}

__global__ void computeConwayUniverse(const char *in_universe, char *out_universe, long long int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (; i < numElements; i += blockDim.x * gridDim.x) {
        if (in_universe[i] == BORDER) {
            out_universe[i] = BORDER;
        } else {
            int neighboursCount = getNeighboursCount(in_universe, i);
            out_universe[i] =
                    neighboursCount == 3 ? ALIVE : neighboursCount == 2 && in_universe[i] == ALIVE ? ALIVE : DEAD;
        }
    }
}

void checkError(cudaError_t err, const char *format) {
    if (err != cudaSuccess) {
        fprintf(stderr, format, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    srand(time(NULL));
    clock_t start, end;

    cudaError_t err = cudaSuccess;

    long long int numElements = (long long int) ROW_WITH_BORDER_ELEMENTS * (long long int) ROW_WITH_BORDER_ELEMENTS;
    size_t universe_size = numElements * sizeof(char);

    printf("[Universe of size %d x %d (%d blocks with %d threads, %d steps)]\n", ROW_ELEMENTS, ROW_ELEMENTS, BLOCKS_PER_GRID, THREADS_PER_BLOCK, STEPS);

    char *h_universe = (char *) malloc(universe_size);
    if (h_universe == NULL) {
        fprintf(stderr, "Failed to allocate host h_universe!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize universe
    for (int i = 0; i < numElements; ++i) {
        if (i < ROW_WITH_BORDER_ELEMENTS ||  // first row
            i >= numElements - ROW_WITH_BORDER_ELEMENTS || // last row 
            i % ROW_WITH_BORDER_ELEMENTS == 0 || // first column
            i != 0 && i % ROW_WITH_BORDER_ELEMENTS == ROW_WITH_BORDER_ELEMENTS - 1) { // last column
            h_universe[i] = BORDER;
        } else {
            h_universe[i] = rand() % 2 == 0 ? DEAD : ALIVE;
        }
    }

    char *d_in_universe = NULL;
    err = cudaMalloc((void **) &d_in_universe, universe_size);
    checkError(err, "Failed to allocate device universe (error code %s)!\n");

    char *d_out_universe = NULL;
    err = cudaMalloc((void **) &d_out_universe, universe_size);
    checkError(err, "Failed to allocate device universe (error code %s)!\n");

    start = clock();
    for (int i = 0; i < STEPS; i++) {
        err = cudaMemcpy(d_in_universe, h_universe, universe_size, cudaMemcpyHostToDevice);
        checkError(err, "Failed to copy universe from host to device (error code %s)!\n");

        computeConwayUniverse<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_in_universe, d_out_universe, numElements);

        err = cudaGetLastError();
        checkError(err, "Failed to launch Conway kernel (error code %s)!\n");

        err = cudaMemcpy(h_universe, d_out_universe, universe_size, cudaMemcpyDeviceToHost);
        checkError(err, "Failed to copy universe from device to host (error code %s)!\n");

#ifdef PRINT
        printf("\e[0;1H\e[2J");
        printf("Step (%d) CUDA Conway kernel launch with %d blocks of %d threads\n", i, BLOCKS_PER_GRID, THREADS_PER_BLOCK);
        for (int i=0; i < numElements; ++i) {
            if (i % ROW_WITH_BORDER_ELEMENTS == 0){
                printf("\n");
            }
                printf("%c ", h_universe[i]);
        }
        printf("\n");
        sleep(1);
#endif
    }
    end = clock();
    printf("Computations took %f s\n", ((double) (end - start) / CLOCKS_PER_SEC));

    err = cudaFree(d_in_universe);
    checkError(err, "Failed to free device in universe (error code %s)!\n");

    err = cudaFree(d_out_universe);
    checkError(err, "Failed to free device out universe (error code %s)!\n");

    free(h_universe);

    err = cudaDeviceReset();
    checkError(err, "Failed to deinitialize the device! error=%s\n");

    return 0;
}
