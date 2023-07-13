
extern "C" {

    __constant__ float filterData[64*64];
    __global__ void correlateShared(unsigned char* input, unsigned char* out, 
            int inp_rows, int inp_cols, int filter_rows, 
            int filter_cols, int maxDown, int maxRight
    ) {
        // make filter size constant?, or use extern shared
        __shared__ unsigned char sharedInput[32 + 16][32 + 16];

        int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
        int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

        if (moveDown >= maxDown) {
            return;
        }
        if (moveRight >= maxRight) {
            return;
        }

        // 32 + filter_rows, 32 + filter_cols ==> 32 + 16, 32 + 16, however, theadIdx max is 32
        if (threadIdx.x < 32 && threadIdx.y < 32) {
            //sharedInput[threadIdx.x * (blockDim.y + filter_cols) + threadIdx.y] = input[moveDown * inp_cols + moveRight];
            sharedInput[threadIdx.x][threadIdx.y] = input[moveDown * inp_cols + moveRight];
            
            if (threadIdx.x < filter_rows) {
                // sharedInput[(threadIdx.x + blockDim.x) * (blockDim.y + filter_cols) + threadIdx.y ] = input[(moveDown + blockDim.x) * inp_cols + moveRight];
                sharedInput[threadIdx.x + blockDim.x][threadIdx.y] = input[(moveDown + blockDim.x) * inp_cols + moveRight];
            }
            if (threadIdx.y < filter_cols) {
                // sharedInput[threadIdx.x * (blockDim.y + filter_cols) + threadIdx.y + blockDim.y] = input[moveDown * inp_cols + moveRight + blockDim.y];
                sharedInput[threadIdx.x][threadIdx.y + blockDim.y] = input[moveDown * inp_cols + moveRight + blockDim.y];
            }
            if (threadIdx.x < filter_rows && threadIdx.y < filter_cols) {
                // sharedInput[(threadIdx.x + blockDim.x) * (blockDim.y + filter_cols) + threadIdx.y +blockDim.y] = input[(moveDown + blockDim.x) * inp_cols + moveRight + blockDim.y];
                sharedInput[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = input[(moveDown + blockDim.x) * inp_cols + moveRight + blockDim.y];
            }
        }
        __syncthreads();
        float sum = 0;
        for (int filterRow = 0; filterRow < filter_rows; filterRow++) {
            int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols; 
            if (inputIdx >= inp_rows * inp_cols) {
                continue;
            }
            for (int filterCol = 0; filterCol < filter_cols; filterCol++) {
                // sum += ((float) sharedInput[(threadIdx.x+filterRow) *(blockDim.y + filter_cols) + threadIdx.y + filterCol ]) * filterData[filterRow * filter_cols + filterCol];
                sum += ((float) sharedInput[threadIdx.x + filterRow][threadIdx.y + filterCol]) * filterData[filterRow * filter_cols + filterCol];                
            }
        }

        out[moveDown * inp_cols + moveRight] = (unsigned char) (sum);
    }

    __global__ void correlateSharedCol(unsigned char* input, unsigned char* out, 
            int inp_rows, int inp_cols, int filter_cols
    ) {
        // make filter size constant?, or use extern shared
        __shared__ unsigned char sharedInput[32][32 + 16];

        int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
        int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

        if (moveDown >= inp_rows) {
            return;
        }
        if (moveRight >= inp_cols) {
            return;
        }

        if (threadIdx.x < 32 && threadIdx.y < 32) {
            // sharedInput[threadIdx.x * (blockDim.y + filter_cols) + threadIdx.y] = input[moveDown * inp_cols + moveRight];
            sharedInput[threadIdx.x][threadIdx.y] = input[moveDown * inp_cols + moveRight];
            
            if (threadIdx.y < filter_cols) {
                // sharedInput[threadIdx.x * (blockDim.y + filter_cols) + threadIdx.y + blockDim.y] = input[moveDown * inp_cols + moveRight + blockDim.y];
                sharedInput[threadIdx.x][threadIdx.y + blockDim.y] = input[moveDown * inp_cols + moveRight + blockDim.y];
            }
        }
        
        __syncthreads();
        
        float sum = 0;
        for (int filterCol = 0; filterCol < filter_cols; filterCol++) {
            sum += ((float) sharedInput[threadIdx.x][threadIdx.y + filterCol]) * filterData[filterCol];
        }

        // printf("sum: %f\n", sum);
        out[moveDown * inp_cols + moveRight] = (unsigned char) (sum);
    }

    __global__ void correlateSharedRow(unsigned char* input, unsigned char* out, 
            int inp_rows, int inp_cols, int filter_rows
    ) {
        // make filter size constant?, or use extern shared
        __shared__ unsigned char sharedInput[32 + 16][32];

        int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
        int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

        if (moveDown >= inp_rows) {
            return;
        }
        if (moveRight >= inp_cols) {
            return;
        }

        if (threadIdx.x < 32 && threadIdx.y < 32) {
            // sharedInput[threadIdx.x * (blockDim.y + filter_cols) + threadIdx.y] = input[moveDown * inp_cols + moveRight];
            sharedInput[threadIdx.x][threadIdx.y] = input[moveDown * inp_cols + moveRight];
            
            if (threadIdx.x < filter_rows) {
                // sharedInput[(threadIdx.x + blockDim.x) * (blockDim.y + filter_cols) + threadIdx.y ] = input[(moveDown + blockDim.x) * inp_cols + moveRight];
                sharedInput[threadIdx.x + blockDim.x][threadIdx.y] = input[(moveDown + blockDim.x) * inp_cols + moveRight];
            }
        }
        
        __syncthreads();
        
        float sum = 0;
        for (int filterRow = 0; filterRow < filter_rows; filterRow++) {
            // filterData ("2d" size) is the same as correlateSharedRow? 
            sum += ((float) sharedInput[threadIdx.x + filterRow][threadIdx.y]) * filterData[filterRow];
        }

        // printf("sum: %f\n", sum);
        out[moveDown * inp_cols + moveRight] += (unsigned char) (sum);
    }
}