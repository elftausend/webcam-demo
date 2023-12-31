extern "C"{
    
    __global__ void writeToSurface(cudaSurfaceObject_t target, int width, int height, char r, char g, char b) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < width && y < height) {
            uchar4 data = make_uchar4(r, g, b, 0xff);
            surf2Dwrite(data, target, x * sizeof(uchar4), y);
        }
    }

    __global__ void interleaveRGB(cudaSurfaceObject_t target, int width, int height,
            unsigned char *R, unsigned char *G, unsigned char *B )
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) {       
            unsigned char valR = R[y * width + x]; 
            unsigned char valG = G[y * width + x]; 
            unsigned char valB = B[y * width + x]; 
            uchar4 data = make_uchar4(valR, valG, valB, 0xff);
            surf2Dwrite(data, target, x * sizeof(uchar4), height -1- y);
        }
    }
    __constant__ float filterData[64*64];

    __global__ void correlateWithTex(cudaTextureObject_t inputTexture, cudaSurfaceObject_t out, 
            int inp_rows, int inp_cols, int filter_rows, 
            int filter_cols, int maxDown, int maxRight, int paddedCols
    ) {
        int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
        int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

        if (moveDown >= maxDown) {
            return;
        } 
        if (moveRight >= maxRight) {
            return;
        }  

        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int filterRow = 0; filterRow < filter_rows; filterRow++) {
            int inputIdx = moveDown * paddedCols + moveRight + filterRow * paddedCols; 

            for (int filterCol = 0; filterCol < filter_cols; filterCol++) {
                float filterVal = filterData[filterRow * filter_cols + filterCol];
                // float filterVal = 1.0 / (float) (filter_cols * filter_rows);

                float4 color = tex2D<float4>(inputTexture, (moveRight + filterCol), inp_rows -1- (moveDown + filterRow));
                sum.x += color.x * filterVal;
                sum.y += color.y * filterVal;
                sum.z += color.z * filterVal;
            }
        }

        uchar4 data = make_uchar4((unsigned char) (sum.x * 255.0f), (unsigned char) (sum.y * 255.0f), (unsigned char) (sum.z * 255.0f), 0xff);

        //uchar4 data = make_uchar4(0, 255, 0, 255);

        //printf("R: %d, G: %d, B: %d, A: %d\n", data.x, data.y, data.z, data.w);
        surf2Dwrite(data, out, moveRight * sizeof(uchar4), inp_rows -1- moveDown);
    }

    __global__ void correlateWithTexShared(cudaTextureObject_t inputTexture, cudaSurfaceObject_t out, 
            int inp_rows, int inp_cols, int filter_rows, int filter_cols
    ) {

        __shared__ float4 sharedInput[8 + 3][32 + 3];
        // extern __shared__ float4 sharedInput[];
        int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
        int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

        if (moveDown >= inp_rows) {
            return;
        } 
        if (moveRight >= inp_cols) {
            return;
        } 


        //if (threadIdx.x < 32 && threadIdx.y < 32) {
            //sharedInput[threadIdx.y * (blockDim.y + filter_cols) + threadIdx.x] = tex2D<float4>(inputTexture, moveRight, moveDown);
            sharedInput[threadIdx.y][threadIdx.x] = tex2D<float4>(inputTexture, moveRight, moveDown);
            
            if (threadIdx.x < filter_rows) {
                //sharedInput[threadIdx.y * (blockDim.y + filter_cols) + threadIdx.x + blockDim.x] = tex2D<float4>(inputTexture, moveRight, moveDown + blockDim.x);
                sharedInput[threadIdx.y][threadIdx.x + blockDim.x] = tex2D<float4>(inputTexture, moveRight, moveDown + blockDim.x);
            }
            if (threadIdx.y < filter_cols) {
                // sharedInput[(threadIdx.y + blockDim.y) * (blockDim.y + filter_cols) + threadIdx.x] = tex2D<float4>(inputTexture, moveRight + blockDim.y, moveDown);
                sharedInput[threadIdx.y + blockDim.y][threadIdx.x] = tex2D<float4>(inputTexture, moveRight + blockDim.y, moveDown);
            }
            if (threadIdx.x < filter_rows && threadIdx.y < filter_cols) {
                // sharedInput[(threadIdx.y + blockDim.y) * (blockDim.y + filter_cols) + threadIdx.x + blockDim.x] = tex2D<float4>(inputTexture, moveRight + blockDim.y, moveDown + blockDim.x);
                sharedInput[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = tex2D<float4>(inputTexture, moveRight + blockDim.y, moveDown + blockDim.x);
            }
        //}

        __syncthreads();

        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int filterRow = 0; filterRow < filter_rows; filterRow++) {
            int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols;
 
            for (int filterCol = 0; filterCol < filter_cols; filterCol++) {
                float filterVal = filterData[filterRow * filter_cols + filterCol];
                //float filterVal = 1.0 / (float) (filter_cols * filter_rows);

                //float4 color = sharedInput[(threadIdx.y + filterCol) * (blockDim.y + filter_cols) + threadIdx.x + filterRow];
                float4 color = sharedInput[threadIdx.y + filterCol][threadIdx.x + filterRow];
                // float4 color = tex2D<float4>(inputTexture, (moveRight + filterCol), (moveDown + filterRow));

                sum.x += color.x * filterVal;
                sum.y += color.y * filterVal;
                sum.z += color.z * filterVal;
            }
        }

        uchar4 data = make_uchar4((unsigned char) (sum.x * 255.0f), (unsigned char) (sum.y * 255.0f), (unsigned char) (sum.z * 255.0f), 0xff);

        //uchar4 data = make_uchar4(0, 255, 0, 255);

        //printf("R: %d, G: %d, B: %d, A: %d\n", data.x, data.y, data.z, data.w);
        surf2Dwrite(data, out, moveRight * sizeof(uchar4), moveDown);
    }
}