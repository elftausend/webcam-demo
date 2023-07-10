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

    __global__ void correlateWithTex(cudaTextureObject_t inputTexture, float* filter, unsigned char* out, 
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
                //float4 color = tex2D<float4>(inputTexture, 0, 0);
                float filterVal = filter[filterRow * filter_cols + filterCol];

                float4 color = tex2D<float4>(inputTexture, (moveRight + filterCol) * sizeof(uchar4), inp_rows -1- (moveDown + filterRow));
                sum.x += color.x * filterVal;
                sum.y += color.y * filterVal;
                sum.z += color.z * filterVal;

                //print every color
                //printf("R: %f, G: %f, B: %f, A: %f\n", color.x, color.y, color.z, color.w);
                //sum += (((float) input[inputIdx + filterCol]))  * filter[filterRow * filter_cols + filterCol];
            }
        }
        
        //out[moveDown * inp_cols + moveRight] = (unsigned char) (sum);
    }
}