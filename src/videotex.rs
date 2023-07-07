use custos::{cuda::launch_kernel, prelude::CUBuffer};

pub fn fill_cuda_surface(
    to_fill: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
    r: u8,
    g: u8,
    b: u8,
) -> custos::Result<()> {
    let src = r#"
    extern "C" __global__ void writeToSurface(cudaSurfaceObject_t target, int width, int height, char r, char g, char b) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        if (x < width && y < height) {
            uchar4 data = make_uchar4(r, g, b, 0xff);
            surf2Dwrite(data, target, x * sizeof(uchar4), y);
        }
    }
"#;

    launch_kernel(
        to_fill.device(),
        [256, 256, 1],
        [16, 16, 1],
        0,
        src,
        "writeToSurface",
        &[to_fill, &width, &height, &r, &g, &b],
    )
}

pub fn interleave_rgb(
    target: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    let src = r#"
        extern "C" __global__ void interleaveRGB(cudaSurfaceObject_t target, int width, int height,
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
    "#;

    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        src,
        "interleaveRGB",
        &[target, &width, &height, red, green, blue],
    )
}
