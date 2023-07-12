use std::{ffi::CString, mem::size_of};

use custos::{
    cuda::{fn_cache, launch_kernel, launch_kernel_with_fn},
    prelude::CUBuffer,
};

pub const CUDA_SOURCE: &'static str = include_str!("./videotex.cu");

pub fn fill_cuda_surface(
    to_fill: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
    r: u8,
    g: u8,
    b: u8,
) -> custos::Result<()> {
    launch_kernel(
        to_fill.device(),
        [256, 256, 1],
        [16, 16, 1],
        0,
        CUDA_SOURCE,
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
    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "interleaveRGB",
        &[target, &width, &height, red, green, blue],
    )
}

pub fn correlate_cu_tex(
    texture: &CUBuffer<u8>,
    filter: &CUBuffer<f32>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    const THREADS: u32 = 8;

    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols: usize = inp_cols + x_padding * 2;

    //let max_down = padded_rows - filter_rows - y_padding;
    let max_down = inp_rows;
    //let max_right = padded_cols - filter_cols - x_padding;
    let max_right = inp_cols;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (padded_cols as f32 / THREADS as f32).ceil() as u32;
    launch_kernel(
        texture.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        CUDA_SOURCE,
        "correlateWithTex",
        &[
            texture,
            filter,
            out,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
            &max_down,
            &max_right,
            &padded_cols,
        ],
    )
    .unwrap();
}

pub fn correlate_cu_tex_shared(
    texture: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    const THREADS: u32 = 32;

    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;

    let func = fn_cache(texture.device(), CUDA_SOURCE, "correlateWithTexShared").unwrap();

    launch_kernel_with_fn(
        texture.device(),
        &func,
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,//(THREADS + filter_rows as u32) * (THREADS + filter_cols as u32) * size_of::<f32>() as u32 * 4,
        &[
            texture,
            out,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
        ],
    )
    .unwrap();
}


pub fn correlate_shared(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;

    const THREADS: u32 = 32;

    //let max_down = padded_rows - filter_rows - y_padding;
    let max_down = inp_rows;
    //let max_right = padded_cols - filter_cols - x_padding;
    let max_right = inp_cols;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        CUDA_SOURCE,
        "correlateShared",
        &[
            input,
            out,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
            &max_down,
            &max_right,
        ],
    )
    .unwrap();
}

// move to custos, as well as the other cu functions
// mind the todo in the fn_cache function (inefficient module stuff)
extern "C" {
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut custos::cuda::CUdeviceptr,
        bytes: *mut usize,
        hmod: custos::cuda::api::CUmodule,
        name: *const std::ffi::c_char,
    ) -> u32;
}

#[cfg(test)]
mod tests {
    use custos::{buf, Buffer};

    use crate::{get_constant_memory, cu_filter::correlate_cu_out_auto_pad};

    use super::{CUDA_SOURCE, correlate_shared};

    #[test]
    fn test_correlate_cu_tex_shared() {
        let height = 1080;
        let width = 1920;
        // let input = buf![128; height * width].to_gpu();
        let input = (0..height*width).into_iter().map(|_| fastrand::u8(0..255)).collect::<Buffer<u8>>().to_gpu();
        
        let filter_rows = 16;
        let filter_cols = 16;
        let filter = buf![1. / (filter_rows * filter_cols) as f32; filter_rows * filter_cols];

        let mut filter_data = get_constant_memory::<f32>(input.device(), CUDA_SOURCE, "correlateShared", "filterData");
        filter_data.write(&filter);

        let mut out = buf![0; height * width].to_gpu();
        
        correlate_shared(&input, &mut out, height, width, filter_rows, filter_cols);

        let start = std::time::Instant::now();
        correlate_shared(&input, &mut out, height, width, filter_rows, filter_cols);
        input.device().stream().sync().unwrap();
        println!("shared {:?}", start.elapsed());
        // correlate_cu_out_auto_pad(&input, &filter.to_cuda(), &mut output_auto_pad, height, width, filter_rows, filter_cols);



        let mut output_auto_pad = buf![0; height * width].to_gpu();
        let filter_cu = filter.to_cuda();
        correlate_cu_out_auto_pad(&input, &filter_cu, &mut output_auto_pad, height, width, filter_rows, filter_cols);

        let start = std::time::Instant::now();
        correlate_cu_out_auto_pad(&input, &filter_cu, &mut output_auto_pad, height, width, filter_rows, filter_cols);
        input.device().stream().sync().unwrap();
        println!("auto pad {:?}", start.elapsed());

        for (op, o) in output_auto_pad.read().iter().zip(out.read().iter()) {
            if ((*op as f32 - *o as f32)).abs() > 3. {
                panic!("{} {}", op, o);
            }
        }
        
    }
}