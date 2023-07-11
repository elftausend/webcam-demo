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

    let func = fn_cache(texture.device(), CUDA_SOURCE, "correlateWithTexShared").unwrap();


    launch_kernel_with_fn(
        texture.device(),
        &func,
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
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
