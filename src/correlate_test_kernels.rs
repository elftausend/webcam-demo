use custos::{prelude::CUBuffer, cuda::launch_kernel};

pub const CUDA_COR_SOURCE: &'static str = include_str!("./correlate_test_kernels.cu");

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
        CUDA_COR_SOURCE,
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

pub fn correlate_shared_col(
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

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        CUDA_COR_SOURCE,
        "correlateSharedCol",
        &[
            input,
            out,
            &inp_rows,
            &inp_cols,
            &filter_cols,
        ],
    )
    .unwrap();
}

pub fn correlate_shared_row(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
) {
    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;

    const THREADS: u32 = 32;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (inp_cols as f32 / THREADS as f32).ceil() as u32;

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        CUDA_COR_SOURCE,
        "correlateSharedRow",
        &[
            input,
            out,
            &inp_rows,
            &inp_cols,
            &filter_rows,
        ],
    )
    .unwrap();
}


#[cfg(test)]
mod tests {
    use custos::{buf, Buffer};

    use crate::{get_constant_memory, cu_filter::correlate_cu_out_auto_pad, correlate_test_kernels::{correlate_shared_col, correlate_shared_row}};

    use super::{CUDA_COR_SOURCE, correlate_shared};

    #[test]
    fn test_correlate_cu_tex_shared() {
        let height = 1080;
        let width = 1920;
        // let input = buf![128; height * width].to_gpu();
        let input = (0..height*width).into_iter().map(|_| fastrand::u8(0..255)).collect::<Buffer<u8>>().to_gpu();
        
        let filter_rows = 16;
        let filter_cols = 16;
        let filter = buf![1. / (filter_rows * filter_cols) as f32; filter_rows * filter_cols];

        let mut filter_data = get_constant_memory::<f32>(input.device(), CUDA_COR_SOURCE, "correlateShared", "filterData");
        filter_data.write(&filter);

        let mut out = buf![0; height * width].to_gpu();
        
        correlate_shared(&input, &mut out, height, width, filter_rows, filter_cols);

        input.device().stream().sync().unwrap();

        let start = std::time::Instant::now();
        correlate_shared(&input, &mut out, height, width, filter_rows, filter_cols);
        input.device().stream().sync().unwrap();
        println!("shared {:?}", start.elapsed());
        // correlate_cu_out_auto_pad(&input, &filter.to_cuda(), &mut output_auto_pad, height, width, filter_rows, filter_cols);



        let mut output_auto_pad = buf![0; height * width].to_gpu();
        let filter_cu = filter.to_cuda();
        correlate_cu_out_auto_pad(&input, &filter_cu, &mut output_auto_pad, height, width, filter_rows, filter_cols);

        input.device().stream().sync().unwrap();

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

    #[test]
    fn test_correlate_cu_tex_row_col() {
        let height = 1080;
        let width = 1920;
        // let input = buf![128; height * width].to_gpu();
        let input = (0..height*width).into_iter().map(|_| fastrand::u8(0..255)).collect::<Buffer<u8>>().to_gpu();
        
        let filter_rows = 16;
        let filter_cols = 16;
        let filter = buf![1. / (filter_rows * filter_cols) as f32; filter_rows * filter_cols];

        let mut filter_data = get_constant_memory::<f32>(input.device(), CUDA_COR_SOURCE, "correlateSharedCol", "filterData");
        filter_data.write(&filter);

        // 'CUDA_SOURCES' is compiled again therefore, a new module is created, which is why this needs to be called again for correlateSharedRow
        let mut filter_data = get_constant_memory::<f32>(input.device(), CUDA_COR_SOURCE, "correlateSharedRow", "filterData");
        filter_data.write(&filter);

        let mut out = buf![0; height * width].to_gpu();
        
        correlate_shared_col(&input, &mut out, height, width, filter_rows, filter_cols);
        input.device().stream().sync().unwrap();
        correlate_shared_row(&input, &mut out, height, width, filter_rows);

        input.device().stream().sync().unwrap();

        let start = std::time::Instant::now();
        correlate_shared_col(&input, &mut out, height, width, filter_rows, filter_cols);
        input.device().stream().sync().unwrap();
        correlate_shared_row(&input, &mut out, height, width, filter_rows);
        input.device().stream().sync().unwrap();
        println!("shared {:?}", start.elapsed());
        // correlate_cu_out_auto_pad(&input, &filter.to_cuda(), &mut output_auto_pad, height, width, filter_rows, filter_cols);



        let mut output_auto_pad = buf![0; height * width].to_gpu();
        let filter_cu = filter.to_cuda();
        correlate_cu_out_auto_pad(&input, &filter_cu, &mut output_auto_pad, height, width, filter_rows, filter_cols);

        input.device().stream().sync().unwrap();

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