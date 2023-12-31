use std::ops::Mul;

use custos::{
    cuda::launch_kernel,
    prelude::{CUBuffer, Number},
    CDatatype,
};

pub fn correlate_cu(
    input: &CUBuffer<u8>,
    filter: &CUBuffer<f32>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let (out_rows, out_cols) = (inp_rows - filter_rows + 1, inp_cols - filter_cols + 1);

    const THREADS: u32 = 8;

    // THREADS
    let grid_x = (out_cols as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (out_rows as f32 / THREADS as f32).ceil() as u32;

    let src = format!(
        r#"
        extern "C" __global__ void correlate2(unsigned char* input, float* filter, unsigned char* out, int inp_rows, int inp_cols, int filter_rows, int filter_cols) {{
            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

            int outRows = inp_rows - filter_rows + 1;
            int outCols = inp_cols - filter_cols + 1;
            if (moveDown >= outRows) {{
                return;
            }} 
            if (moveRight >= outCols) {{
                return;
            }}
            float sum = 0;
            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols;  
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += (((float) input[inputIdx + filterCol]) / 255.0f)  * filter[filterRow * filter_cols + filterCol];
                }}
            }}

            out[moveDown * outCols + moveRight] = (unsigned char) (sum * 255.0f);
        }}
    "#,
    );

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        &src,
        "correlate2",
        &[
            input,
            filter,
            out,
            &inp_rows,
            &inp_cols,
            &filter_rows,
            &filter_cols,
        ],
    )
    .unwrap();
}

pub fn correlate_cu_out_req_pad(
    input: &CUBuffer<u8>,
    filter: &CUBuffer<f32>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols: usize = inp_cols + x_padding * 2;

    const THREADS: u32 = 8;

    //let max_down = padded_rows - filter_rows - y_padding;
    let max_down = inp_rows;
    //let max_right = padded_cols - filter_cols - x_padding;
    let max_right = inp_cols;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (padded_cols as f32 / THREADS as f32).ceil() as u32;

    let src = format!(
        r#"
        extern "C" __global__ void correlateWithTex(
            unsigned char* input, float* filter, unsigned char* out, 
            int inp_rows, int inp_cols, int filter_rows, 
            int filter_cols, int maxDown, int maxRight, int paddedCols
        ) {{
            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

            if (moveDown >= maxDown) {{
                return;
            }} 
            if (moveRight >= maxRight) {{
                return;
            }}
            float sum = 0;
            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * paddedCols + moveRight + filterRow * paddedCols; 
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += (((float) input[inputIdx + filterCol]))  * filter[filterRow * filter_cols + filterCol] ;
                }}
            }}

            out[moveDown * inp_cols + moveRight] = (unsigned char) (sum);
        }}
    "#,
    );

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        &src,
        "correlate2",
        &[
            input,
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

pub fn correlate_cu_out_auto_pad(
    input: &CUBuffer<u8>,
    filter: &CUBuffer<f32>,
    out: &mut CUBuffer<u8>,
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let y_padding = filter_rows - 1;

    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols = inp_cols;

    const THREADS: u32 = 32;

    //let max_down = padded_rows - filter_rows - y_padding;
    let max_down = inp_rows;
    //let max_right = padded_cols - filter_cols - x_padding;
    let max_right = inp_cols;

    // THREADS
    let grid_x = (padded_rows as f32 / THREADS as f32).ceil() as u32;
    let grid_y = (padded_cols as f32 / THREADS as f32).ceil() as u32;

    let src = format!(
        r#"
        extern "C" __global__ void correlateAutoPad(
            unsigned char* input, float* filter, unsigned char* out, 
            int inp_rows, int inp_cols, int filter_rows, 
            int filter_cols, int maxDown, int maxRight, int paddedCols
        ) {{
            int moveDown = blockDim.x * blockIdx.x + threadIdx.x;
            int moveRight = blockDim.y * blockIdx.y + threadIdx.y;

            if (moveDown >= maxDown) {{
                return;
            }} 
            if (moveRight >= maxRight) {{
                return;
            }}
            float sum = 0;
            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * paddedCols + moveRight + filterRow * paddedCols; 
                if (inputIdx >= inp_rows * inp_cols) {{
                    continue;
                }}

                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += (((float) input[inputIdx + filterCol]))  * filter[filterRow * filter_cols + filterCol] ;
                }}
            }}

            out[moveDown * inp_cols + moveRight] = (unsigned char) (sum);
        }}
    "#,
    );

    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [THREADS, THREADS, 1],
        0,
        &src,
        "correlateAutoPad",
        &[
            input,
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

pub fn correlate_valid_mut<T: Number>(
    lhs_slice: &[T],
    lhs_dims: (usize, usize),
    kernel_slice: &[T],
    kernel_dims: (usize, usize),
    out: &mut [T],
) {
    let (lhs_rows, lhs_cols) = lhs_dims;
    let (kernel_rows, kernel_cols) = kernel_dims;

    let (out_rows, out_cols) = (lhs_rows - kernel_rows + 1, lhs_cols - kernel_cols + 1);

    //loop for row-axis (y)
    //moves multiplication 1 down
    for y in 0..out_rows {
        //loop for col-axis (x)
        //moves multiplication 1 to the right
        for x in 0..out_cols {
            let mut sum = T::default();
            //repeat kernel rows times to use move through all kernel rows
            for idx in 0..kernel_rows {
                let index = idx * lhs_cols + x + y * lhs_cols;
                let lhs_kernel_row = &lhs_slice[index..index + kernel_cols];

                let index = idx * kernel_cols;
                let kernel_row = &kernel_slice[index..index + kernel_cols];

                for (i, value) in lhs_kernel_row.iter().enumerate() {
                    sum += (*value * kernel_row[i]) / T::from_u64(16);
                }
            }
            // y * final_cols + x
            out[y * out_cols + x] = sum;
        }
    }
}

pub fn correlate_fully<T: Number + Mul<U, Output = T>, U: Number>(
    inputs: &[T],
    filter: &[U],
    out: &mut [T],
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_inputs = add_padding(inputs, inp_rows, inp_cols, x_padding, y_padding);
    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols = inp_cols + x_padding * 2;

    // attention: leaves the last padded row, col out
    for move_down in 0..=padded_rows - filter_rows - y_padding {
        for move_right in 0..=padded_cols - filter_cols - x_padding {
            let mut sum = T::default();
            for idx in 0..filter_rows {
                let filter_idx = idx * filter_cols;
                let filter_row = &filter[filter_idx..filter_idx + filter_cols];

                let input_idx = move_down * padded_cols + move_right + idx * padded_cols;
                let input_row = &padded_inputs[input_idx..input_idx + filter_cols];

                for (filter_row, input_row) in filter_row.iter().zip(input_row) {
                    sum += *input_row * *filter_row;
                }
            }
            out[move_down * inp_cols + move_right] = sum;
        }
    }
}

pub fn correlate_fully_u8(
    inputs: &[u8],
    out: &mut [u8],
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_inputs = add_padding(inputs, inp_rows, inp_cols, x_padding, y_padding);
    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols = inp_cols + x_padding * 2;

    // attention: leaves the last padded row, col out
    for move_down in 0..=padded_rows - filter_rows - y_padding {
        for move_right in 0..=padded_cols - filter_cols - x_padding {
            let mut sum = u8::default();
            for idx in 0..filter_rows {
                let filter_idx = idx * filter_cols;

                let input_idx = move_down * padded_cols + move_right + idx * padded_cols;
                let input_row = &padded_inputs[input_idx..input_idx + filter_cols];

                for input_row in input_row {
                    sum += *input_row / 16;
                }
            }
            out[move_down * inp_cols + move_right] = sum;
        }
    }
}

pub fn add_padding<T: Number>(
    inputs: &[T],
    inp_rows: usize,
    inp_cols: usize,
    x_padding: usize,
    y_padding: usize,
) -> Vec<T> {
    let mut padded_inputs =
        vec![T::zero(); (inp_rows + y_padding * 2) * (inp_cols + x_padding * 2)];

    for inp_row in 0..inp_rows {
        for inp_col in 0..inp_cols {
            padded_inputs[y_padding * (inp_cols + 2 * x_padding)
                + x_padding
                + inp_row * (inp_cols + 2 * x_padding)
                + inp_col] = inputs[inp_row * inp_cols + inp_col];
        }
    }
    padded_inputs
}
