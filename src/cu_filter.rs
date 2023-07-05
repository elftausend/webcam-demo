use custos::{cuda::launch_kernel, prelude::{CUBuffer, Number}, CDatatype};

pub fn cu_padding<T: CDatatype>(
    input: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
    inp_rows: usize,
    inp_cols: usize,
    x_padding: usize,
    y_padding: usize,
) {
    let grid_x = ((inp_cols + x_padding * 2) as f32 / 16.).ceil() as u32;
    let grid_y = ((inp_rows + y_padding * 2) as f32 / 16.).ceil() as u32;

    let src = format!(
        r#"
        extern "C" __global__ void addPadding({dtype}* input, {dtype}* out, int inpRows, int inpCols, int xPadding, int yPadding) {{
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            int col = blockDim.y * blockIdx.y + threadIdx.y;

            if (row >= inpRows || col >= inpCols) {{
                return;
            }}

            out[yPadding * (inpRows + xPadding * 2) + row * (inpCols + 2 * xPadding) + col] = input[row * inpCols + col];
        }}
    "#,
        dtype = T::as_c_type_str()
    );
    launch_kernel(
        input.device(),
        [grid_x, grid_y, 1],
        [16, 16, 1],
        0,
        &src,
        "addPadding",
        &[input, out, &inp_rows, &inp_cols, &x_padding, &y_padding],
    )
    .unwrap();
}

pub fn correlate_cu<T: CDatatype>(
    input: &CUBuffer<T>,
    filter: &CUBuffer<T>,
    out: &mut CUBuffer<T>,
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
        extern "C" __global__ void correlate2({dtype}* input, {dtype}* filter, {dtype}* out, int inp_rows, int inp_cols, int filter_rows, int filter_cols) {{
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
            {dtype} sum = 0;
            for (int filterRow = 0; filterRow < filter_rows; filterRow++) {{
                int inputIdx = moveDown * inp_cols + moveRight + filterRow * inp_cols;  
                for (int filterCol = 0; filterCol < filter_cols; filterCol++) {{
                    sum += input[inputIdx + filterCol] * filter[filterRow * filter_cols + filterCol];
                }}
            }}
            out[moveDown * outCols + moveRight] = sum;
        }}
    "#,
        dtype = T::as_c_type_str()
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
                    sum += *value * kernel_row[i];
                }
            }
            // y * final_cols + x
            out[y * out_cols + x] = sum;
        }
    }
}