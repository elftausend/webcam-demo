use custos::{cuda::launch_kernel, prelude::CUBuffer, CDatatype};

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
