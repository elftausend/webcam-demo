use std::mem::size_of;

use custos::{
    cuda::{launch_kernel, CUDAPtr, api::CUstream},
    flag::AllocFlag,
    prelude::CUBuffer,
    CUDA,
};
use glow::*;

pub fn main2() {
    let device = CUDA::new(0).unwrap();
    unsafe {
        let (gl, shader_version, window, event_loop) = {
            let event_loop = glutin::event_loop::EventLoop::new();
            let window_builder = glutin::window::WindowBuilder::new()
                .with_title("Hello triangle!")
                .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
            let window = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(window_builder, &event_loop)
                .unwrap()
                .make_current()
                .unwrap();
            let gl =
                glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
            (gl, "#version 410", window, event_loop)
        };

        //gl.enable(glow::BLEND);
        gl.enable(DEBUG_OUTPUT);

        let width = 300;
        let height = 300;

        let program = gl.create_program().expect("Cannot create program");

        let (vertex_shader_source, fragment_shader_source) = (
            r#"
            in vec2 position;
            in vec2 tex_coords;
            out vec2 v_tex_coords;
            
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_tex_coords = tex_coords;
            }"#,
            r#"
            in vec2 v_tex_coords;
            out vec4 f_color;
            uniform sampler2D tex;
            void main() {
                f_color = texture(tex, v_tex_coords);
            }
            "#,
        );

        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_shader_source),
            (glow::FRAGMENT_SHADER, fragment_shader_source),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());

        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl
                .create_shader(*shader_type)
                .expect("Cannot create shader");
            gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("{}", gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        let texture = gl.create_texture().expect("Cannot create texture");
        gl.active_texture(TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, width, height);
        println!("{}", gl.get_error());
        /*let data = vec![120u8; width as usize * height as usize * 4];
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            Some(&data),
        );*/

        const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: u32 = 4;
        let mut cuda_resource: CUgraphicsResource = std::ptr::null_mut();
        cuGraphicsGLRegisterImage(
            &mut cuda_resource,
            texture.0.into(),
            glow::TEXTURE_2D,
            CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST,
        );
        

        cuGraphicsMapResources(1, &mut cuda_resource, device.stream().0);

        cuGraphicsSubResourceGetMappedArray

        cuda_driver_sys::cuSurfObjectCreate

        let ptr = std::ptr::addr_of!(cuda_resource) as u64;
        let mut buf: CUBuffer<u8> = CUBuffer {
            ptr: CUDAPtr {
                ptr,
                flag: AllocFlag::Wrapper,
                len: (width * height * 4) as usize,
                p: std::marker::PhantomData,
            },
            device: Some(&device),
            ident: None,
        };

        let src = r#"
            extern "C" __global__ void writeToSurface(cudaSurfaceObject_t target, int width, int height) {
                unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                //printf("%d %d w:%d h:%d \n", x, y, width, height);
                if (x < width && y < height) {
                    uchar4 data = make_uchar4(0xff, 0x00, 0x00, 0xff);
                    printf("writing to %d %d\n", x, y);
                    surf2Dwrite(data, target, x * sizeof(uchar4), y);
                }
            }
        "#;

        launch_kernel(
            &device,
            [256, 256, 1],
            [16, 16, 1],
            0,
            src,
            "writeToSurface",
            &[&buf, &(width as usize), &(height as usize)],
        )
        .unwrap();

        device.stream().sync().unwrap();

        //buf.write(&vec![120u8; width as usize * height as usize * 4]);

        // necessary for drawing!
        gl.generate_mipmap(glow::TEXTURE_2D);
        //
        gl.use_program(Some(program));

        let (buf, vertex_array, ebo) = create_vertex_buffer(&gl);

        // set 'texture' to sampler2D tex in fragment shader

        gl.uniform_1_i32(gl.get_uniform_location(program, "tex").as_ref(), 0);

        gl.bind_texture(glow::TEXTURE_2D, None);

        // We handle events differently between targets
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            event_loop.run(move |event, _, control_flow| {
                //*control_flow = ControlFlow::Wait;

                gl.clear_color(0.1, 0.2, 0.3, 0.3);

                //gl.enable(glow::TEXTURE_2D);

                gl.clear(glow::COLOR_BUFFER_BIT);

                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                gl.bind_vertex_array(Some(vertex_array));
                gl.use_program(Some(program));
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
                let uniform_location = gl.get_uniform_location(program, "tex");
                gl.uniform_1_i32(uniform_location.as_ref(), 0);

                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 5);
                //gl.draw_elements(glow::TRIANGLES, 6, glow::UNSIGNED_INT, 0);

                window.swap_buffers().unwrap();
                gl.use_program(None);

                match event {
                    Event::LoopDestroyed => {
                        return;
                    }
                    Event::MainEventsCleared => {
                        window.window().request_redraw();
                    }
                    Event::RedrawRequested(_) => {}
                    Event::WindowEvent { ref event, .. } => match event {
                        WindowEvent::Resized(physical_size) => {
                            window.resize(*physical_size);
                        }
                        WindowEvent::CloseRequested => {
                            gl.delete_program(program);
                            gl.delete_vertex_array(vertex_array);
                            *control_flow = ControlFlow::Exit
                        }
                        _ => (),
                    },
                    _ => (),
                }
            });
        }
    }
}


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUarray_st {
    _unused: [u8; 0],
}
pub type CUarray = *mut CUarray_st;

pub type CUsurfObject = ::std::os::raw::c_ulonglong;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphicsResource_st {
    _unused: [u8; 0],
}
pub type CUgraphicsResource = *mut CUgraphicsResource_st;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
}
pub use self::CUresourcetype_enum as CUresourcetype;

#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
    _bindgen_union_align: [u64; 16usize],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: cuda_driver_sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    pub flags: ::std::os::raw::c_uint,
}

pub type CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st;

extern "C" {
    fn cuGraphicsGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: u32,
        target: u32,
        Flags: u32,
    ) -> u32;

    pub fn cuGraphicsMapResources(
        count: ::std::os::raw::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> u32;

    pub fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::std::os::raw::c_uint,
        mipLevel: ::std::os::raw::c_uint,
    ) -> u32;

    pub fn cuSurfObjectCreate(
        pSurfObject: *mut CUsurfObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
    ) -> u32;
}

unsafe fn create_vertex_buffer(
    gl: &glow::Context,
) -> (NativeBuffer, NativeVertexArray, NativeBuffer) {
    let indices = [0u32, 2, 1, 0, 3, 2];

    // This is a flat array of f32s that are to be interpreted as vec2s.
    #[rustfmt::skip]
    let triangle_vertices = [
        -1f32, -1., 
        -1., 1., 
        1., 1., 
        1., -1.
    ];

    #[rustfmt::skip]
    let texcoords = [
        0f32, 0., 
        0., 1., 
        1., 1., 
        1., 0.
    ];

    #[rustfmt::skip]
    let triangle_vertices = [
        -0.5f32, -0.5,
        0.5, -0.5,
        -0.5, 0.5,
        0.5, 0.5,
    ];

    #[rustfmt::skip]
    let texcoords = [
        0f32, 0.,
        1., 0.,
        0., 1.,
        1., 1.,
    ];

    let triangle_vertices_u8: &[u8] = core::slice::from_raw_parts(
        triangle_vertices.as_ptr() as *const u8,
        triangle_vertices.len() * core::mem::size_of::<f32>(),
    );

    let tex_coords_u8 = core::slice::from_raw_parts(
        texcoords.as_ptr() as *const u8,
        texcoords.len() * core::mem::size_of::<f32>(),
    );

    let indices_u8 = core::slice::from_raw_parts(
        indices.as_ptr() as *const u8,
        indices.len() * core::mem::size_of::<u32>(),
    );

    let ebo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
    gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, indices_u8, glow::STATIC_DRAW);

    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(vao));

    let vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_size(
        glow::ARRAY_BUFFER,
        (16 * size_of::<f32>()) as i32,
        glow::STATIC_DRAW,
    );
    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, triangle_vertices_u8);
    gl.buffer_sub_data_u8_slice(
        glow::ARRAY_BUFFER,
        (8 * size_of::<f32>()) as i32,
        tex_coords_u8,
    );

    // We construct a buffer and upload the data
    //let vbo = gl.create_buffer().unwrap();
    //gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    //gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, triangle_vertices_u8, glow::STATIC_DRAW);

    // We now construct a vertex array to describe the format of the input buffer
    //let vao = gl.create_vertex_array().unwrap();
    //gl.bind_vertex_array(Some(vao));

    // gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    gl.enable_vertex_attrib_array(0);
    // gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 8, 0);
    //gl.vertex_attrib_pointer_i32(index, size, data_type, stride, offset)
    gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 8, 8 * size_of::<f32>() as i32);
    gl.enable_vertex_attrib_array(1);

    gl.bind_buffer(glow::ARRAY_BUFFER, None);
    gl.bind_vertex_array(None);
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

    (vbo, vao, ebo)
}
