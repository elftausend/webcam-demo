use glow::*;


pub fn main2() {
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


        let width = 1920;
        let height = 1080;

        let (buf, vertex_array) = create_vertex_buffer(&gl);

        let program = gl.create_program().expect("Cannot create program");

        let (vertex_shader_source, fragment_shader_source) = (
            r#"in vec2 position;
            in vec2 tex_coords;
            out vec2 v_tex_coords;
            
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_tex_coords = tex_coords;
            }"#,
            r#"
            uniform sampler2D tex;
            in vec2 v_tex_coords;
            out vec4 f_color;
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
        gl.tex_storage_2d(texture.0.into(), 1, glow::RGBA8, width, height);

        // upload data to texture
        let data = vec![255u8; width as usize * height as usize * 4];
        gl.tex_sub_image_2d(
            texture.0.into(),
            0,
            0,
            0,
            width,
            height,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(&data)            
        );

        let uniform_location = gl.get_uniform_location(program, "tex").unwrap();
        // set 'texture' to sampler2D tex in fragment shader
        gl.uniform_1_i32(Some(uniform_location).as_ref(), 0);
    

        gl.use_program(Some(program));
        gl.clear_color(0.1, 0.2, 0.3, 1.0);

        // We handle events differently between targets
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Wait;
                match event {
                    Event::LoopDestroyed => {
                        return;
                    }
                    Event::MainEventsCleared => {
                        window.window().request_redraw();
                    }
                    Event::RedrawRequested(_) => {
                        gl.clear(glow::COLOR_BUFFER_BIT);

                        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                        
                        gl.draw_arrays(glow::TRIANGLES, 0, 4);
                        window.swap_buffers().unwrap();
                    }
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



unsafe fn create_vertex_buffer(gl: &glow::Context) -> (NativeBuffer, NativeVertexArray) {
    // This is a flat array of f32s that are to be interpreted as vec2s.
    let triangle_vertices = [-1., -1., 0., 0., -1., 1., 0., 1., 1., 1., 1., 1., 1., -1., 1., 0.];

    let triangle_vertices_u8: &[u8] = core::slice::from_raw_parts(
        triangle_vertices.as_ptr() as *const u8,
        triangle_vertices.len() * core::mem::size_of::<f32>(),
    );

    // We construct a buffer and upload the data
    let vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, triangle_vertices_u8, glow::STATIC_DRAW);

    // We now construct a vertex array to describe the format of the input buffer
    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(vao));
    gl.enable_vertex_attrib_array(0);
    
    
    //gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    // look in docs
    gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 16, 0);

    (vbo, vao)
}
