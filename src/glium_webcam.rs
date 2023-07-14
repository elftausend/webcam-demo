use std::io;
use std::sync::{mpsc, RwLock};
use std::thread;
use std::time::Instant;

use custos::buf;
use glium::index::PrimitiveType;
use glium::{glutin, Surface};
use glium::{implement_vertex, program, uniform};

use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::{Format, FourCC};

use crate::cu_filter::{correlate_cu, correlate_fully, correlate_fully_u8, correlate_valid_mut};
use crate::{jpeg_decoder, Args};

// https://github.com/raymanfx/libv4l-rs/blob/master/examples/glium.rs

pub fn glium_webcam(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let width = args.width;
    let height = args.height;

    let path = "/dev/video0";
    println!("Using device: {}\n", path);

    // Allocate 4 buffers by default
    let buffer_count = 4;

    let mut format: Format;
    let params: Parameters;

    let dev = RwLock::new(Device::with_path(path)?);
    {
        let dev = dev.write().unwrap();
        //format = dev.format()?;
        format = Format::new(width as u32, height as u32, FourCC::new(b"RGB3"));
        println!("format: {format}");
        params = dev.params()?;

        // try RGB3 first
        format.fourcc = FourCC::new(b"RGB3");
        format = dev.set_format(&format)?;

        if format.fourcc != FourCC::new(b"RGB3") {
            // fallback to Motion-JPEG
            format.fourcc = FourCC::new(b"MJPG");
            format = dev.set_format(&format)?;

            if format.fourcc != FourCC::new(b"MJPG") {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    "neither RGB3 nor MJPG supported by the device, but required by this example!",
                ))?;
            }
        }
    }

    println!("Active format:\n{}", format);
    println!("Active parameters:\n{}", params);

    // Setup the GL display stuff
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    // building the vertex buffer, which contains all the vertices that we will draw
    let vertex_buffer = {
        #[derive(Copy, Clone)]
        struct Vertex {
            position: [f32; 2],
            tex_coords: [f32; 2],
        }

        implement_vertex!(Vertex, position, tex_coords);

        glium::VertexBuffer::new(
            &display,
            &[
                Vertex {
                    position: [-1.0, -1.0],
                    tex_coords: [0.0, 0.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    tex_coords: [0.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                    tex_coords: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                    tex_coords: [1.0, 0.0],
                },
            ],
        )
        .unwrap()
    };

    // building the index buffer
    let index_buffer =
        glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip, &[1u16, 2, 0, 3]).unwrap();

    // compiling shaders and linking them together
    let program = program!(&display,
        140 => {
            vertex: "
            #version 140
            uniform mat4 matrix;
            in vec2 position;
            in vec2 tex_coords;
            out vec2 v_tex_coords;
            void main() {
                gl_Position = matrix * vec4(position, 0.0, 1.0);
                v_tex_coords = tex_coords;
            }
        ",

            fragment: "
            #version 140
            uniform sampler2D tex;
            in vec2 v_tex_coords;
            out vec4 f_color;
            void main() {
                f_color = texture(tex, v_tex_coords);
            }
        "
        },
    )
    .unwrap();

    let (tx, rx) = mpsc::channel();

    // light sensitivity => u8 overflow
    let filter_rows = 5;
    let filter_cols = 5;

    thread::spawn(move || {
        let dev = dev.write().unwrap();

        let mut decoder: jpeg_decoder::JpegDecoder<'_> =
            unsafe { jpeg_decoder::JpegDecoder::new(width as usize, height as usize).unwrap() };

        let filter = buf![1; filter_rows * filter_cols].to_gpu();
        let mut filtered = buf![0; width as usize * height as usize * 3].to_gpu();

        // Setup a buffer stream
        let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, buffer_count).unwrap();

        let mut out = vec![0; width as usize * height as usize * 3];
        loop {
            let (buf, _) = stream.next().unwrap();
            let data = match &format.fourcc.repr {
                b"RGB3" => buf.to_vec(),
                b"MJPG" => {
                    // Decode the JPEG frame to RGB

                    //let mut input = buf![10; width as usize * height as usize * 3].to_gpu();

                    // directly write into 2d gl texture

                    unsafe { decoder.decode_rgb(buf) }.unwrap();
                    let mut data = Vec::with_capacity(height as usize * 3 * width as usize);
                    let res = decoder
                        .channels
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|x| x.read_to_vec())
                        .collect::<Vec<Vec<u8>>>();

                    let mut channel0 = vec![0; height as usize * width as usize];
                    correlate_fully_u8(
                        &res[0],
                        &mut channel0,
                        height as usize,
                        width as usize,
                        filter_rows,
                        filter_cols,
                    );
                    //correlate_valid_mut(&res[0], (height as usize, width as usize), &filter.read(), (filter_rows, filter_cols), &mut channel0);

                    let mut channel1 = vec![0; height as usize * width as usize];
                    correlate_fully_u8(
                        &res[1],
                        &mut channel1,
                        height as usize,
                        width as usize,
                        filter_rows,
                        filter_cols,
                    );
                    //correlate_valid_mut(&res[1], (height as usize, width as usize), &filter.read(), (filter_rows, filter_cols), &mut channel1);

                    let mut channel2 = vec![0; height as usize * width as usize];
                    correlate_fully_u8(
                        &res[2],
                        &mut channel2,
                        height as usize,
                        width as usize,
                        filter_rows,
                        filter_cols,
                    );
                    //correlate_valid_mut(&res[2], (height as usize, width as usize), &filter.read(), (filter_rows, filter_cols), &mut channel2);

                    for (i, _) in channel0.iter().enumerate() {
                        data.push(channel0[i]);
                        data.push(channel1[i]);
                        data.push(channel2[i]);
                    }

                    data
                    //unsafe { decoder.decode_rgbi(buf) }.unwrap();
                    //correlate_cu(&decoder.channel, &filter, &mut filtered, height as usize * 3, width as usize, filter_rows, filter_cols);

                    //correlate_fully(&decoder.channel.as_ref().unwrap().read(), &filter.read(), &mut out, height as usize * 3, width as usize, filter_rows, filter_cols);
                    //correlate_valid_mut(&decoder.channel.read(), (height as usize * 3, width as usize), &filter.read(), (filter_rows, filter_cols), &mut out);

                    //correlate_cu(&input, &filter, &mut filtered, height as usize, width as usize, filter_rows, filter_cols);

                    //decoder.channel.as_mut().unwrap().read()
                    //filtered.read()
                    //out.clone()
                }
                _ => panic!("invalid buffer pixelformat"),
            };
            tx.send(data).unwrap();
        }
    });

    event_loop.run(move |event, _, control_flow| {
        let t0 = Instant::now();
        let data = rx.recv().unwrap();
        let t1 = Instant::now();

        let image =
            glium::texture::RawImage2d::from_raw_rgb_reversed(&data, (format.width, format.height));
        let opengl_texture = glium::texture::Texture2d::new(&display, image).unwrap();

        // building the uniforms
        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]
            ],
            tex: &opengl_texture
        };

        // drawing a frame
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 0.0);
        target
            .draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniforms,
                &Default::default(),
            )
            .unwrap();

        target.finish().unwrap();

        // polling and handling the events received by the window
        if let glutin::event::Event::WindowEvent {
            event: glutin::event::WindowEvent::CloseRequested,
            ..
        } = event
        {
            *control_flow = glutin::event_loop::ControlFlow::Exit;
        }

        print!(
            "\rms: {}\t (buffer) + {}\t (UI)",
            t1.duration_since(t0).as_millis(),
            t0.elapsed().as_millis()
        );
    });
}

#[test]
fn test_ptr() {
    let val = 432;
    let x = val as *mut i32;

    let res = std::ptr::addr_of!(x) as u64;
    println!("res: {res}");
}
