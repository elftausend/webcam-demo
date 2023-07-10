mod cu_filter;
mod glium_webcam;
mod glow_webcam;
mod jpeg_decoder;
mod videotex;

fn main() {
    if true {
        glow_webcam::glow_webcam();
    } else {
        glium_webcam::glium_webcam().unwrap();
    }
}
