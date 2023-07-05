use std::io::Write;
use std::fs::File;

fn main() {
    // Open the video device (webcam)
    let mut device = v4l::Device::open(0).expect("Failed to open video device");
    
    // Set the desired video format
    let format = v4l::Format {
        width: 640,
        height: 480,
        fourcc: v4l::FourCC::new(b"MJPEG"), // You can specify the desired format here
        ..Default::default()
    };
    device.set_format(&format).expect("Failed to set video format");
    
    // Start the video stream
    device.start().expect("Failed to start video stream");
    
    // Capture frames
    let mut frame_count = 0;
    while frame_count < 100 {
        let mut buffer = device.capture().expect("Failed to capture frame");
        
        // Process the captured frame here (e.g., save it to a file)
        let mut file = File::create(format!("frame_{}.jpg", frame_count)).expect("Failed to create file");
        file.write_all(&buffer).expect("Failed to write frame to file");
        
        frame_count += 1;
    }
    
    // Stop the video stream
    device.stop().expect("Failed to stop video stream");
}
