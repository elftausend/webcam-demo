use std::{ffi::CString, str::FromStr};

use clap::Parser;
use custos::{
    cuda::{fn_cache, CUDAPtr},
    flag::AllocFlag,
    prelude::CUBuffer,
    CUDA,
};
use filter::Filter;
use serde_derive::{Deserialize, Serialize};

use crate::{glow_webcam::check_error, videotex::cuModuleGetGlobal_v2};

mod correlate_test_kernels;
mod cu_filter;
mod glium_webcam;
mod glow_webcam;
mod jpeg_decoder;
mod videotex;
mod filter;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GLBackend {
    Glium,
    Glow,
}

impl FromStr for GLBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "glium" => GLBackend::Glium,
            "glow" => GLBackend::Glow,
            _ => return Err(format!("Unknown filter: {s}")),
        })
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value = "sharpen")]
    filter: Filter,

    #[arg(short, long, default_value = "glow")]
    gl_backend: GLBackend,

    #[arg(short, long, default_value = "0.13")]
    marklight_intensity: f32,
}

fn main() {
    let args = Args::parse();

    match args.gl_backend {
        GLBackend::Glium => glium_webcam::glium_webcam().unwrap(),
        GLBackend::Glow => glow_webcam::glow_webcam(&args),
    }
}

// integrate into custos -> this should be a buffer ref (concept does not exist in custos -> "allocflag" instead)
pub fn get_constant_memory<'a, T>(
    device: &'a CUDA,
    src: &str,
    fn_name: &str,
    var_name: &str,
) -> CUBuffer<'a, T> {
    let func = fn_cache(device, src, fn_name).unwrap();

    let module = device.modules.borrow().get(&func).unwrap().0;

    let filter_var = CString::new(var_name).unwrap();

    let mut size = 0;
    let mut filter_data_ptr = 0;
    unsafe {
        check_error(
            cuModuleGetGlobal_v2(&mut filter_data_ptr, &mut size, module, filter_var.as_ptr()),
            "Cannot get global variable",
        )
    };

    CUBuffer {
        ptr: CUDAPtr {
            ptr: filter_data_ptr,
            flag: AllocFlag::Wrapper,
            len: size as usize / std::mem::size_of::<T>(),
            p: std::marker::PhantomData,
        },
        device: Some(device),
        ident: None,
    }
}
