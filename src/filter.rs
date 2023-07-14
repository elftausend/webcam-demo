use std::str::FromStr;

use custos::{prelude::CUBuffer, buf};
use serde_derive::{Serialize, Deserialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    Sharpen,
    BoxBlur,
    Overflow,
    // car lights? reflections?
    MarkLight,
    EdgeDetect,
    Test,
    None,
}

impl Filter {
    pub fn to_data(&self, marklight_intensity: f32) -> (usize, usize, CUBuffer<'static, f32>) {
        let filter_rows;
        let filter_cols;
        let filter;
        match self {
            #[rustfmt::skip]
            Filter::Sharpen => {
                filter_rows = 3;
                filter_cols = 3;
                filter = custos::buf![
                    0., -1., 0.,
                    -1., 5., -1.,
                    0., -1., 0.,
                ].to_cuda();
            }
            Filter::BoxBlur => {
                // 48x49
                // 23x23 for shared
                filter_rows = 23;
                filter_cols = 23;

                filter =
                    custos::buf![1. / (filter_rows*filter_cols) as f32; filter_rows * filter_cols]
                        .to_cuda();
            }
            Filter::Overflow => {
                filter_rows = 3;
                filter_cols = 3;
                filter = custos::buf![
                    1.; 9
                ]
                .to_cuda();
            }
            Filter::MarkLight => {
                filter_rows = 3;
                filter_cols = 3;
                filter = custos::buf![marklight_intensity; 9].to_cuda();
            }

            #[rustfmt::skip]
            Filter::EdgeDetect => {
                filter_rows = 3;
                filter_cols = 3;
                filter = custos::buf![
                    -1., -1., -1.,
                    -1., 8., -1.,
                    -1., -1., -1.,
                ].to_cuda();
            }
            Filter::Test => {
                filter_rows = 3;
                filter_cols = 3;
                filter = custos::buf![
                    0.14; 9
                ]
                .to_cuda();
            }
            // could skip calculation(s) afterwards completely
            Filter::None => {
                filter_rows = 1;
                filter_cols = 1;
                filter = buf![1.; 1].to_cuda()
            }
        }
        (filter_rows, filter_cols, filter)
    }
}

impl FromStr for Filter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "sharpen" => Filter::Sharpen,
            "boxblur" | "blur" => Filter::BoxBlur,
            "overflow" => Filter::Overflow,
            "marklight" => Filter::MarkLight,
            "edgedetect" | "edge" => Filter::EdgeDetect,
            "test" => Filter::Test,
            "none" => Filter::None,
            _ => return Err(format!("Unknown filter: {s}")),
        })
    }
}