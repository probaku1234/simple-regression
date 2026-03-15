use burn::module::Param;
use burn::prelude::{Backend, Tensor};

pub mod json_recorder;

pub trait Recorder {
    fn save_all_params<B: Backend>(
        w1: &Param<Tensor<B, 2>>,
        b1: &Param<Tensor<B, 1>>,
        w2: &Param<Tensor<B, 2>>,
        b2: &Param<Tensor<B, 1>>,
        path: &str,
    );
    fn load_all_params<B: Backend>(
        path: &str,
    ) -> (
        Param<Tensor<B, 2>>,
        Param<Tensor<B, 1>>,
        Param<Tensor<B, 2>>,
        Param<Tensor<B, 1>>,
    );
}
