use crate::initializer::Initializer;
use burn::module::Param;
use burn::prelude::{Backend, Tensor};

/// Represents a neural network layer with parameters for weights and biases.
///
/// # Type Parameters
/// - `B`: A type that implements the `Backend` trait, which defines the backend
///   used for tensor operations (e.g., CPU, GPU).
#[derive(Debug)]
pub struct Layer<B: Backend> {
    /// The weight parameter of the layer, represented as a 2-dimensional tensor.
    weight: Param<Tensor<B, 2>>,
    /// The bias parameter of the layer, represented as a 1-dimensional tensor.
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Layer<B> {
    pub fn init_with(
        initializer: &Initializer,
        d_input: usize,
        d_out: usize,
        device: &B::Device,
    ) -> Self {
        let weight = initializer.init_with::<B, 2, [usize; 2]>([d_input, d_out], &device);
        let bias = initializer.init_with::<B, 1, [usize; 1]>([d_out], &device);
        Self { weight, bias }
    }
}
