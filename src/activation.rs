use burn::prelude::{Backend, Tensor};

pub trait Activation<const D: usize, B: Backend> {
    fn forward(tensor: Tensor<B, D>) -> Tensor<B, D>;
    fn backward(grad_output: Tensor<B, D>, input: Tensor<B, D>) -> Tensor<B, D>;
}

pub struct ReLU;

impl<const D: usize, B: Backend> Activation<D, B> for ReLU {
    fn forward(tensor: Tensor<B, D>) -> Tensor<B, D> {
        tensor.clamp_min(0)
    }

    fn backward(grad_output: Tensor<B, D>, input: Tensor<B, D>) -> Tensor<B, D> {
        let zero = Tensor::<B, D>::zeros(input.shape(), &input.device());
        let mask = input.clone().greater(zero).float();

        grad_output * mask
    }
}