use burn::module::Param;
use burn::prelude::{Backend, Tensor};

pub trait Optimizer<B: Backend> {
    fn step<const D: usize>(
        &mut self,
        lr: f32,
        param: &mut Param<Tensor<B, D>>,
        grad: Tensor<B, D>,
    );
}

pub struct SGD {}

impl<B: Backend> Optimizer<B> for SGD {
    fn step<const D: usize>(
        &mut self,
        lr: f32,
        param: &mut Param<Tensor<B, D>>,
        grad: Tensor<B, D>,
    ) {
        // w_new = w - lr * grad
        let update = grad.mul_scalar(lr);
        let updated = param.val().unsqueeze().sub(update);
        *param = Param::initialized(param.id, updated);
    }
}
