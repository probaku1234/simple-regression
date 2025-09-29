use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::module::Param;

pub trait Optimizer<B: Backend> {
    fn step<const D: usize>(
        &self,
        param: &mut Param<Tensor<B, D>>,
        grad: Tensor<B, D>,
    );
}

pub struct SGD {
    pub lr: f32,
}

impl<B: Backend> Optimizer<B> for SGD {
    fn step<const D: usize>(
        &self,
        param: &mut Param<Tensor<B, D>>,
        grad: Tensor<B, D>,
    ) {
        // w_new = w - lr * grad
        let updated = param.val().unsqueeze().sub(grad.mul_scalar(self.lr));
        *param = Param::initialized(param.id, updated);

    }
}

pub struct Adam {
    pub lr: f32,
}

impl<B: Backend> Optimizer<B> for Adam {
    fn step<const D: usize>(
        &self,
        param: &mut Param<Tensor<B, D>>,
        grad: Tensor<B, D>,
    ) {
        // w_new = w - lr * grad
        let updated = param.val().unsqueeze().sub(grad.mul_scalar(self.lr));
        *param = Param::initialized(param.id, updated);

    }
}