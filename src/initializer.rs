use burn::module::{Param, ParamId};
use burn::prelude::{Backend, Shape};
use burn::tensor::Tensor;

#[derive(Clone)]
pub enum Initializer {
    Zeroes,
    Ones,
}

impl Initializer {
    pub fn init_with<B: Backend, const D: usize, S: Into<Shape>>(
        &self,
        shape: S,
        device: &B::Device,
    ) -> Param<Tensor<B, D>> {
        let device = device.clone();
        let shape: Shape = shape.into();
        let config = self.clone();

        Param::uninitialized(
            ParamId::new(),
            move |device, _| {
                let tensor = match config {
                    Initializer::Zeroes => Tensor::<B, D>::zeros(shape, &device),
                    Initializer::Ones => Tensor::<B, D>::ones(shape, &device),
                };

                tensor
            },
            device,
            false,
        )
    }
}
