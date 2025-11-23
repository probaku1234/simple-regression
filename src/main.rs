mod data_generator;
mod initializer;
mod model;

use crate::initializer::Initializer;
use crate::model::Layer;
use burn::backend::NdArray;

fn main() {
    type B = NdArray;
    let zero_tensors =
        Initializer::Zeroes.init_with::<B, 2, [usize; 2]>([2, 2], &Default::default());
    let one_tensors = Initializer::Ones.init_with::<B, 2, [usize; 2]>([2, 2], &Default::default());

    println!(
        "Zero Tensors: {:?}",
        zero_tensors.val().to_data().to_vec::<f32>().unwrap()
    );
    println!(
        "One Tensors: {:?}",
        one_tensors.val().to_data().to_vec::<f32>().unwrap()
    );

    let layer: Layer<B> = Layer::init_with(&Initializer::Ones, 1, 1, &Default::default());
    println!("Layer: {:?}", layer);
}
