use burn::{tensor::Tensor};
use burn::backend::NdArray;

fn main() {
    fn print_tensor<const D: usize>(label: &str, t: &Tensor<B, D>) {
        let data: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap();
        println!("{label} shape={:?}, values={:?}", t.shape(), data);
    }

    type B = NdArray;

    let t = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0],
                                            [4.0, 5.0, 6.0]],
                                        &Default::default());
    print_tensor("Original", &t);

    let sum0 = t.clone().sum_dim(0);
    print_tensor("After sum_dim(0)", &sum0);

    let sum0_squeezed = sum0.squeeze::<1>(0);
    print_tensor("After squeeze::<0>()", &sum0_squeezed);

    let sum1 = t.sum_dim(1);
    print_tensor("After sum_dim(1)", &sum1);
    let sum1_squeezed = sum1.squeeze::<1>(1);
    print_tensor("After squeeze::<1>()", &sum1_squeezed);

    let tensor = Tensor::<B, 3>::from_data(
        [[[3.0, 4.9, 2.0]], [[2.0, 1.9, 3.0]], [[4.0, 5.9, 8.0]]],
        &Default::default(),
    );
    print_tensor("Original", &tensor);

    let squeezed = tensor.squeeze::<2>(1);
    print_tensor("After squeeze::<2>()", &squeezed);
}
