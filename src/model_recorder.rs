use std::fs::File;
use std::io::{BufReader, BufWriter};
use burn::module::Param;
use burn::prelude::{Backend, Shape, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct TensorJson {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct ModelJson {
    w1: TensorJson,
    b1: TensorJson,
    w2: TensorJson,
    b2: TensorJson,
}



fn tensor_to_json<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> TensorJson {
    let shape = tensor.dims();
    // This works on ANY backend
    let values: Vec<f32> = tensor.clone()
        .into_data()
        .convert::<f32>().to_vec().expect("");      // directly get Vec<f32>
    println!("values {:?}", values);
    TensorJson { shape: Vec::from(shape), values }
}

fn tensor_from_json<B: Backend, const D: usize>(tj: &TensorJson) -> Tensor<B, D> {
    let hohoho = Tensor::<B, 1>::from_floats(tj.values.as_slice(), &B::Device::default());
    hohoho
        .reshape(Shape::new::<D>(tj.shape.clone().try_into().unwrap()))
}


pub fn save_all_params<B: Backend>(
    w1: &Param<Tensor<B, 2>>,
    b1: &Param<Tensor<B, 1>>,
    w2: &Param<Tensor<B, 2>>,
    b2: &Param<Tensor<B, 1>>,
    path: &str,
) {
    let model_json = ModelJson {
        w1: tensor_to_json(&w1.val()),
        b1: tensor_to_json(&b1.val()),
        w2: tensor_to_json(&w2.val()),
        b2: tensor_to_json(&b2.val()),
    };

    let file = File::create(path).expect("Failed to create JSON file");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &model_json).expect("Failed to write JSON");
}

pub fn load_all_params<B: Backend>(
    path: &str,
) -> (Param<Tensor<B, 2>>, Param<Tensor<B, 1>>, Param<Tensor<B, 2>>, Param<Tensor<B, 1>>) {
    let file = File::open(path).expect("Failed to open JSON file");
    let reader = BufReader::new(file);
    let model_json: ModelJson = serde_json::from_reader(reader).expect("Failed to parse JSON");

    println!("{:?}", model_json.w1);
    let w1 = tensor_from_json::<B, 2>(&model_json.w1);
    println!("asdasd");
    let b1 = tensor_from_json::<B, 1>(&model_json.b1);
    println!("asdasd");
    let w2 = tensor_from_json::<B, 2>(&model_json.w2);
    println!("asdasd");
    let b2 = tensor_from_json::<B, 1>(&model_json.b2);

    (
        Param::from_tensor(w1),
        Param::from_tensor(b1),
        Param::from_tensor(w2),
        Param::from_tensor(b2),
    )
}