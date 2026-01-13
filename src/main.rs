mod activation;
mod data_generator;
mod initializer;
mod model;
mod util;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use dotenv::dotenv;

fn main() {
    dotenv().ok();

    let make_dataset = std::env::var("GENERATE_DATASET").is_ok_and(|v| v == "true");

    if make_dataset {
        let num_dataset: usize = std::env::var("NUM_DATASET")
            .unwrap_or_else(|_| "100000".to_string())
            .parse()
            .unwrap_or(100000);
        data_generator::generate_and_save_data(num_dataset).expect("Failed to generate dataset");
    }

    let device = WgpuDevice::default();
    let model: model::SimpleRegressionModel<Wgpu> = model::SimpleRegressionModel::init(&device);
    let (tensors, _, _) = model.prepare_tensors(0..1000);

    model.do_train(Option::from(tensors));
}
