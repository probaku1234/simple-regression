use crate::activation::{Activation, ReLU};
use crate::initializer::Initializer;
use crate::optimizer::{Optimizer, SGD};
use crate::util::{debug_tensor, read_data_from_csv};
use burn::module::Param;
use burn::prelude::{Backend, Tensor};

pub static ARTIFACT_DIR: &str = "./model";

fn create_artifact_dir(run_name: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
    std::fs::create_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
}
#[derive(Debug, Clone, PartialEq)]
enum ScalingMethod {
    None,
    Norm,
    Stand,
}

/// Configuration for training a neural network model.
///
/// This structure holds the parameters required to configure the training process,
/// such as the size of the hidden layer, the number of epochs, and the batch size.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// The size of the hidden layer in the neural network.
    hidden_size: usize,
    /// The size of each batch used during training.
    batch_size: usize,
    num_epochs: usize,
    scaling_method: ScalingMethod,
    learning_rate: f32,
    run_name: String,
}

impl TrainConfig {
    fn new() -> Self {
        let hidden_size = std::env::var("HIDDEN_SIZE")
            .unwrap_or_else(|_| "64".to_string())
            .parse()
            .unwrap_or(64);

        let batch_size = std::env::var("BATCH_SIZE")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);
        let num_epochs = std::env::var("NUM_EPOCHS")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);
        let scaling_method = match std::env::var("SCALING_METHOD").unwrap_or_default().as_str() {
            "norm" => ScalingMethod::Norm,
            "stand" => ScalingMethod::Stand,
            _ => ScalingMethod::None,
        };
        let learning_rate = std::env::var("LEARNING_RATE")
            .unwrap_or_else(|_| "0.01".to_string())
            .parse()
            .unwrap_or(0.01);

        let run_name = std::env::var("RUN_NAME").unwrap_or_else(|_| "default_run".to_string());

        TrainConfig {
            hidden_size,
            batch_size,
            num_epochs,
            scaling_method,
            learning_rate,
            run_name,
        }
    }
}

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

pub struct SimpleRegressionModel<B: Backend> {
    // Stores the hyperparameters (hidden_size, batch_size, etc.)
    train_config: TrainConfig,
    // Dimensionality of the input (1 for our single feature x)
    d_input: usize,
    // Dimensionality of the output (1 for our single predicted value y)
    d_output: usize,
    // The Input -> Hidden layer connection (H neurons)
    input_layer: Layer<B>,
    // The Hidden -> Output layer connection (1 neuron)
    output_layer: Layer<B>,
    // The device (CPU or GPU) the model is initialized on
    device: B::Device,
    optimizer: SGD,
}

impl<B: Backend> SimpleRegressionModel<B> {
    /// Initializes the SimpleRegressionModel on the specified device.
    pub fn init(device: &B::Device) -> Self {
        // 1. Load Configuration
        let train_config = TrainConfig::new();

        // 2. Define Dimensions
        let d_input = 1;
        let d_output = 1;

        // 3. Choose Initializer (Weights and Biases)
        let initializer = Initializer::Ones;

        // 4. Create Layers
        // Input Layer: 1 input feature -> H (hidden_size) neurons
        let input_layer = Layer::init_with(&initializer, d_input, train_config.hidden_size, device);

        // Output Layer: H (hidden_size) features -> 1 output feature
        let output_layer =
            Layer::init_with(&initializer, train_config.hidden_size, d_output, device);

        let optimizer = SGD {};

        Self {
            train_config,
            d_input,
            d_output,
            input_layer,
            output_layer,
            device: device.clone(),
            optimizer,
        }
    }

    pub fn prepare_tensors(
        &self,
        range: std::ops::Range<usize>,
    ) -> (Vec<(Tensor<B, 2>, Tensor<B, 1>)>, (f32, f32), (f32, f32)) {
        let data = read_data_from_csv("data.csv").expect("should read data from csv");
        let batch_size = self.train_config.batch_size;

        let (mut x_stats, mut y_stats) = ((0.0, 1.0), (0.0, 1.0));

        let start = range.start;
        let end = range.end;
        let slice = &data[start..end];

        let xs: Vec<f32> = slice.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = slice.iter().map(|(_, y)| *y).collect();

        match self.train_config.scaling_method {
            ScalingMethod::Stand => {
                let x_mean = xs.iter().sum::<f32>() / xs.len() as f32;
                let y_mean = ys.iter().sum::<f32>() / ys.len() as f32;
                let x_std = (xs.iter().map(|&v| (v - x_mean).powi(2)).sum::<f32>()
                    / xs.len() as f32)
                    .sqrt();
                let y_std = (ys.iter().map(|&v| (v - y_mean).powi(2)).sum::<f32>()
                    / ys.len() as f32)
                    .sqrt();
                x_stats = (x_mean, x_std);
                y_stats = (y_mean, y_std);
            }
            ScalingMethod::Norm => {
                let x_min = xs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let x_max = xs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let y_min = ys.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let y_max = ys.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                x_stats = (x_min, x_max);
                y_stats = (y_min, y_max);
            }
            ScalingMethod::None => {}
        }

        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1>> = Vec::new();

        for (x, y) in slice.iter() {
            let (x_final, y_final) = match self.train_config.scaling_method {
                ScalingMethod::Stand => (
                    (*x - x_stats.0) / x_stats.1.max(1e-8),
                    (*y - y_stats.0) / y_stats.1.max(1e-8),
                ),
                ScalingMethod::Norm => (
                    (*x - x_stats.0) / (x_stats.1 - x_stats.0).max(1e-8),
                    (*y - y_stats.0) / (y_stats.1 - y_stats.0).max(1e-8),
                ),
                ScalingMethod::None => (*x, *y),
            };

            inputs.push(Tensor::<B, 1>::from_floats([x_final], &self.device).unsqueeze());
            targets.push(Tensor::<B, 1>::from_floats([y_final], &self.device));
        }

        let mut batched_inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut batched_targets: Vec<Tensor<B, 1>> = Vec::new();

        for i in (0..inputs.len()).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, inputs.len());
            let input_tensor = Tensor::cat(
                inputs[i..end]
                    .iter()
                    .map(|t| t.clone().unsqueeze())
                    .collect(),
                0,
            );
            let target_tensor = Tensor::cat(
                targets[i..end]
                    .iter()
                    .map(|t| t.clone().unsqueeze())
                    .collect(),
                0,
            );

            batched_inputs.push(input_tensor);
            batched_targets.push(target_tensor);
        }

        (
            batched_inputs.into_iter().zip(batched_targets).collect(),
            x_stats,
            y_stats,
        )
    }

    fn compute_loss(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss = logits.sub(targets).powf_scalar(2.0);

        loss.mean()
    }

    pub fn do_train(&mut self, input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>>) {
        create_artifact_dir(&self.train_config.run_name);

        let input_target_tensors = input_target_tensors.unwrap();

        let mut iteration: usize = 1;
        let num_epochs = self.train_config.num_epochs;

        for epoch in 0..num_epochs {
            println!(
                "================= Epoch {}/{} =================",
                epoch + 1,
                num_epochs
            );

            for (inputs, targets) in input_target_tensors.iter() {
                println!("---------{iteration}th iteration start---------");

                println!("---------forward pass---------");
                let weight_1 = self.input_layer.weight.val().unsqueeze();
                let bias_1 = self.input_layer.bias.val().unsqueeze();
                let z1 = inputs.clone().matmul(weight_1) + bias_1;
                debug_tensor("Layer 1 pre-activation (z1)", &z1);

                let a1 = ReLU::forward(z1.clone());
                debug_tensor("Layer 1 activation (a1)", &a1);

                let weight_2 = self.output_layer.weight.val().unsqueeze();
                let bias_2 = self.output_layer.bias.val().unsqueeze();
                let z2 = a1.clone().matmul(weight_2.clone()) + bias_2;
                debug_tensor("Layer 2 pre-activation (z2)", &z2);

                let targets: Tensor<B, 2> = targets.clone().unsqueeze_dim(1);
                let loss = self.compute_loss(z2.clone(), targets.clone());
                debug_tensor("Loss", &loss);

                println!("---------backward pass---------");
                let grad_logits = z2.sub(targets) * (2.0 / self.train_config.batch_size as f32);
                debug_tensor("grad_logits", &grad_logits);

                let grad_weight_2 = a1.transpose().matmul(grad_logits.clone());
                let grad_bias_2 = grad_logits.clone().sum_dim(0);
                debug_tensor("grad_weight_2", &grad_weight_2);
                debug_tensor("grad_bias_2", &grad_bias_2);

                let grad_hidden = grad_logits.matmul(weight_2.transpose());
                let grad_hidden_relu = ReLU::backward(grad_hidden, z1.clone());

                let grad_weight_1 = inputs.clone().transpose().matmul(grad_hidden_relu.clone());
                let grad_bias_1 = grad_hidden_relu.clone().sum_dim(0);
                debug_tensor("grad_weight_1", &grad_weight_1);
                debug_tensor("grad_bias_1", &grad_bias_1);

                let mut grads = [
                    grad_weight_1.clone(),
                    grad_bias_1.clone(),
                    grad_weight_2.clone(),
                    grad_bias_2.clone(),
                ];

                println!("---------updating weights and biases---------");
                self.optimizer.step(
                    self.train_config.learning_rate,
                    &mut self.input_layer.weight,
                    grads[0].clone(),
                );
                // sum_dum preserves the shape of tensor, so we need to squeeze the first dimension
                self.optimizer.step(
                    self.train_config.learning_rate,
                    &mut self.input_layer.bias,
                    grads[1].clone().squeeze(0),
                );

                self.optimizer.step(
                    self.train_config.learning_rate,
                    &mut self.output_layer.weight,
                    grads[2].clone(),
                );
                // sum_dum preserves the shape of tensor, so we need to squeeze the first dimension
                self.optimizer.step(
                    self.train_config.learning_rate,
                    &mut self.output_layer.bias,
                    grads[3].clone().squeeze(0),
                );

                println!("---------{iteration}th iteration end---------");
                iteration += 1;
            }
        }
    }
}
