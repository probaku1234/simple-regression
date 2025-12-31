use crate::activation::{Activation, ReLU};
use crate::initializer::Initializer;
use crate::util::{debug_tensor, read_data_from_csv};
use burn::module::Param;
use burn::prelude::{Backend, Tensor};

pub static ARTIFACT_DIR: &str = "./model";

fn create_artifact_dir(run_name: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
    std::fs::create_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
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
        
        let run_name = std::env::var("RUN_NAME")
            .unwrap_or_else(|_| "default_run".to_string());
        
        TrainConfig {
            hidden_size,
            batch_size,
            num_epochs,
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

        Self {
            train_config,
            d_input,
            d_output,
            input_layer,
            output_layer,
            device: device.clone(),
        }
    }

    pub fn prepare_tensors(
        &self,
        range: std::ops::Range<usize>,
    ) -> (Vec<(Tensor<B, 2>, Tensor<B, 1>)>) {
        let data = read_data_from_csv("data.csv").expect("should read data from csv");
        let batch_size = self.train_config.batch_size;
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1>> = Vec::new();

        if range.end - range.start > data.len() || range.end > data.len() {
            panic!("Range is greater than dataset length {}", data.len());
        }

        let start = range.start;
        let end = range.end;

        for (x, y) in data[start..end].iter() {
            let input_tensor = Tensor::<B, 1>::from_floats([*x], &self.device);
            let target_tensor = Tensor::<B, 1>::from_floats([*y], &self.device);
            inputs.push(input_tensor.unsqueeze()); // [1] → [1, 1]
            targets.push(target_tensor);
        }

        let mut batched_inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut batched_targets: Vec<Tensor<B, 1>> = Vec::new();

        for i in (0..inputs.len()).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, inputs.len());
            let batch_inputs = &inputs[i..end];
            let batch_targets = &targets[i..end];

            let input_tensor = Tensor::cat(
                batch_inputs.iter().map(|t| t.clone().unsqueeze()).collect(),
                0,
            );
            let target_tensor = Tensor::cat(
                batch_targets
                    .iter()
                    .map(|t| t.clone().unsqueeze())
                    .collect(),
                0,
            );

            batched_inputs.push(input_tensor);
            batched_targets.push(target_tensor);
        }

        batched_inputs.into_iter().zip(batched_targets).collect()
    }

    fn compute_loss(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss = logits.sub(targets).powf_scalar(2.0);

        loss.mean()
    }

    pub fn do_train(
        &self,
        input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>>,
    ) {
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
            }
        }
    }
}
