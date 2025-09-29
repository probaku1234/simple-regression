use crate::activation::{Activation, ReLU};
use crate::model_recorder::{load_all_params, save_all_params};
use crate::optimizer::{Optimizer, SGD};
use crate::plotter::{plot_loss, plot_predictions_vs_expectations};
use crate::util::{debug_tensor, read_data_from_csv};
use burn::module::Param;
use burn::nn::Initializer;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

pub static ARTIFACT_DIR: &str = "./model";

fn create_artifact_dir(run_name: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
    std::fs::create_dir_all(format!("{ARTIFACT_DIR}/{run_name}")).ok();
}

#[derive(Debug, Serialize)]
enum OptimizerType {
    SGD,
    Adam,
}

impl FromStr for OptimizerType {
    type Err = String; // you can create a custom error type if you like

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sgd" => Ok(OptimizerType::SGD),
            "adam" => Ok(OptimizerType::Adam),
            _ => Err(format!("Unknown optimizer: {}", s)),
        }
    }
}

#[derive(Debug)]
struct TrainConfig {
    hidden_size: usize,
    num_epochs: usize,
    batch_size: usize,
    generate_dataset: bool,
    learning_rate: f32,
    normalize_dataset: bool,
    do_train: bool,
    do_validation: bool,
    do_test: bool,
    num_train_samples: usize,
    num_valid_samples: usize,
    num_test_samples: usize,
    log_step: usize,
    optimizer: OptimizerType,
    run_name: String,
}

impl TrainConfig {
    fn new() -> Self {
        let hidden_size = std::env::var("HIDDEN_SIZE")
            .unwrap_or_else(|_| "64".to_string())
            .parse()
            .unwrap_or(64);
        let num_epochs = std::env::var("NUM_EPOCHS")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);
        let batch_size = std::env::var("BATCH_SIZE")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);
        let generate_dataset = std::env::var("GENERATE_DATASET")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);
        let learning_rate = std::env::var("LEARNING_RATE")
            .unwrap_or_else(|_| "0.01".to_string())
            .parse()
            .unwrap_or(0.01);
        let normalize_dataset = std::env::var("NORMALIZE_DATASET")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);
        let do_train = std::env::var("DO_TRAIN")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        let do_validation = std::env::var("DO_VALIDATION")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);
        let do_test = std::env::var("DO_TEST")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);
        let num_train_samples = std::env::var("NUM_TRAIN_SAMPLES")
            .unwrap_or_else(|_| "1000".to_string())
            .parse()
            .unwrap_or(1000);
        let num_valid_samples = std::env::var("NUM_VALID_SAMPLES")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        let num_test_samples = std::env::var("NUM_TEST_SAMPLES")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        let log_step = std::env::var("LOG_STEP")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);
        let optimizer: OptimizerType = std::env::var("OPTIMIZER")
            .unwrap_or_else(|_| "sgd".to_string())
            .parse()
            .unwrap_or(OptimizerType::Adam);
        let run_name = std::env::var("RUN_NAME").unwrap_or_else(|_| "default".to_string());

        TrainConfig {
            hidden_size,
            num_epochs,
            batch_size,
            generate_dataset,
            learning_rate,
            normalize_dataset,
            do_train,
            do_validation,
            do_test,
            num_train_samples,
            num_valid_samples,
            num_test_samples,
            log_step,
            optimizer,
            run_name,
        }
    }

    fn get_config_for_logging(&self) -> Value {
        json!({
            "hidden_size": self.hidden_size,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "normalize_dataset": self.normalize_dataset,
            "do_train": self.do_train,
            "do_validation": self.do_validation,
            "do_test": self.do_test,
            "num_train_samples": self.num_train_samples,
            "num_valid_samples": self.num_valid_samples,
            "num_test_samples": self.num_test_samples,
            "log_step": self.log_step,
            "optimizer": self.optimizer,
        })
    }
}

struct Layer<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Layer<B> {
    fn init_with(
        initializer: &Initializer,
        d_input: usize,
        d_out: usize,
        device: &B::Device,
    ) -> Self {
        let weight = initializer.init_with::<B, 2, [usize; 2]>(
            [d_input, d_out],
            Some(d_input),
            Some(d_out),
            &device,
        );
        let bias =
            initializer.init_with::<B, 1, [usize; 1]>([d_out], Some(d_input), Some(d_out), &device);
        Self { weight, bias }
    }
}

pub struct SimpleRegressionModel<B: Backend> {
    train_config: TrainConfig,
    d_input: usize,
    d_output: usize,
    input_layer: Layer<B>,
    output_layer: Layer<B>,
    device: B::Device,
    activation: ReLU,
    train_loss_log_file: String,
    val_loss_log_file: String,
    grad_loss_log_file: String,
    model_name: String,
    optimizer: SGD,
}

impl<B: Backend> SimpleRegressionModel<B> {
    pub fn init(device: &B::Device) -> Self {
        let train_config = TrainConfig::new();
        let d_input = 1;
        let d_output = 1;

        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(3.0),
            fan_out_only: false,
        };
        let input_layer = Layer::init_with(&initializer, d_input, train_config.hidden_size, device);
        let output_layer =
            Layer::init_with(&initializer, train_config.hidden_size, d_output, device);

        let activation = ReLU;
        let optimizer = SGD {
            lr: train_config.learning_rate,
        };

        Self {
            train_config,
            d_input,
            d_output,
            input_layer,
            output_layer,
            device: device.clone(),
            activation,
            train_loss_log_file: String::from("training_loss.log"),
            val_loss_log_file: String::from("validation_loss.log"),
            grad_loss_log_file: String::from("gradient_norm.log"),
            model_name: String::from("model.json"),
            optimizer,
        }
    }

    fn prepare_tensors(
        &self,
        range: std::ops::Range<usize>,
    ) -> (Vec<(Tensor<B, 2>, Tensor<B, 1>)>, (f32, f32), (f32, f32)) {
        let data = read_data_from_csv("data.csv").expect("should read data from csv");
        let batch_size = self.train_config.batch_size;
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1>> = Vec::new();
        let mut x_mean: f32 = 0.0;
        let mut x_std: f32 = 1.0;
        let mut y_mean: f32 = 0.0;
        let mut y_std: f32 = 1.0;

        if range.end - range.start > data.len() || range.end > data.len() {
            panic!("Range is greater than dataset length {}", data.len());
        }

        let start = range.start;
        let end = range.end;

        if self.train_config.normalize_dataset {
            // ---- 1. Collect raw xs for normalization ----
            let xs: Vec<f32> = data[start..end].iter().map(|(x, _)| *x).collect();
            let ys: Vec<f32> = data[start..end].iter().map(|(_, y)| *y).collect();

            // ---- 2. Compute mean/std for X ----
            x_mean = xs.iter().sum::<f32>() / xs.len() as f32;
            y_mean = ys.iter().copied().sum::<f32>() / ys.len() as f32;

            x_std =
                (xs.iter().map(|&v| (v - x_mean).powi(2)).sum::<f32>() / xs.len() as f32).sqrt();
            y_std =
                (ys.iter().map(|&v| (v - y_mean).powi(2)).sum::<f32>() / ys.len() as f32).sqrt();

            for (x, y) in data[start..end].iter() {
                let x_norm = (*x - x_mean) / x_std.max(1e-8);
                let y_norm = (*y - y_mean) / y_std.max(1e-8);

                let input_tensor = Tensor::<B, 1>::from_floats([x_norm], &self.device);
                let target_tensor = Tensor::<B, 1>::from_floats([y_norm], &self.device); // keep y unchanged

                inputs.push(input_tensor.unsqueeze()); // [1] → [1, 1]
                targets.push(target_tensor);
            }
        } else {
            for (x, y) in data[start..end].iter() {
                let input_tensor = Tensor::<B, 1>::from_floats([*x], &self.device);
                let target_tensor = Tensor::<B, 1>::from_floats([*y], &self.device);
                inputs.push(input_tensor.unsqueeze()); // [1] → [1, 1]
                targets.push(target_tensor);
            }
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

        (
            batched_inputs.into_iter().zip(batched_targets).collect(),
            (x_mean, x_std),
            (y_mean, y_std),
        )
    }

    fn compute_loss(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss = logits.sub(targets).powf_scalar(2.0);

        loss.mean()
    }

    fn compute_grad_norm(&self, grads: &[Tensor<B, 2>]) -> Tensor<B, 1> {
        let mut total = Tensor::<B, 1>::zeros([1], &self.device);
        for g in grads {
            total = total + g.clone().powf_scalar(2.0).sum();
        }
        total.sqrt()
    }

    fn log_float_value(&self, loss: f32, path: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!(
                "{}/{}/{}",
                ARTIFACT_DIR, self.train_config.run_name, path
            ))
            .expect("Failed to open loss log file");

        writeln!(file, "{:.6}", loss).expect("Failed to write loss");
    }

    fn save_config(&self) {
        let config_value = self.train_config.get_config_for_logging();
        let path = format!(
            "{}/{}/{}",
            ARTIFACT_DIR, self.train_config.run_name, "config.json"
        );

        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent).expect("Failed to create directories");
        }

        fs::write(
            &path,
            serde_json::to_string_pretty(&config_value).expect("Failed to serialize JSON"),
        )
        .expect("Failed to write config.json");
    }

    fn do_train(
        &mut self,
        input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>>,
        valid_input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>>,
    ) {
        create_artifact_dir(&self.train_config.run_name);

        println!("Train Config: {:?}", self.train_config);
        println!("weights_1\n {:}", self.input_layer.weight.val());
        println!("bias_1\n {:}", self.input_layer.bias.val());
        println!("weights_2\n {:}", self.output_layer.weight.val());
        println!("bias_2\n {:}", self.output_layer.bias.val());

        let input_target_tensors = input_target_tensors.unwrap();
        let valid_input_target_tensors = valid_input_target_tensors.unwrap();

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
                let loss_value: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
                if iteration % self.train_config.log_step == 0 {
                    self.log_float_value(loss_value, &*self.train_loss_log_file);
                }

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

                let grad_norm = self.compute_grad_norm(&[
                    grad_weight_1.clone(),
                    grad_bias_1.clone(),
                    grad_weight_2.clone(),
                    grad_bias_2.clone(),
                ]);
                debug_tensor("grad norm", &grad_norm);

                let grad_norm_value: f32 =
                    grad_norm.to_data().convert::<f32>().to_vec().unwrap()[0];
                if iteration % self.train_config.log_step == 0 {
                    self.log_float_value(grad_norm_value, &*self.grad_loss_log_file);
                }

                println!("---------updating weights and biases---------");
                self.optimizer
                    .step(&mut self.input_layer.weight, grad_weight_1.clone());
                // sum_dum preserves the shape of tensor, so we need to squeeze the first dimension
                self.optimizer
                    .step(&mut self.input_layer.bias, grad_bias_1.clone().squeeze(0));

                self.optimizer
                    .step(&mut self.output_layer.weight, grad_weight_2.clone());
                // sum_dum preserves the shape of tensor, so we need to squeeze the first dimension
                self.optimizer
                    .step(&mut self.output_layer.bias, grad_bias_2.clone().squeeze(0));

                if self.train_config.do_validation {
                    self.do_validation(&valid_input_target_tensors, iteration);
                }

                println!("---------{iteration}th iteration end---------");
                iteration += 1;
            }
        }

        save_all_params(
            &self.input_layer.weight,
            &self.input_layer.bias,
            &self.output_layer.weight,
            &self.output_layer.bias,
            format!(
                "{}/{}/{}",
                ARTIFACT_DIR, self.train_config.run_name, self.model_name
            )
            .as_str(),
        );
        plot_loss(
            &self.train_config.run_name,
            &self.train_loss_log_file,
            &format!(
                "{}/{}/train_loss.png",
                ARTIFACT_DIR, self.train_config.run_name
            ),
            "Train Loss Curve",
            "Train Loss",
            self.train_config.log_step,
        )
        .expect("should plot loss curve");
        plot_loss(
            &self.train_config.run_name,
            &self.val_loss_log_file,
            &format!(
                "{}/{}/valid_loss.png",
                ARTIFACT_DIR, self.train_config.run_name
            ),
            "Validation Loss Curve",
            "Validation Loss",
            self.train_config.log_step,
        )
        .expect("should plot loss curve");
        plot_loss(
            &self.train_config.run_name,
            &self.grad_loss_log_file,
            &format!(
                "{}/{}/grad_loss.png",
                ARTIFACT_DIR, self.train_config.run_name
            ),
            "Gradient Loss Curve",
            "Gradient Loss",
            1,
        )
        .expect("should plot loss curve");
        self.save_config();
    }

    fn do_validation(
        &mut self,
        valid_input_target_tensors: &Vec<(Tensor<B, 2>, Tensor<B, 1>)>,
        iteration: usize,
    ) {
        println!("---------validation---------");
        let mut losses: Vec<f32> = Vec::new();

        for (val_input, val_target) in valid_input_target_tensors.iter() {
            let val_weight_1 = self.input_layer.weight.val().unsqueeze();
            let val_bias_1 = self.input_layer.bias.val().unsqueeze();
            let val_z1 = val_input.clone().matmul(val_weight_1) + val_bias_1;
            let val_a1 = ReLU::forward(val_z1.clone());

            let val_weight_2 = self.output_layer.weight.val().unsqueeze();
            let val_bias_2 = self.output_layer.bias.val().unsqueeze();
            let val_z2 = val_a1.matmul(val_weight_2) + val_bias_2;

            let val_targets: Tensor<B, 2> = val_target.clone().unsqueeze_dim(1);
            let val_loss = self.compute_loss(val_z2.clone(), val_targets.clone());
            let val_loss_value: f32 = val_loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            losses.push(val_loss_value);
        }

        let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
        println!("Validation loss: {:.6}", avg_loss);

        if iteration % self.train_config.log_step == 0 {
            self.log_float_value(avg_loss, &*self.val_loss_log_file);
        }
    }

    fn do_test(
        &mut self,
        input_target_tensors: Vec<(Tensor<B, 2>, Tensor<B, 1>)>,
        y_mean_std: Option<(f32, f32)>,
    ) {
        let (y_mean, y_std) = y_mean_std.unwrap_or((0.0, 1.0));

        let (w1, b1, w2, b2) = load_all_params::<B>(
            format!(
                "{}/{}/{}",
                ARTIFACT_DIR, self.train_config.run_name, self.model_name
            )
            .as_str(),
        );

        println!("W1: {:?}", w1.val().to_data().to_vec::<f32>().unwrap());
        println!("B1: {:?}", b1.val().to_data().to_vec::<f32>().unwrap());
        println!("W2: {:?}", w2.val().to_data().to_vec::<f32>().unwrap());
        println!("B2: {:?}", b2.val().to_data().to_vec::<f32>().unwrap());

        let mut predictions: Vec<f32> = Vec::new();
        let mut expectations: Vec<f32> = Vec::new();
        let mut mses: Vec<f32> = Vec::new();
        let mut maes: Vec<f32> = Vec::new();
        let mut r2s: Vec<f32> = Vec::new();

        for (inputs, targets) in input_target_tensors.iter() {
            println!("---------inference---------");

            // First layer
            let weight_1 = w1.val(); // shape [1, 64]
            let bias_1 = b1.val().reshape([1, self.train_config.hidden_size]); // reshape bias to match broadcast shape
            let z1 = inputs.clone().matmul(weight_1) + bias_1;
            let a1 = ReLU::forward(z1.clone());

            // Second layer
            let weight_2 = w2.val(); // shape [64, 1]
            let bias_2 = b2.val().reshape([1, 1]); // reshape bias to match broadcast shape
            let z2 = a1.matmul(weight_2.clone()) + bias_2;

            let mut predicted_values: Vec<f32> = z2
                .clone()
                .to_device(&B::Device::default()) // move to CPU
                .into_data()
                .to_vec()
                .unwrap()
                .iter()
                .map(|y_norm: &f32| y_norm * y_std + y_mean) // denormalize
                .collect();

            predictions.append(&mut predicted_values.clone());

            let mut expected_values: Vec<f32> = targets
                .clone()
                .to_device(&B::Device::default())
                .into_data()
                .to_vec()
                .unwrap()
                .iter()
                .map(|y_norm: &f32| y_norm * y_std + y_mean) // denormalize
                .collect();

            expectations.append(&mut expected_values.clone());

            println!("Predicted: {:?}", predicted_values);
            println!("Expected: {:?}", expected_values);

            // Compute metrics
            let mut mae = 0.0;
            let mut mse = 0.0;
            for (pred, exp) in predicted_values.iter().zip(expected_values.iter()) {
                let diff = pred - exp;
                mae += diff.abs();
                mse += diff * diff;
            }

            let n = predicted_values.len() as f32;
            mae /= n;
            mse /= n;

            // Optional: R² score
            let mean_y: f32 = expected_values.iter().copied().sum::<f32>() / n;
            let ss_tot: f32 = expected_values.iter().map(|y| (y - mean_y).powi(2)).sum();
            let ss_res: f32 = predicted_values
                .iter()
                .zip(expected_values.iter())
                .map(|(p, y)| (y - p).powi(2))
                .sum();
            let r2 = 1.0 - ss_res / ss_tot;

            mses.push(mse);
            maes.push(mae);
            r2s.push(r2);
            println!("Predicted: {:?}", predicted_values);
            println!("Expected: {:?}", expected_values);
            println!("MAE: {:.4}, MSE: {:.4}, R²: {:.4}", mae, mse, r2);
        }

        let avg_mse: f32 = mses.iter().copied().sum::<f32>() / mses.len() as f32;
        let avg_mae: f32 = maes.iter().copied().sum::<f32>() / maes.len() as f32;
        let avg_r2: f32 = r2s.iter().copied().sum::<f32>() / r2s.len() as f32;
        println!(
            "Average MAE: {:.4}, Average MSE: {:.4}, Average R²: {:.4}",
            avg_mae, avg_mse, avg_r2
        );

        plot_predictions_vs_expectations(
            &predictions,
            &expectations,
            &format!(
                "{}/{}/predictions_vs_expectations.png",
                ARTIFACT_DIR, self.train_config.run_name
            ),
        )
        .expect("should plot predictions vs expectations");
    }

    pub fn process(&mut self) {
        let num_train_samples = self.train_config.num_train_samples;
        let num_valid_samples = self.train_config.num_valid_samples;
        let mut input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>> = None;
        let mut valid_input_target_tensors: Option<Vec<(Tensor<B, 2>, Tensor<B, 1>)>> = None;
        let mut y_mean_std: Option<(f32, f32)> = None;

        if self.train_config.do_train {
            let result = self.prepare_tensors(0..num_train_samples);
            input_target_tensors = Some(result.0);
            y_mean_std = Some(result.2);
        }
        if self.train_config.do_validation {
            valid_input_target_tensors = Some(
                self.prepare_tensors(num_train_samples..(num_train_samples + num_valid_samples))
                    .0,
            );
        }

        if input_target_tensors.is_some() {
            self.do_train(input_target_tensors, valid_input_target_tensors);
        }

        if self.train_config.do_test {
            let start = num_train_samples + num_valid_samples;
            let num_test_samples = self.train_config.num_test_samples;
            let (input_target_tensors, _, _) =
                self.prepare_tensors(start..(start + num_test_samples));
            self.do_test(input_target_tensors, y_mean_std);
        }
    }
}
