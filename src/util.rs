use burn::prelude::Backend;
use burn::tensor::Tensor;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;

pub fn read_data_from_csv(filename: &str) -> Result<Vec<(f32, f32)>, Box<dyn Error>> {
    let mut data = Vec::new();
    let file = File::open(filename)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("x,y") {
            // Skip header
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let x: f32 = parts[0].parse()?;
            let y: f32 = parts[1].parse()?;
            data.push((x, y));
        }
    }

    Ok(data)
}

pub fn debug_tensor<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D>) {
    let v: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "{} shape={:?}, min={:.3}, max={:.3}",
        name,
        t.dims(),
        min,
        max
    );
}
