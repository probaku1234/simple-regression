use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn generate_and_save_data(num_points: usize) -> Result<(), Box<dyn Error>> {
    let noise_std_dev = 10.0;

    let normal_dist = Normal::new(0.0, noise_std_dev)?;
    let mut rng = rand::rng();

    let x_range_min = -100.0;
    let x_range_max = 100.0;

    let mut data: Vec<(f32, f32)> = Vec::new();

    for _ in 0..num_points {
        let x_f: f32 = rng.random_range(x_range_min..x_range_max);
        let noise = normal_dist.sample(&mut rng);
        let y = 0.01 * x_f.powi(3) - 0.5 * x_f + 20.0 + noise;

        data.push((x_f, y));
    }

    save_data_to_csv(&data, "data.csv")?;
    Ok(())
}

fn save_data_to_csv(data: &Vec<(f32, f32)>, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    // Optional: write header
    writeln!(writer, "x,y")?;

    for (x, y) in data {
        writeln!(writer, "{},{}", x, y)?;
    }

    println!("Data saved to {}", filename);
    Ok(())
}
