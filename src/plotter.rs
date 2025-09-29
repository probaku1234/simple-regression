use crate::model::ARTIFACT_DIR;
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn plot_loss(
    run_name: &str,
    log_file_name: &str,
    path_png: &str,
    caption: &str,
    y_desc: &str,
    step: usize,
) -> Result<(), Box<dyn Error>> {
    // Read all losses
    let log_file_path = format!("{ARTIFACT_DIR}/{run_name}/{log_file_name}");
    let file = File::open(log_file_path)?;
    let reader = BufReader::new(file);

    let losses: Vec<f32> = reader
        .lines()
        .filter_map(|line| line.ok()?.parse::<f32>().ok())
        .collect();

    if losses.is_empty() {
        return Err("No loss values found".into());
    }

    let ymin = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    let ymax = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let root = BitMapBackend::new(path_png, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..losses.len(), ymin..ymax)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc(y_desc)
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..losses.len()).map(|i| (i * step, losses[i])),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_predictions_vs_expectations(
    predictions: &[f32],
    expectations: &[f32],
    path: &str,
) -> Result<(), Box<dyn Error>> {
    if predictions.len() != expectations.len() {
        return Err("predictions and expectations must have the same length".into());
    }

    let n = predictions.len();
    if n == 0 {
        return Err("empty data".into());
    }

    // Find Y range
    let ymin = predictions
        .iter()
        .chain(expectations.iter())
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let ymax = predictions
        .iter()
        .chain(expectations.iter())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let pad_y = (ymax - ymin).abs().max(1e-3) * 0.1;

    // Set up chart
    let root = BitMapBackend::new(path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Prediction vs Expectation", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..n, (ymin - pad_y)..(ymax + pad_y))?;

    chart
        .configure_mesh()
        .x_desc("Sample Index")
        .y_desc("Value")
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    // Plot true line (red)
    chart
        .draw_series(LineSeries::new((0..n).map(|i| (i, expectations[i])), &RED))?
        .label("True")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // Plot prediction line (blue)
    chart
        .draw_series(LineSeries::new((0..n).map(|i| (i, predictions[i])), &BLUE))?
        .label("Pred")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
