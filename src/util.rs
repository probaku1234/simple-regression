use std::error::Error;
use std::fs::File;
use std::io::BufRead;

pub fn read_data_from_csv(filename: &str) -> Result<Vec<(f32, f32)>, Box<dyn Error>> {
    let mut data = Vec::new();
    let file = File::open(filename)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("x,y") { // Skip header
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