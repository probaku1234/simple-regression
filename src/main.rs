mod data_generator;

use crate::data_generator::generate_and_save_data;

fn main() {
    generate_and_save_data(100000).expect("Failed to generate and save data.");
}
