use std::{error::Error, fs::File};

use crate::calculations::matrix::Matrix;

/// Reads a CSV and parses it into a Matrix<T> struct
pub fn read_parse(path: &str) -> Result<(Vec<String>, Matrix<f64>), Box<dyn Error>> {
    let file = File::open(path)?;

    let mut rdr = csv::Reader::from_reader(&file);
    let unparsed_headers = rdr.headers()?;

    let headers: Vec<String> = unparsed_headers
        .iter()
        .map(|attr| attr.to_string())
        .collect();

    let mut matrix: Matrix<f64> = Matrix::empty();
    matrix.columns = headers.len();

    rdr.records().for_each(|f| {
        match f {
            Ok(record) => {
                record.iter().for_each(|entry| {
                    match entry.parse::<f64>() {
                        Ok(entr) => matrix.content.push(entr), //new_row.push(entr),
                        Err(_) => panic!("Could not parse {entry:?} to a f64"),
                    }
                });
            }
            Err(_) => panic!("Nothing to read"),
        }

        matrix.rows += 1;
    });

    println!("Rows: {:?} Columns {:?}", matrix.rows, matrix.columns);

    Ok((headers, matrix))
}
