//! Matrix<T> data structure used to hold the data

use std::{fmt::Debug, iter::Sum};

use num::Float;

use crate::calculations::linear_algebra::dot_product;

/// A Matrix<T> is a data structure holding the data
///
/// This data structure is a key component of the project as 
/// most of the operations are performed on it. 
pub struct Matrix<T> {
    pub content: Vec<T>,
    pub rows: usize,
    pub columns: usize,
}

impl<T: Float + Sized + Debug + Copy + Send + Sync + Sum + 'static> Matrix<T> {
    /// Initialise an empty matrix
    pub fn empty() -> Matrix<T> {
        Matrix {
            content: Vec::new(),
            rows: 0,
            columns: 0,
        }
    }

    /// Initialize an array from a one dimensional vector 
    pub fn from_1d_vector(content: Vec<T>, rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            content,
            rows,
            columns,
        }
    }

    /// Initialize an array from a two dimensional vector 
    pub fn from_2d_vector(content: Vec<Vec<T>>) -> Matrix<T> {
        let mut content_1d: Vec<T> = Vec::with_capacity(content.len() * content[0].len());

        content.iter().for_each(|row| {
            row.iter().for_each(|item| content_1d.push(*item));
        });

        Matrix {
            content: content_1d,
            rows: content.len(),
            columns: content[0].len(),
        }
    }

    /// Initilizes an idendity matrix
    pub fn identity(dim: usize) -> Result<Matrix<T>, String> {

        if dim == 0 {
            return Err("We cannot create a 0 dimensional idendity matrix. Please use Matrix::empty() instead".to_string());
        }

        let mut identity = vec![T::from(0.0).unwrap(); dim.pow(2)];

        (0..dim).fold(0, |prev,_| {
            identity[prev] = T::one();
            println!("{identity:?}");
            prev + dim + 1
        });

        Ok(Matrix {
            content: identity,
            rows: dim,
            columns: dim,
        })
    }

    /// Initialize a matrix containing zeros
    pub fn zeroes(rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            content: vec![T::from(0.0).unwrap(); rows * columns],
            rows,
            columns,
        }
    }

    /// Returns the matrix in a 
    pub fn get_2d_vector(&self) -> Vec<Vec<T>> {
        let mut v: Vec<Vec<T>> = Vec::with_capacity(self.rows);
        self.content
            .chunks(self.columns)
            .for_each(|row| v.push(row.to_vec()));
        v
    }

    /// Inline multiplies the matrix by the matrix b
    pub fn multiply_matrix(&mut self, b: &Matrix<T>) -> Result<u8, String> {
      
        let mut new_matrix: Vec<T> = vec![T::from(0.0).unwrap(); self.rows * b.columns];

        if self.columns != b.rows {
            return Err(format!("Cannot multiply ({}, {}) and ({}, {})", self.rows, self.columns, b.rows, b.columns));
        }

        let mut idx = 0;

        (0..self.rows).for_each(|row| {
            (0..b.columns).for_each(|column| {
                let col = b.get_col(column);
                new_matrix[idx] = dot_product(&self[row], &col.content).unwrap();
                idx += 1;
            });
        });

        self.content = new_matrix;
        self.columns = b.columns;

        Ok(1)
    }

    /// Returns the transpose
    pub fn get_transpose(&self) -> Matrix<T> {
        let mut transposed: Vec<T> = vec![T::from(0.0).unwrap(); self.rows * self.columns];

        (0..self.columns).for_each(|col| {
            (0..self.rows).for_each(|row| {
                transposed[self.rows * col + row] = self.content[self.columns * row + col]
            })
        });

        Matrix {
            content: transposed,
            columns: self.rows,
            rows: self.columns,
        }
    }

    pub fn split(&mut self) {
        
        let mut sub_matrices: Vec<Matrix<T>> = vec![Matrix::empty(), Matrix::empty(), Matrix::empty(), Matrix::empty()];
        let column_limit: usize = self.columns/2;

        (0..self.rows).for_each(|row| {
            if row < self.rows/2 {
                sub_matrices[0].append_vector(&self[row][..column_limit], 0);
                sub_matrices[1].append_vector(&self[row][..column_limit], 0);

            }
        });
    }

    /// Sums the matrix either column or row wise
    pub fn sum(&self, axis: u8) -> Result<Matrix<T>, &str> {
        // Column wise
        if axis == 0 {
            let mut sums: Matrix<T> = Matrix::zeroes(1, self.columns);

            for col in 0..self.columns {
                (0..self.rows).for_each(|row| {
                    let current_row = (row * self.columns) + col;
                    sums.content[col] = sums.content[col] + self.content[current_row];
                });
            }

            return Ok(sums);
        }

        // Row wise
        if axis == 1 {
            let mut sums: Matrix<T> = Matrix::zeroes(1, self.rows);

            for i in 0..self.rows {
                let s = self[i].iter().fold(T::from(0.0).unwrap(), |a, &b| a + b);
                sums.content[i] = s;
            }

            return Ok(sums);
        }

        Err("Can only compute the sum column (0) or row wise (1)")
    }

    /// Computes the row or column wise mean
    pub fn mean(&self, axis: u8) -> Result<Matrix<T>, &str> {
        // column wise
        if axis == 0 {
            let mut sums: Matrix<T> = self.sum(axis).unwrap();
            println!("Sums: {:?}", sums.content);
            sums.divide_by_scalar(T::from(self.rows).unwrap());
            return Ok(sums);
        }

        // Row wise
        if axis == 1 {
            let mut sums: Matrix<T> = self.sum(axis).unwrap();
            sums.divide_by_scalar(T::from(self.columns).unwrap());
            return Ok(sums);
        }

        Err("Wrong axis: pass 0 for column wise and 1 for row wise")
    }

    /// Append a vector column or row wise
    ///
    /// Row wise: 0 Column wise: 1
    pub fn append_vector(&mut self, to_add: &[T], axis: u8) -> Result<u8, &str> {
        if axis > 1 {
            return Err("The axis has to be 0 (rows) or 1 (columns)");
        }

        if self.rows == 0 && axis == 0 {
            self.columns = to_add.len();
            self.content = to_add.to_vec();
            self.rows = 1;

            return Ok(1);
        }

        if self.rows == 0 && axis == 1 {
            self.rows = to_add.len();
            self.columns = 1;
            self.content = to_add.to_vec();

            return Ok(1);
        }

        // Add row
        if axis == 0 {
            if to_add.len() != self.columns {
                return Err("vector size does not match the number of columns");
            }

            self.rows += 1;

            if self.columns == 0 {
                self.columns = to_add.len();
            }

            to_add.iter().for_each(|entry| self.content.push(*entry));
        }

        // Add column
        if axis == 1 {
            if to_add.len() != self.rows {
                return Err("vector size does not match the number of columns");
            }

            let mut new = Vec::with_capacity((self.rows * self.columns) + to_add.len());
            let mut to_add_index = 0;

            for i in 0..(self.rows * self.columns) {
                if (i) % self.columns == 0 && i != 0 {
                    new.push(to_add[to_add_index]);
                    to_add_index += 1;
                }

                new.push(self.content[i]);
            }

            new.push(to_add[to_add_index]);

            self.content = new;
            self.columns += 1;

            if self.rows == 0 {
                self.rows = to_add.len();
            }
        }

        Ok(1)
    }


    /// Removes a column or a row at the given index
    ///
    /// Row: 0, Column: 1
    pub fn remove(&mut self, index: usize, axis: usize) -> Result<Matrix<T>, String> {
        if axis > 1 {
            return Err("The axis has to be 0 (rows) or 1 (columns)".to_string());
        }

        if axis == 0 && index > self.rows {
            let msg = format!(
                "The matrix contains {:?} rows but it was asked to remove row {:?}",
                self.rows, index
            );
            return Err(msg);
        }

        if axis == 1 && index > self.columns {
            let msg = format!(
                "The matrix contains {:?} columns but it was asked to remove  column {:?}",
                self.rows, index
            );
            return Err(msg);
        }

        if axis == 0 {
            let begin_to_remove = self.columns * index;
            let end_to_remove = self.columns * (index + 1);

            let mut new_vector: Vec<T> = self.content[..begin_to_remove].to_vec();
            new_vector.append(&mut self.content[end_to_remove..].to_vec());

            let removed = self.content[begin_to_remove..end_to_remove].to_vec();

            self.content = new_vector;
            self.rows -= 1;

            return Ok(Matrix::from_1d_vector(removed, 1, self.columns));
        }

        if axis == 1 {
            let mut new: Vec<T> = Vec::with_capacity((self.rows * self.columns) - self.columns);
            let mut removed: Vec<T> = Vec::with_capacity(self.columns);

            (0..self.rows).for_each(|row_index| {
                self[row_index].iter().enumerate().for_each(|(idx, item)| {
                    if idx != index {
                        new.push(item.to_owned());
                    } else {
                        removed.push(item.to_owned());
                    }
                });
            });

            self.content = new;
            self.columns -= 1;

            return Ok(Matrix::from_1d_vector(removed, self.rows, 1));
        }

        Err("An unforseen error happened".to_string())
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn multiply_by_scalar(&mut self, scalar: T) {
        (0..self.rows * self.columns)
            .for_each(|idx| self.content[idx] = self.content[idx] * scalar);
    }

    pub fn divide_by_scalar(&mut self, scalar: T) {
        (0..self.rows * self.columns)
            .for_each(|idx| self.content[idx] = self.content[idx] / scalar);
    }

    pub fn get_row(&self, row: usize) -> Vec<T> {
        self.content[self.columns * row..self.columns * (row + 1)].to_vec()
    }

    pub fn get_col(&self, col: usize) -> Matrix<T> {
        let col = (0..self.rows)
            .map(|row| {
                let current_row = (row * self.columns) + col;
                self.content[current_row]
            })
            .collect();

        Matrix::from_1d_vector(col, self.rows, 1)
    }

    pub fn exp(&mut self) {
        let e: T = T::from(std::f64::consts::E).unwrap();

        (0..self.content.len()).for_each(|idx| self.content[idx] = T::powf(self.content[idx], e));
    }

}
