use crate::calculations::matrix::Matrix;

use std::fmt::Debug;
use std::iter::{zip, Sum};

use num::Float;
use rayon::prelude::*;

/// Computes the dot product between two Matrix<T> structs and returns the result
pub fn matrix_dot_product<T: Float + Sum + Send + Debug + Sync + 'static>(
    a: &Matrix<T>,
    b: &Matrix<T>,
) -> Result<Matrix<T>, &'static str> {
    if a.columns != b.rows {
        return Err("The vector size do not match");
    }

    println!("A: {a:?}");
    println!("B: {b:?}");

    let mut new: Matrix<T> = Matrix::zeroes(a.rows, b.columns);

    (0..a.rows).for_each(|rows_a| {
        let r = &a[rows_a];

        (0..b.columns).for_each(|column| {
            let col = b.get_col(column);
            match dot_product(r, &col.content) {
                Ok(res) => {
                    new[rows_a][column] = res;
                }
                Err(e) => {
                    panic!("matrix_dot_product: {e}");
                }
            }
        });
    });

    println!("New: {new:?}");
    Ok(new)
}

/// Computes the dot product between two &[T] and returns the result
pub fn dot_product<T: Float + Sum>(a: &[T], b: &[T]) -> Result<T, &'static str> {
    if a.len() != b.len() {
        return Err("The vector size do not match");
    }

    let res: T = zip(a, b).fold(T::zero(), |val, entry| val + *entry.0 * *entry.1);

    Ok(res)
}

/// Divide and Conquer approach to transposition using parallelisation
///
/// Implementation not finished
pub fn recursive_step<T: Float + Send + Sync>(
    to_transp: &Vec<Vec<T>>,
    range: Vec<usize>,
    mut level: usize,
    number_threads: usize,
) -> Vec<Vec<T>> {
    if level == number_threads || range.last().unwrap() / 2usize <= level {
        let res: Vec<Vec<T>> = range
            .into_par_iter()
            .map(|col| {
                (0..to_transp.len())
                    .map(|row| to_transp[row][col])
                    .collect::<Vec<T>>()
            })
            .collect();

        res
    } else {
        let chunck_size = (range.len() / 2usize) + 1;
        let mut new_vec: Vec<Vec<T>> = Vec::with_capacity(range.len());
        level += 1;

        range.chunks(chunck_size).for_each(|chunck| {
            let res = recursive_step(to_transp, chunck.to_vec(), level, number_threads);
            res.into_iter().for_each(|row| new_vec.push(row));
        });

        new_vec
    }
}

/// Inverse the matrix
pub fn slow_inverse_matrix<T: Float + Debug + 'static + Send + Sync + Sum>(
    to_inverse: &Matrix<T>,
) -> Matrix<T> {
    let mut inverse_matrix = Matrix::identity(to_inverse.columns).unwrap();
    let mut tmp_to_inverse = to_inverse.clone();

    inplace_upper_triangle_matrix(&mut tmp_to_inverse, &mut inverse_matrix);
    inplace_lower_triangle_matrix(&mut tmp_to_inverse, &mut inverse_matrix);

    (0..tmp_to_inverse.len()).for_each(|ind| {
        if tmp_to_inverse[ind][ind] != T::from(1.0).unwrap() {
            let ratio = T::from(1.0).unwrap() / tmp_to_inverse[ind][ind];

            (0..tmp_to_inverse[ind].len()).for_each(|it| {
                tmp_to_inverse[ind][it] = tmp_to_inverse[ind][it] * ratio;
                inverse_matrix[ind][it] = inverse_matrix[ind][it] * ratio;
            });
        }
    });

    inverse_matrix
}

// Inplace upper triangle computation
pub fn inplace_upper_triangle_matrix<T: Float + 'static + Debug + Send + Sync + Sum>(
    tmp_to_inverse: &mut Matrix<T>,
    inverse_matrix: &mut Matrix<T>,
) {
    (0..tmp_to_inverse.len()).for_each(|pivot_col: usize| {
        (pivot_col..tmp_to_inverse.len()).for_each(|current_row| {
            if pivot_col != current_row
                && tmp_to_inverse[current_row][pivot_col] != T::from(0.0).unwrap()
            {
                let ratio =
                    tmp_to_inverse[current_row][pivot_col] / tmp_to_inverse[pivot_col][pivot_col];

                (0..tmp_to_inverse[current_row].len()).for_each(|it| {
                    tmp_to_inverse[current_row][it] =
                        tmp_to_inverse[current_row][it] - ratio * tmp_to_inverse[pivot_col][it];
                    inverse_matrix[current_row][it] =
                        inverse_matrix[current_row][it] - ratio * inverse_matrix[pivot_col][it];
                });
            }
        });
    });
}

/// Inplace lower triangle computation
pub fn inplace_lower_triangle_matrix<T: Float + 'static + Debug + Send + Sync + Sum>(
    tmp_to_inverse: &mut Matrix<T>,
    inverse_matrix: &mut Matrix<T>,
) {
    (0..tmp_to_inverse.len())
        .rev()
        .for_each(|pivot_col: usize| {
            (0..tmp_to_inverse.len()).rev().for_each(|current_row| {
                if pivot_col != current_row
                    && tmp_to_inverse[current_row][pivot_col] != T::from(0.0).unwrap()
                {
                    let ratio = tmp_to_inverse[current_row][pivot_col]
                        / tmp_to_inverse[pivot_col][pivot_col];

                    (0..tmp_to_inverse[current_row].len()).for_each(|it| {
                        tmp_to_inverse[current_row][it] =
                            tmp_to_inverse[current_row][it] - ratio * tmp_to_inverse[pivot_col][it];
                        inverse_matrix[current_row][it] =
                            inverse_matrix[current_row][it] - ratio * inverse_matrix[pivot_col][it];
                    });
                }
            });
        });
}

/// Performs LU decomposition
pub fn lu_decomposition_matrix<T: Float + Debug + 'static + Send + Sync + Sum>(
    to_decompose: &Matrix<T>,
) -> Result<(Matrix<T>, Matrix<T>), &'static str> {
    let mut lower: Matrix<T> = Matrix::zeroes(to_decompose.len(), to_decompose[0].len());
    let mut upper: Matrix<T> = to_decompose.clone();

    (0..to_decompose.len()).for_each(|pivot_col: usize| {
        lower[pivot_col][pivot_col] = T::from(1.0).unwrap();
        (pivot_col..to_decompose.len()).for_each(|current_row| {
            if pivot_col != current_row && upper[current_row][pivot_col] != T::from(0.0).unwrap() {
                let ratio = upper[current_row][pivot_col] / upper[pivot_col][pivot_col];

                (0..to_decompose[current_row].len()).for_each(|it| {
                    upper[current_row][it] = upper[current_row][it] - ratio * upper[pivot_col][it];
                });

                lower[current_row][pivot_col] = ratio;
            }
        });
    });

    Ok((lower, upper))
}

/// Computes the determinant using LU decomposition
pub fn determinant<T: Float + Debug + Send + Sync + 'static + Sum>(
    to_compute: &Matrix<T>,
) -> Result<T, &'static str> {
    if to_compute.len() != to_compute[0].len() {
        return Err("Matrix is not squared");
    }

    match lu_decomposition_matrix(to_compute) {
        Ok((res_lower, res_upper)) => {
            let mut det_upper = res_upper[0][0];
            let mut det_lower = res_lower[0][0];

            for i in 1..res_upper.len() {
                det_upper = det_upper * res_upper[i][i];
                det_lower = det_lower * res_lower[i][i];
            }

            Ok(det_lower * det_upper)
        }

        Err(res) => Err(res),
    }
}