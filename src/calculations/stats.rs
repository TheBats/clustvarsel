use std::fmt::Debug;
use std::{iter::Sum, ops::Mul};

use num::Float;

use crate::calculations::linear_algebra::dot_product;

use super::linear_algebra::determinant;
use super::{linear_algebra::slow_inverse_matrix, matrix::Matrix};

/// Computes the covariance
pub fn covariance<T: Float>(a: &Matrix<T>, zs: &Vec<u8>) -> Matrix<T>
where
    T: Mul + Sum + Debug + Send + Sync + 'static,
{
    let mut cov_ma: Matrix<T> = Matrix::zeroes(a.columns, a.columns);
    let means = means_zs(a, zs).unwrap();

    let number_attributes: u32 = zs.iter().fold(0u32, |acc, i| acc + *i as u32);
    let number_attributes = T::from(number_attributes).unwrap();

    (0..a[0].len()).for_each(|col1| {
        (0..a[0].len()).for_each(|col2| {
            let sum: T = (0..a.len())
                .map(|row| {
                    if zs[row] == 1 {
                        (a[row][col1] - means[col1]).mul(a[row][col2] - means[col2])
                    } else {
                        T::from(0.0).unwrap()
                    }
                })
                .sum();

            cov_ma[col1][col2] = sum / (number_attributes - T::from(1.0).unwrap());
            cov_ma[col2][col1] = sum / (number_attributes - T::from(1.0).unwrap());
        })
    });

    cov_ma
}

/// Computes the covariance but takes the z value into account
pub fn covariance_no_z<T: Float>(a: &Matrix<T>, gammas: &[T], means: &[T]) -> Matrix<T>
where
    T: Mul + Sum + Debug + Send + Sync + 'static,
{
    let mut cov_ma: Matrix<T> = Matrix::zeroes(a.columns, a.columns);

    (0..a[0].len()).for_each(|col1| {
        (0..a[0].len()).for_each(|col2| {
            let sum: T = (0..a.len())
                .map(|row| {
                    gammas[row].mul((a[row][col1] - means[col1]).mul(a[row][col2] - means[col2]))
                })
                .sum();

            cov_ma[col1][col2] = sum; // /(number_attributes-T::from(1.0).unwrap());
            cov_ma[col2][col1] = sum; // /(number_attributes-T::from(1.0).unwrap());
        })
    });

    cov_ma
}

/// Computes the column wise means by taking the z values into account
pub fn means_zs<T: Float>(a: &Matrix<T>, zs: &Vec<u8>) -> Result<Vec<T>, &'static str>
where
    T: Send + Sync + Debug + Sum + 'static,
{
    let number_attributes = zs.iter().fold(0u32, |acc, i| acc + *i as u32);
    let number_attributes = T::from(number_attributes).unwrap();

    let means = (0..a[0].len())
        .map(|attr| {
            let col: T = (0..a.len())
                .zip(zs)
                .map(|(row, z)| match z {
                    1 => a[row][attr],
                    0 => T::zero(),
                    _ => panic!("z can only be 0 or 1"),
                })
                .sum();

            col / number_attributes
        })
        .collect();

    Ok(means)
}

/// Computes the likelihood the given gaussian generated the data
pub fn slow_multivariate_gaussian<T: Float>(
    data: &Matrix<T>,
    covariance: &Matrix<T>,
    means: &[T],
) -> Result<Vec<T>, &'static str>
where
    T: Send + Sync + Debug + Sum + 'static,
{
    let pi: T = T::from(std::f64::consts::PI).unwrap();
    let two: T = T::from(2.0).unwrap();
    let number_rows: T = T::from(covariance.len()).unwrap();

    let determinant: T = match determinant(covariance) {
        Ok(det) => det,
        Err(err) => return Err(err),
    };

    let const1: T = T::powf(two * pi, -number_rows / two);
    let const2: T = T::powf(determinant, -T::from(0.5).unwrap());
    let const_final = const1.mul(const2);

    let inverse_covariance = slow_inverse_matrix(covariance);

    let mut copy_data = data.clone();

    means.iter().enumerate().for_each(|(col, mean)| {
        (0..copy_data.rows).for_each(|row| copy_data[row][col] = copy_data[row][col] - *mean);
    });

    let covariance_cols: Vec<Vec<T>> = (0..covariance.rows)
        .map(|idx| inverse_covariance.get_col(idx).content)
        .collect();

    let likelihoods: Vec<T> = (0..data.rows)
        .map(|row| {
            let data_times_inv_conv: Vec<T> = covariance_cols
                .iter()
                .map(|cov| dot_product(&copy_data[row], cov).unwrap())
                .collect();
            let times_data: T = dot_product(&data_times_inv_conv, &copy_data[row]).unwrap()
                / T::from(-2.0).unwrap();
            const_final * T::exp(times_data)
        })
        .collect();

    Ok(likelihoods)
}
