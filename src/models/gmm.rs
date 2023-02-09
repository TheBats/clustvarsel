use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use num::Float;

use super::{distances, kmeans};

use crate::calculations::matrix::Matrix;
use crate::calculations::stats::{
    means_zs, covariance_no_z, slow_multivariate_gaussian, covariance,
};
use crate::helpers::mean_squared_error;

/// Struct containing all the information about the Gaussian Mixture Model
pub struct GaussianMixtureModel<T> {
    means: Matrix<T>,
    covariance_matrices: Vec<Matrix<T>>,
    k: usize,
    seed: u64,
    mixtures: Vec<T>,
    tolerance: T,
    pub final_difference: f32,
    max_steps: i32,
    pub steps: u64,
    pub gammas: Matrix<T>,
}

impl<
        T: Float
            + Div
            + Mul
            + Add
            + Sub<Output = T>
            + Sum<<T as Mul>::Output>
            + Copy
            + Debug
            + Sync
            + Send
            + PartialOrd
            + 'static,
    > GaussianMixtureModel<T>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
    T: FromStr,
    Vec<T>: FromIterator<<T as Div>::Output>,
{
    /// Returns a new GaussianMixtureModel struct
    pub fn new(
        k: usize,
        seed: u64,
        mixtures: Vec<T>,
        max_steps: i32,
        tolerance: T,
    ) -> GaussianMixtureModel<T> {
        let m: Matrix<T> = Matrix::empty();

        Self {
            means: m,
            covariance_matrices: Vec::new(),
            k,
            seed,
            mixtures,
            tolerance,
            final_difference: 1000.0,
            max_steps,
            steps: 0,
            gammas: Matrix::empty(),
        }
    }

    /// Fits the Gaussian Mixture Model to the data using EM
    pub fn fit(
        &mut self,
        data: &Matrix<T>,
        distance: distances::Distance<T>,
    ) -> Result<&'static str, &'static str> {
        let mut init = kmeans::Kmeans::init(self.k, data, self.seed, distance);
        init.fit(data);

        let curr_z = init.final_z;

        let mut covariances: Vec<Matrix<T>> = Vec::new();
        let mut means: Matrix<T> = Matrix::empty();

        for z in curr_z {
            let cov: Matrix<T> = covariance(data, &z);
            covariances.push(cov);
            let m = means_zs(data, &z).unwrap();
            means.append_vector(&m, 0).expect("Unable to append row");
        }

        self.means = means;
        self.covariance_matrices = covariances;

        let mut counter = 0;

        // EM Loop
        loop {
            let mut gammas: Matrix<T> = Matrix::zeroes(self.k, data.rows);

            // E step
            for i in 0..self.k {
                let tmp_gammas =
                    slow_multivariate_gaussian(data, &self.covariance_matrices[i], &self.means[i])
                        .unwrap();
                tmp_gammas
                    .iter()
                    .enumerate()
                    .for_each(|(idx, gamma)| gammas[i][idx] = *gamma * self.mixtures[i]);
            }

            let sum_gammas_0 = gammas.sum(0).unwrap();

            let mut new_means: Matrix<T> = Matrix::zeroes(self.means.rows, self.means.columns);

            // M step
            data.into_iter()
                .enumerate()
                .for_each(|(idx_data_point, dp)| {
                    // Normalize Gammas for data point dp
                    for i in 0..gammas.rows {
                        gammas[i][idx_data_point] =
                            gammas[i][idx_data_point] / sum_gammas_0[0][idx_data_point];
                    }

                    // Update the means
                    (0..new_means.columns).for_each(|col| {
                        (0..new_means.rows).for_each(|row| {
                            new_means[row][col] =
                                new_means[row][col] + (dp[col] * gammas[row][idx_data_point]);
                        });
                    });
                });

            let mut nk = gammas.sum(1).unwrap();

            // Normalize the means
            for row in 0..self.k {
                for col in 0..new_means.columns {
                    new_means[row][col] = new_means[row][col] / nk[0][row];
                }
            }

            let mut covs: Vec<Matrix<T>> = Vec::with_capacity(self.k);

            // Covariance calculation
            for i in 0..self.k {
                let mut cov: Matrix<T> = covariance_no_z(data, &gammas[i], &new_means[i]);
                cov.divide_by_scalar(nk[0][i]);
                covs.push(cov);
            }

            counter += 1;

            let mean_error = mean_squared_error(&self.means.content, &new_means.content).unwrap();

            if counter == self.max_steps {
                return Err("Did not converge");
            }

            if mean_error < self.tolerance {
                self.gammas = gammas;
                self.final_difference = T::to_f32(&mean_error).unwrap();
                return Ok("Converged");
            }

            self.means = new_means;
            self.covariance_matrices = covs;
            nk.divide_by_scalar(T::from(data.rows).unwrap());
            self.mixtures = nk.content;
            self.steps += 1;
        }
    }
}
