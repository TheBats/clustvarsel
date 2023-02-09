use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::{zip, Sum};
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use num::Float;

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::calculations::matrix::Matrix;

use super::distances::{self, eucleadian_distance};

/// Struct containing the data needed for Kmeans
pub struct Kmeans<T: 'static> {
    pub centroids: Vec<Vec<T>>,
    k: usize,
    distance: distances::Distance<T>,
    pub final_z: Vec<Vec<u8>>,
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
            + PartialOrd,
    > Kmeans<T>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
    T: FromStr,
    Vec<T>: FromIterator<<T as Div>::Output>,
{
    /// Initializes a new Kmeans struct 
    ///
    /// It runs Kmeans++ to find the best starting centroids
    pub fn init(
        k: usize,
        data: &Matrix<T>,
        seed: u64,
        distance: distances::Distance<T>,
    ) -> Kmeans<T> {
        // Build distance matrix based on indices.
        // We get a vector of (index, distance)
        let distances: Vec<Vec<(usize, T)>> = (0..data.rows)
            .into_par_iter()
            .map(|f| {
                let mut x: Vec<(usize, T)> =
                    zip(0..data.rows, (0..data.len()).collect::<Vec<usize>>())
                        .map(|r| (r.1, eucleadian_distance(&data[f], &data[r.0]).unwrap()))
                        .collect();

                x.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                x
            })
            .collect();

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut nodes: Vec<usize> = vec![rng.gen_range(0..data.len())];

        if k > 1 {
            for _ in 1..k {
                let mut bins: Vec<(usize, usize)> = (0..data.len()).map(|f| (f, 0)).collect();

                for node in &mut nodes {
                    for (ind, obj) in distances[*node].iter().enumerate() {
                        bins[obj.0].1 += ind;
                    }
                }

                bins.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                nodes.push(bins[0].0)
            }
        }

        Self {
            centroids: nodes
                .iter()
                .map(|ind| data[ind.to_owned()].to_vec())
                .collect(),
            k,
            distance,
            final_z: Vec::new(),
        }
    }

    /// Fits Kmeans to the data
    pub fn fit(&mut self, data: &Matrix<T>) {
        let len_data: usize = data.len();
        let mut z: Vec<Vec<u8>> = (0..self.k)
            .map(|_| (0..len_data).map(|_| 0).collect())
            .collect();

        let mut prev_z: Vec<Vec<u8>> = (0..self.k)
            .map(|_| (0..len_data).map(|_| 0).collect())
            .collect();

        let mut stable = false;

        while !stable {
            stable = true;

            // Update assignments
            (0..data.rows).enumerate().for_each(|(ind, point)| {
                let mut closest = (
                    0,
                    (self.distance)(&data[point], &self.centroids[0])
                        .expect("Kmeans: Error in distance computation"),
                );
                (1..self.k).for_each(|ind| {
                    let tmp_dist = (self.distance)(&data[point], &self.centroids[ind])
                        .expect("Kmeans: Error in distance computation");
                    if let Some(Ordering::Less) = tmp_dist.partial_cmp(&closest.1) {
                        closest = (ind, tmp_dist);
                    }
                });

                z[closest.0][ind] = 1;

                if prev_z[closest.0][ind] == 0u8 && stable {
                    stable = false;
                }
            });

            // Update centroids coordinates
            z.iter().enumerate().for_each(|(ind, centroid_z)| {
                let mut total: T = T::zero();
                let mut new_coords: Vec<T> = (0..self.centroids[0].len())
                    .map(|_| T::from(0.0).unwrap())
                    .collect();

                centroid_z.iter().enumerate().for_each(|(ind, val)| {
                    if val == &1 {
                        total = total + T::one();
                        data[ind]
                            .iter()
                            .enumerate()
                            .for_each(|(i, attr)| new_coords[i] = new_coords[i] + *attr);
                    }
                });

                self.centroids[ind] = new_coords
                    .iter()
                    .map(|val| val.to_owned() / total)
                    .collect::<Vec<T>>();
            });

            prev_z = z;
            z = (0..self.k)
                .map(|_| (0..len_data).map(|_| 0).collect())
                .collect();
        }

        self.final_z = prev_z;
    }
}
