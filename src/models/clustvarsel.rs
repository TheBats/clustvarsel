use num::Float;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use super::distances::{eucleadian_distance, Distance};
use super::gmm::GaussianMixtureModel;

use crate::calculations::matrix::Matrix;
use crate::training_setup::TrainingSetup;

/// Struct storing the information need for the ClustVarSel algorithm
pub struct CLUSTVARSEL<T> {
    pub final_selection: Vec<usize>,
    pub best_bic: T,
    training_setup: TrainingSetup<T>,
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
    > CLUSTVARSEL<T>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
    T: FromStr,
    Vec<T>: FromIterator<<T as Div>::Output>,
{
    /// Returns a new CLUSTVARSEL struct
    pub fn new(
        number_clusters: usize,
        seed: u64,
        tolerance: T,
        max_steps: i32,
        initial_mixtures: Vec<T>,
        verbose: bool,
        cores: usize,
    ) -> CLUSTVARSEL<T> {
        CLUSTVARSEL {
            final_selection: Vec::new(),
            best_bic: T::from(0.0).unwrap(),
            training_setup: TrainingSetup {
                number_clusters,
                seed,
                tolerance,
                max_steps,
                initial_mixtures,
                verbose,
                cores,
            },
        }
    }  

    /// Performs selection and fitting on the data
    ///
    /// This is the main loop performing attribute selection and removal
    /// Once finished, the struct contains the selected attributes
    pub fn fit(&mut self, data: Matrix<T>) -> Result<String, String> {
        let mut selected_columns = Vec::with_capacity(data.columns);
        let mut current_matrix: Matrix<T> = Matrix::empty();

        loop {
            let mut added: bool = false;
            let mut removed: bool = false;

            let mut bic_add: Vec<T> = Vec::new();

            // Addition step
            (0..data.columns).into_iter().for_each(|to_add| {
                if !selected_columns.contains(&to_add) {
                    let mut local_matrix: Matrix<T> = current_matrix.clone();
                    local_matrix
                        .append_vector(&data.get_col(to_add).content, 1)
                        .expect("Unable to append column");

                    let mut gmm = GaussianMixtureModel::new(
                        self.training_setup.number_clusters,
                        self.training_setup.seed,
                        self.training_setup.initial_mixtures.clone(),
                        self.training_setup.max_steps,
                        self.training_setup.tolerance,
                    );

                    let distance: Distance<T> = eucleadian_distance;

                    match gmm.fit(&local_matrix, distance) {
                        Ok(msg) => {
                            if self.training_setup.verbose {
                                println!(
                                    "- {}",
                                    msg.to_owned()
                                        + " adding: "
                                        + &to_add.to_string()
                                        + "\n Steps: "
                                        + &gmm.steps.to_string()
                                        + "\n Final Difference: "
                                        + &gmm.final_difference.to_string()
                                )
                            }
                        }
                        Err(msg) => {
                            let message: String = msg.to_owned()
                                + "\n Steps: "
                                + &gmm.steps.to_string()
                                + "\n Final Difference: "
                                + &gmm.final_difference.to_string();
                            panic!("{}", message)
                        }
                    };

                    let b = self.bic(gmm, local_matrix);

                    println!(" BIC: {b:?}");

                    bic_add.push(b);
                } else {
                    bic_add.push(T::from(100).unwrap());
                }
            });

            let argmin_bic_add = bic_add.iter().enumerate().fold(
                (0usize, T::from(100).unwrap()),
                |current, possible_new| {
                    if &current.1 > possible_new.1 {
                        (possible_new.0, possible_new.1.to_owned())
                    } else {
                        current
                    }
                },
            );

            if argmin_bic_add.1 < self.best_bic {
                println!("-- Adding {:?}--", argmin_bic_add.0);
                selected_columns.push(argmin_bic_add.0 as usize);
                current_matrix
                    .append_vector(&data.get_col(argmin_bic_add.0 as usize).content, 1)
                    .expect("Unable to append column");

                self.best_bic = argmin_bic_add.1;
                added = true;
            }

            let mut bic_remove: Vec<T> = Vec::with_capacity(selected_columns.len());

            // Removal step
            if selected_columns.len() > 1 {
                selected_columns.iter().for_each(|attr_index| {
                    let mut local_matrix = current_matrix.clone();
                    local_matrix
                        .remove(attr_index.to_owned(), 1)
                        .expect("Unable to remove a column");

                    let mut gmm = GaussianMixtureModel::new(
                        self.training_setup.number_clusters,
                        self.training_setup.seed,
                        self.training_setup.initial_mixtures.clone(),
                        self.training_setup.max_steps,
                        self.training_setup.tolerance,
                    );

                    let distance: Distance<T> = eucleadian_distance;

                    match gmm.fit(&local_matrix, distance) {
                        Ok(msg) => {
                            if self.training_setup.verbose {
                                println!(
                                    "{}",
                                    msg.to_owned()
                                        + "\n Steps: "
                                        + &gmm.steps.to_string()
                                        + "\n Final Difference: "
                                        + &gmm.final_difference.to_string()
                                )
                            }
                        }
                        Err(msg) => {
                            let message: String = msg.to_owned()
                                + "\n Steps: "
                                + &gmm.steps.to_string()
                                + "\n Final Difference: "
                                + &gmm.final_difference.to_string();
                            panic!("{}", message)
                        }
                    };

                    bic_remove.push(self.bic(gmm, local_matrix));
                });

                let argmin_bic_remove = bic_remove.iter().enumerate().fold(
                    (0usize, T::from(100).unwrap()),
                    |current, possible_new| {
                        if &current.1 > possible_new.1 {
                            (possible_new.0, possible_new.1.to_owned())
                        } else {
                            current
                        }
                    },
                );

                if argmin_bic_remove.1 < self.best_bic {
                    println!("-- Removing --");
                    current_matrix
                        .remove(argmin_bic_remove.0, 1)
                        .expect("Unable to remove a column");
                    selected_columns.remove(argmin_bic_remove.0);
                    self.best_bic = argmin_bic_remove.1;
                    removed = true;
                }
            }

            println!(
                "Selected columns {:?} Removed: {removed} Added: {added} Best Bic: {:?}",
                selected_columns, self.best_bic
            );

            if !removed && !added || selected_columns.len() == data.columns {
                self.final_selection = selected_columns.to_vec();
                return Ok("Converged".to_string());

                /*
                panic!(
                    "CVS Converged \n Selected: {:?} \n BIC: {:?}",
                    self.final_selection, self.best_bic
                ); */
            }
        }
    }

    pub fn bic(&self, model: GaussianMixtureModel<T>, data: Matrix<T>) -> T {
        let ck: T = T::from(
            self.training_setup.number_clusters * data.columns
                + (self.training_setup.number_clusters - 1)
                + self.training_setup.number_clusters * data.columns * (data.columns + 1) / 2,
        )
        .unwrap();
        let log_likelihood: T = (model.gammas.sum(0).unwrap().sum(1).unwrap().content[0]).ln();

        -T::from(2.0).unwrap() * log_likelihood + T::from(data.columns).unwrap() * ck.ln()
    }
}
