use num::Float;

/// Struct containing the training information
pub struct TrainingSetup<T> {
    pub number_clusters: usize,
    pub seed: u64,
    pub tolerance: T,
    pub max_steps: i32,
    pub initial_mixtures: Vec<T>,
    pub verbose: bool,
    pub cores: usize,
}

impl<T: Float> Clone for TrainingSetup<T> {
    fn clone(&self) -> Self {
        TrainingSetup {
            number_clusters: self.number_clusters,
            seed: self.seed,
            tolerance: self.tolerance,
            max_steps: self.max_steps,
            initial_mixtures: self.initial_mixtures.clone(),
            verbose: self.verbose,
            cores: self.cores,
        }
    }
}
