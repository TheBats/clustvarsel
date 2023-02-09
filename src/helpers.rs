use std::iter::zip;

use num::Float;

/// Computes the mean squared error between two vectors 
pub fn mean_squared_error<T: Float>(a: &[T], b: &[T]) -> Result<T, &'static str> {
    if a.len() != b.len() {
        return Err("Sizes do not match");
    }

    let size = a.len() as f64;

    let error: T = zip(a, b).fold(T::zero(), |curr, (sub_a, sub_b)| {
        curr + (*sub_a - *sub_b).powf(T::from(2.0).unwrap())
    });

    let mse = error.sqrt() / T::from(size).unwrap();
    Ok(mse)
}
