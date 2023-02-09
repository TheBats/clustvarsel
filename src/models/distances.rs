use std::{
    iter::{zip, Sum},
    ops::{Mul, Sub},
};

use num::Float;

use crate::calculations::linear_algebra::dot_product;

/// Type of a distance computation function
pub type Distance<T> = fn(&[T], &[T]) -> Result<T, &'static str>;

/// Eucleadian distance computation
pub fn eucleadian_distance<T: Float + Sum<<T as Mul>::Output> + Clone>(
    a: &[T],
    b: &[T],
) -> Result<T, &'static str>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    let tmp: Vec<T> = zip(a, b).map(|(i, k)| i - k).collect();

    dot_product(&tmp, &tmp)
}
