use crate::calculations::matrix::Matrix;

use core::ops::{Index, IndexMut};
use std::{fmt::Debug, iter::Sum};

use num::Float;

impl<T: Float + 'static> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, s: usize) -> &Self::Output {
        &self.content[(s * self.columns)..((s + 1) * self.columns)]
    }
}

impl<T: Float + 'static> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, s: usize) -> &mut [T] {
        &mut self.content[(s * self.columns)..((s + 1) * self.columns)]
    }
}

impl<T: Float + Debug + Sync + Send + Sum + 'static> Debug for Matrix<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("")
            .field("content", &self.get_2d_vector())
            .finish()
    }
}

impl<T: Float> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}

impl<T: Float> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Matrix {
            content: self.content.clone(),
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<'a, T: Float + 'static> IntoIterator for &'a Matrix<T> {
    type Item = Vec<T>;
    type IntoIter = MatrixIntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixIntoIter {
            matrix: self,
            index: 0,
        }
    }
}

pub struct MatrixIntoIter<'a, T> {
    pub matrix: &'a Matrix<T>,
    pub index: usize,
}

impl<'a, T: Float + 'static> Iterator for MatrixIntoIter<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix.rows > self.index {
            let row = self.matrix[self.index].to_vec();
            self.index += 1;
            return Some(row);
        }

        None
    }
}
