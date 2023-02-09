use std::iter::{zip, Sum};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use clustvarsel::calculations::linear_algebra::slow_inverse_matrix;
use clustvarsel::calculations::matrix::Matrix;

use clustvarsel::parser::read_parse;

#[cfg(target_arch = "x86")]
use clustvarsel::simd_functions::{dot_product_f32_x86_64, dot_product_f64x4_x86_64};

#[cfg(target_arch = "x86")]
use core::arch::x86_64::*;
use std::fs::File;

//#[cfg(target_arch = "x86")]
fn linear_algebra(c: &mut Criterion) {
    let test_vec: Vec<Vec<Vec<f64>>> = vec![
        (0..1000)
            .map(|_| (0..1000).map(f64::from).collect())
            .collect(),
        (0..10000)
            .map(|_| (0..10000).map(f64::from).collect())
            .collect(),
    ];
    //(0..100000).map(|_| (0..100000).map(f64::from).collect()).collect()];

    let mut group = c.benchmark_group("Inverse");
    for (ind, v) in test_vec.iter().enumerate() {
        let m = Matrix::from_2d_vector(v.clone());
        group.bench_with_input(BenchmarkId::new("Matrix", ind), &ind, |b, _| {
            b.iter(|| slow_inverse_matrix(black_box(&m.clone())))
        });
        /*
        group.bench_with_input(BenchmarkId::new("Vanilla", ind), &ind, |b, _| b.iter(|| {
            slow_inverse(black_box(&v.clone()))}
        )); */
    }

    //panic!("Rows: {:?} Columns {:?}", matrix.rows, matrix.columns);

    //group.finish()
}

fn parser(c: &mut Criterion) {
    let path = "CATSandDOGS.csv";
    c.bench_function("Read+Parse", move |b| {
        b.iter(|| read_parse(path));
    });
}

fn usual_insert(c: &mut Criterion) {
    c.bench_function("Usual Insert", move |b| {
        b.iter(|| {
            let mut m: Matrix<f64> = Matrix::identity(1000, 1000).unwrap();
            let to_add: Vec<f64> = vec![0.0; 1000];
            m.append_vector(to_add, 1);
        });
    });
}

fn usual_within_insert(c: &mut Criterion) {
    c.bench_function("Usual Within Insert", move |b| {
        b.iter(|| {
            let mut m: Matrix<f64> = Matrix::identity(1000, 1000).unwrap();
            let to_add: Vec<f64> = vec![0.0; 1000];
            m.append_vector(to_add, 1);
        });
    });
}

fn new_insert(c: &mut Criterion) {
    c.bench_function("New Insert", move |b| {
        b.iter(|| {
            let mut m: Matrix<f64> = Matrix::identity(1000, 1000).unwrap();
            let to_add: Vec<f64> = vec![0.0; 1000];

            for (ind, i) in to_add.iter().enumerate() {
                let to_append = (ind + 1) * (m.columns) + ind;
                m.content.insert(to_append, i.to_owned());
            }
        });
    });
}

fn remove(c: &mut Criterion) {
    c.bench_function("Remove", move |b| {
        b.iter(|| {
            let mut m: Matrix<f64> = Matrix::identity(1000, 1000).unwrap();
            m.remove(500, 1);
        });
    });
}

fn dot_prod_sum(c: &mut Criterion) {
    c.bench_function("Dot prod sum", move |b| {
        b.iter(|| {
            let to_dot = [300.0; 100000];
            zip(to_dot, to_dot).map(|(i, k)| i * k).sum::<f64>();
        });
    });
}

fn dot_prod_par(c: &mut Criterion) {
    c.bench_function("Dot prod par", move |b| {
        b.iter(|| {
            let to_dot = [300.0; 100000];
            to_dot
                .into_par_iter()
                .zip(to_dot)
                .fold(|| 0f64, |val, to| val + to.0 * to.1);
        });
    });
}

fn dot_prod_fold(c: &mut Criterion) {
    c.bench_function("Dot prod fold", move |b| {
        b.iter(|| {
            let to_dot = [300.0; 100000];
            zip(to_dot, to_dot).fold(0f64, |val, to| val + to.0 * to.1);
        });
    });
}

criterion_group!(benches, dot_prod_sum, dot_prod_par, dot_prod_fold);
criterion_main!(benches);
