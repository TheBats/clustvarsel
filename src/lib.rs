//! From scratch implemention of the Clustering Variable Selection algorithm (ClustCarSel)
//!
//! The aim of this project/crate is to implement an efficient and fast version of the ClustVarSel algorithm from scratch,
//! Because ClustVarSel relies on Gaussian Mixtures Model, the crate comes with Gaussian Mixture Models and K-means++.
//!
//!
//! It is mostly a project I worked on to learn Rust and improve my software engineering skills.
//! This implementation is still under a lot of work.

#![feature(stdsimd)]

pub mod calculations;
pub mod models;

pub mod simd_functions;

pub mod helpers;
pub mod parser;
pub mod training_setup;

use std::alloc::System;
use std::arch;

#[global_allocator]
static GLOBAL: System = System;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// Asks for a (sample x features) matrix
pub fn main() {
    #[cfg(target_arch = "aarch64")]
    println!(
        "Asimd enabled: {}",
        arch::is_aarch64_feature_detected!("dotprod")
    );

    #[cfg(target_arch = "x86_64")]
    println!("Avx2: {:?}", arch::is_x86_feature_detected!("avx2"));
}
