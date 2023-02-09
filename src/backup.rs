use std::ops::{Mul, Sub, AddAssign, Div};
use std::cmp::Ordering;
use std::iter::{zip, Sum};

use num::Float;
use itertools::izip;

pub fn eucleadian_distance<T: Float + Sum<<T as Mul>::Output> + Clone> (a: &Vec<T>, b: &Vec<T>) -> Result<T, &'static str> 
    where 
        for<'a> &'a T: Sub<&'a T, Output=T> {

        let tmp: Vec<T> = zip(a, b)
            .map(|(i, k)| {
                i-k
            })
            .collect();

        dot_product(tmp.clone(), tmp)
    } 

pub fn dot_product<T: Float + Sum>(a: Vec<T>, b: Vec<T>) -> Result<T, &'static str> {

    let res: T = zip(a, b)
        .map(|(i,k)| i*k)
        .sum();

    Ok(res)
}

pub fn slow_covariance<T: Float + Mul + AddAssign + Sum + std::fmt::Debug>(a: &Vec<Vec<T>>, zs: &Vec<u8>) -> Vec<Vec<T>> {
        
    let mut cov_ma: Vec<Vec<T>> = vec![vec![T::from(0.0).unwrap(); a.len()]; a.len()];

    let count: i32 = zs.iter().fold(0i32, |mut sum, v| {sum += *v as i32; sum});

    let means: Vec<T> = a.iter()
        .map(|feature| {
       		let sum = zip(feature, zs)
       			.fold(T::from(0.0).unwrap(), |mut sum, (row, z)| {

       				if z == &1 {
                        sum += T::from(*z).unwrap().mul(*row); 
                    } 
                    println!("{:?}", sum);
                    sum
                });

       		sum/T::from(count).unwrap()
            
        }).collect();

    let val_minus_mean: Vec<Vec<T>> = a.iter()
    				.enumerate()
    				.map(|(ind, features)| {
    					let current_mean = means[ind];
    					zip(zs,features)
    						.map(|(z, feat)| {
    							T::from(*z).unwrap().mul(*feat-current_mean)
    						}).collect()
    				}).collect();

    val_minus_mean.iter()
        .enumerate()
        .for_each(|(ind, feat1)| {
            val_minus_mean.iter()
                .enumerate()
                .for_each(|(i, feat2)| {

                    if i >= ind {
                        let dp: T = izip!(zs, feat1, feat2)
                            .fold(T::from(0.0).unwrap(), |mut sum, (zs, v1, v2)| {
                                if zs == &1 {
                                    sum += T::from(*zs).unwrap().mul(v1.mul(*v2)); 
                                } 
                                sum
                            });

                        let v = dp/T::from(count-1i32).unwrap();

                        cov_ma[ind][i] = v.clone();
                        cov_ma[i][ind] = v;
                    }

                });
        });

    cov_ma
}

pub fn slow_transpose<T: Float + Clone>(to_transp: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    
    let mut transpose: Vec<Vec<T>> = vec![vec![T::from(0.0).unwrap(); to_transp.len()]; to_transp[0].len()];

    to_transp.iter().enumerate()
        .for_each(|(ind, row)| {
            row.iter().enumerate()
                .for_each(|(i, feat)| transpose[i][ind] = feat.clone());
        });

    transpose
}

pub fn slow_inverse<T: Float>() -> Vec<Vec<T>> {

    

    Vec::new()
}

/*
pub fn borrowed_dot_product<T: Sum + Mul>(a: &Vec<T>, b: &Vec<T>) -> Result<T, &'static str> 
    where 
        &'static T: Mul,
        T: Sum<<&'static T as Mul<&'static T>>::Output> + Mul + 'static {

    let res: T = zip(a, b)
        .map(|(i, k)| i*k)
        .sum();

    Ok(res)
} */

pub fn find_min<T: PartialOrd + Copy + Send + Sync>(to_search: &Vec<Vec<T>>) -> Result<T, &str> {
    let res: Result<T, &str> = to_search
        .iter()
        .flatten()
        .try_fold(to_search[0][0], |a, &b| {
            match a.partial_cmp(&b) {
                Some(Ordering::Less) | Some(Ordering::Equal) => Ok(a),
                Some(Ordering::Greater) => Ok(b),
                None => Err("Encountered NaN")
            }
        });
    res
}

pub fn find_max<T: PartialOrd + Copy + Send + Sync>(to_search: &Vec<Vec<T>>) -> T {
    let first_elem: T = to_search[0][0];
    to_search
        .iter()
        .flatten()
        .fold(first_elem, |a, &b| {
            if a > b {
                a
            } else {
                b
            }
        })
}

pub fn biggest_elements(to_get: Vec<usize>) -> usize {
    let mut max = to_get[0];

    for i in 1..to_get.len() {
        if to_get[i] < max {
            max = i;
        }
    }

    max
}