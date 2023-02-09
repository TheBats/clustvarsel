use clustvarsel::calculations::matrix::Matrix;
use clustvarsel::calculations::stats::slow_covariance;
use clustvarsel::helpers::mean_squared_error;
use clustvarsel::parser::read_parse;

use std::fs::File;

#[test]
#[ignore]
#[cfg(target_arch = "x86")]
fn test_dp_f32_x86_64() {
    let test_vec_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let test_vec_b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];

    unsafe {
        let res = simd_functions::dot_product_f32_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(60f32))
    }

    let test_vec_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let test_vec_b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    unsafe {
        let res = simd_functions::dot_product_f32_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(55f32))
    }

    let test_vec_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let test_vec_b: Vec<f32> = vec![1.0, 2.0, 3.0];

    unsafe {
        let res = simd_functions::dot_product_f32_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(14f32))
    }
}

#[test]
#[ignore]
#[cfg(target_arch = "x86")]
fn test_dp_f64x4_x86_64() {
    let test_vec_a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let test_vec_b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    unsafe {
        let res = simd_functions::dot_product_f64x4_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(30f64))
    }

    let test_vec_a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let test_vec_b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    unsafe {
        let res = simd_functions::dot_product_f64x4_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(55f64))
    }

    let test_vec_a: Vec<f64> = vec![1.0, 2.0, 3.0];
    let test_vec_b: Vec<f64> = vec![1.0, 2.0, 3.0];

    unsafe {
        let res = simd_functions::dot_product_f64x4_x86_64(test_vec_a, test_vec_b);
        assert_eq!(res, Ok(14f64))
    }
}

#[test]
fn test_slow_covariance() {
    let v: Matrix<f32> =
        Matrix::from_1d_vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);

    let zs = vec![1u8, 1u8, 1u8];

    let x = slow_covariance(&v, &zs);

    // Shape expected: columns are the attributes
    assert_eq!(
        x,
        Matrix::from_1d_vector(vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], 3, 3)
    );
    println!("Test 1: Done");

    let v: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, 4.0],
        vec![5.0, 3.0],
        vec![6.0, 7.0],
        vec![8.0, 5.0],
        vec![9.0, 6.0],
    ]);
    let zs = vec![1u8, 1u8, 1u8, 1u8, 1u8];

    let x = slow_covariance(&v, &zs);

    // Shape expected: columns are the attributes
    assert_eq!(
        x,
        Matrix::from_2d_vector(vec![vec![7.5, 2.25], vec![2.25, 2.5]])
    );
    println!("Test 2: Done");

    let zs = vec![0u8, 0u8, 0u8, 1u8, 1u8];

    let x = slow_covariance(&v, &zs);

    // Shape expected: columns are the attributes
    assert_eq!(
        x,
        Matrix::from_2d_vector(vec![vec![0.5, 0.5], vec![0.5, 0.5]])
    );
    println!("Test 3: Done");

    let zs = vec![1u8, 1u8, 1u8, 0u8, 0u8];

    let x = slow_covariance(&v, &zs);
    let res: Matrix<f32> = Matrix::from_1d_vector(vec![4.33333, 2.16667, 2.16667, 4.33333], 2, 2);

    let err = mean_squared_error(&x.content, &res.content).unwrap();
    assert!(err < 1e-5, "Error: {}", err);
    println!("Test 4: Done");
}

#[test]
fn test_read() {
    let path = "test.csv";
    let (_, matrix): (Vec<String>, Matrix<f64>) = read_parse(path).unwrap();

    let file = File::open(path).unwrap();

    let mut rdr = csv::Reader::from_reader(&file);
    //let headers = rdr.headers().unwrap();

    rdr.records().enumerate().for_each(|(id, f)| {
        let mut real_vec: Vec<f64> = Vec::new();

        match f {
            Ok(record) => {
                record.iter().enumerate().for_each(|(_, entry)| {
                    println!("{:?}", entry);
                    match entry.parse::<f64>() {
                        Ok(entr) => real_vec.push(entr), //new_row.push(entr),
                        Err(_) => panic!("Could not parse {:?} to a f64", entry),
                    }
                });
            }
            Err(_) => panic!("Nothing to read"),
        }

        assert_eq!(real_vec, matrix[id].to_vec());
    });

    println!("Rows: {:?} Columns {:?}", matrix.rows, matrix.columns);
}
