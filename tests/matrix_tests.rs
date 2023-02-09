use clustvarsel::calculations::linear_algebra::{
    determinant, lu_decomposition_matrix, matrix_dot_product, slow_inverse_matrix,
};
use clustvarsel::calculations::matrix::Matrix;
use clustvarsel::helpers::mean_squared_error;

#[test]
fn test_various_matrix() {
    let matrix: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let res_2d: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    assert_eq!(matrix.get_2d_vector(), res_2d);

    let mut matrix: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ]);
    let res: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, 4.0, 6.0],
        vec![8.0, 10.0, 12.0],
        vec![14.0, 16.0, 18.0],
        vec![20.0, 22.0, 24.0],
    ]);

    matrix.multiply_by_scalar(2.0);

    assert_eq!(matrix, res);

    let a: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);

    let a_transp: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 4.0, 1.0, 4.0],
        vec![2.0, 5.0, 2.0, 5.0],
        vec![3.0, 6.0, 3.0, 6.0],
    ]);

    let a_recv = a.get_transpose();

    assert_eq!(a_transp, a_recv);
}

#[test]
fn test_matrix_multiplication() {
    let mut a: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);

    let b: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 1.0, 2.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let res: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![20.0, 26.0, 20.0, 26.0],
        vec![47.0, 62.0, 47.0, 62.0],
        vec![20.0, 26.0, 20.0, 26.0],
        vec![47.0, 62.0, 47.0, 62.0],
    ]);

    a.multiply_matrix(&b).expect("Wrong size");

    assert_eq!(a, res);
    assert_eq!(a.rows, 4);
    assert_eq!(a.columns, 4);

    let mut a: Matrix<f32> = Matrix::from_2d_vector(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

    let b: Matrix<f32> =
        Matrix::from_2d_vector(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);

    let real: Matrix<f32> = Matrix::from_2d_vector(vec![vec![58.0, 64.0], vec![139.0, 154.0]]);

    a.multiply_matrix(&b).expect("Wrong size");

    assert_eq!(a, real);
    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 2);
}

#[test]
fn test_determinant_matrix() {
    let to_compute: Vec<Vec<f32>> = vec![
        vec![1.0, -2.0, -2.0, -3.0],
        vec![3.0, -9.0, 0.0, -9.0],
        vec![-1.0, 2.0, 4.0, 7.0],
        vec![-3.0, -6.0, 26.0, 2.0],
    ];

    assert_eq!(determinant(&Matrix::from_2d_vector(to_compute)), Ok(-6.0));

    let to_compute: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![5.0, 6.0, 7.0],
        vec![9.0, 10.0, 11.0],
        vec![13.0, 14.0, 15.0],
    ];

    assert_eq!(
        determinant(&Matrix::from_2d_vector(to_compute)),
        Err("Matrix is not squared")
    );

    let to_compute: Vec<Vec<f32>> = vec![vec![7.5, 2.25], vec![2.25, 2.5]];

    assert_eq!(
        determinant(&Matrix::from_2d_vector(to_compute)),
        Ok(13.6874999)
    );
}

#[test]
fn test_lu_decomposition_matrix() {
    let to_decompose: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, -1.0, -2.0],
        vec![-4.0, 6.0, 3.0],
        vec![-4.0, -2.0, 8.0],
    ]);

    let (mut lower, upper) = lu_decomposition_matrix(&to_decompose).unwrap();

    lower.multiply_matrix(&upper).expect("Shape do not match");

    assert_eq!(lower, to_decompose);

    let to_decompose: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.3, -1.0, -2.4],
        vec![-4.0, 6.1, 3.0],
        vec![-4.0, -2.2222, 8.0],
    ]);

    let (mut lower, upper) = lu_decomposition_matrix(&to_decompose).unwrap();

    lower.multiply_matrix(&upper).expect("Shape do not match");

    assert_eq!(lower, to_decompose);
}

#[test]
fn test_inverse_matrix() {
    let to_inverse: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, 1.0, -1.0],
        vec![-3.0, -1.0, 2.0],
        vec![-2.0, 1.0, 2.0],
    ]);

    let res = slow_inverse_matrix(&to_inverse);
    let real: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![4.0, 3.0, -1.0],
        vec![-2.0, -2.0, 1.0],
        vec![5.0, 4.0, -1.0],
    ]);

    println!("First Inverse");
    assert_eq!(res, real);

    let to_inverse: Matrix<f64> = Matrix::from_2d_vector(vec![vec![7.5, 2.25], vec![2.25, 2.5]]);

    let res = slow_inverse_matrix(&to_inverse);
    let real: Matrix<f64> = Matrix::from_2d_vector(vec![
        vec![0.1826484, -0.16438356],
        vec![-0.16438356, 0.54794521],
    ]);

    println!("Second Inverse: {:?}", res);

    let err = mean_squared_error(&res.content, &real.content).unwrap();
    assert!(err < 1e-5, "Error: {}", err);

    let to_inverse: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.5, 1.0, -1.0, 5.0],
        vec![-3.0, -1.0, 2.0, 100.0],
        vec![-2.0, 1.15, 2.0, 5.0],
        vec![35.0, 18.0, 98.234, 4.0],
    ]);
    let res = slow_inverse_matrix(&to_inverse);
    let real: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![
            1.47777903e-01,
            3.25120482e-03,
            -2.17495001e-01,
            5.86625299e-03,
        ],
        vec![
            4.39416679e-01,
            -4.81911356e-02,
            5.28653093e-01,
            -5.30882623e-03,
        ],
        vec![
            -1.33637313e-01,
            7.28636401e-03,
            -1.93418236e-02,
            9.06482079e-03,
        ],
        vec![
            1.15002501e-02,
            9.46989751e-03,
            -8.51482640e-04,
            -5.83970886e-05,
        ],
    ]);

    println!("Third Inverse: {:?}", res);

    let err = mean_squared_error(&res.content, &real.content).unwrap();
    assert!(err < 1e-5, "Error: {}", err);
}

#[test]
fn test_matrix_means() {
    let v: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, 4.0],
        vec![5.0, 3.0],
        vec![6.0, 7.0],
        vec![8.0, 5.0],
        vec![9.0, 6.0],
    ]);
    let mean0 = v.mean(0).unwrap();
    assert_eq!(mean0.content, vec![6.0, 5.0]);

    let mean1 = v.mean(1).unwrap();
    assert_eq!(mean1.content, vec![3.0, 4.0, 6.5, 6.5, 7.5])
}

#[test]
fn test_matrix_sum() {
    let a: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 2, 3);

    let sum0: Matrix<f32> = a.sum(0).unwrap();
    assert_eq!(sum0.content, [2.0, 4.0, 6.0]);
    let sum1: Matrix<f32> = a.sum(1).unwrap();
    assert_eq!(sum1.content, [6.0, 6.0]);
}

#[test]
fn test_matrix_append() {
    let mut a: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 2, 3);

    a.append_vector(&[100.0, 100.0, 100.0], 0)
        .expect("Unable to append column");

    assert_eq!(
        a.content,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 100.0, 100.0, 100.0]
    );
    assert_eq!(a.rows, 3);
    assert_eq!(a.columns, 3);

    a.append_vector(&[5.0, 5.0, 5.0], 1)
        .expect("Unable to append column");

    assert_eq!(
        a.content,
        vec![1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0, 100.0, 100.0, 100.0, 5.0]
    );
    assert_eq!(a.rows, 3);
    assert_eq!(a.columns, 4);

    let mut a: Matrix<f32> = Matrix::empty();
    a.append_vector(&[1.0, 1.0], 0)
        .expect("Unable to append column");

    assert_eq!(a.content, vec![1.0, 1.0]);
    assert_eq!(a.rows, 1);
    assert_eq!(a.columns, 2);

    let mut a: Matrix<f32> = Matrix::empty();
    a.append_vector(&[1.0, 1.0], 1)
        .expect("Unable to append column");

    assert_eq!(a.content, vec![1.0, 1.0]);
    assert_eq!(a.rows, 2);
    assert_eq!(a.columns, 1);

    let mut m: Matrix<f64> = Matrix::identity(100).unwrap();
    println!("Idendity done");
    let to_add: Vec<f64> = vec![0.0; 100];
    m.append_vector(&to_add, 1).expect("Unable to append column");
}

#[test]
fn test_matrix_remove() {
    let mut a: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 2, 3);
    a.remove(1, 1).expect("Error removing");

    let real: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 3.0, 1.0, 3.0], 2, 2);

    assert_eq!(a, real);

    let mut a: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 2, 3);
    a.remove(1, 0).expect("Error removing");

    let real: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 3.0], 1, 3);

    assert_eq!(a, real);
}

#[test]
fn test_matrix_dot() {
    let v1: Matrix<f32> = Matrix::from_1d_vector(vec![1.0, 2.0, 1.0, 2.0], 2, 2);
    let v2: Matrix<f32> = Matrix::from_1d_vector(vec![2.0, 2.0], 2, 1);

    let res: Matrix<f32> = matrix_dot_product(&v1, &v2).unwrap();

    assert_eq!(res.content, vec![6.0, 6.0]);
    assert_eq!(res.columns, 1);
    assert_eq!(res.rows, 2);
}
