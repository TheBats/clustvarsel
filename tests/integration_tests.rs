use clustvarsel::calculations::stats::slow_covariance;
use clustvarsel::helpers::mean_squared_error;
use clustvarsel::models::clustvarsel::CLUSTVARSEL;
use clustvarsel::models::gmm::GaussianMixtureModel;
use clustvarsel::{
    calculations::{matrix::Matrix, stats::slow_multivariate_gaussian},
    models::{
        distances::{eucleadian_distance, Distance},
        kmeans::Kmeans,
    },
    parser,
};

#[test]
fn test_kmeans() {
    let data_vec = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.5],
        vec![0.5, 0.0],
        vec![0.5, 0.5],
        vec![1.0, 0.0],
        vec![1.0, 0.5],
    ];

    let data: Matrix<f64> = Matrix::from_2d_vector(data_vec);

    let distance: Distance<f64> = eucleadian_distance;

    let mut ini = Kmeans::init(2, &data, 5, distance);

    ini.fit(&data);

    assert_eq!(
        vec![
            vec![0.16666666666666666, 0.3333333333333333],
            vec![0.8333333333333334, 0.16666666666666666]
        ],
        ini.centroids
    );
}

#[test]
fn test_gmm() {
    let data = vec![
        vec![1.0754941903392263, -0.23128656563065886],
        vec![0.4574473936178802, 0.9360309725246923],
        vec![0.32934947297877976, 1.38548750306268],
        vec![0.5282807898527796, -1.2790598727596125],
        vec![0.7229444360398654, 0.051849483195611334],
        vec![-0.9711088688749532, 0.5972057096698321],
        vec![0.059956970208871474, -0.7683195725500758],
        vec![0.06507523188571362, 0.5100866099687131],
        vec![2.621390670414035, 1.1260363315127664],
        vec![-0.8716491907937076, 0.024930994116899585],
        vec![0.16012622229024365, -0.5051931551551303],
        vec![-0.14341732338231827, 1.2396922302991598],
        vec![0.4608632513583871, -0.2355004832808132],
        vec![0.8396452500894849, -0.25630691287117346],
        vec![0.8002203851111599, 2.163947737335655],
        vec![1.3294368435849129, 0.8596105028558736],
        vec![1.8505390645568442, -0.1667868269013153],
        vec![1.0665709281054623, -0.276013437371877],
        vec![1.1246308563933862, 0.7426135129709652],
        vec![0.12585120896214103, 0.4982598386969306],
        vec![2.329942004958563, 3.645549413834696],
        vec![2.9552284431469484, 2.394856391293376],
        vec![4.188921565261378, 2.9422653443714863],
        vec![3.880344690692337, 1.228392170139145],
        vec![2.4887039201402814, 4.659172805066865],
        vec![2.111935788965111, 3.4608191188214916],
        vec![3.170814852951174, 2.9878325954472973],
        vec![2.5262548875177777, 1.2021936440575964],
        vec![0.4320207439021613, 1.786987791429834],
        vec![3.3682528645305965, 4.626355282338605],
        vec![4.761362496567665, 3.9651351147534957],
        vec![3.0308191293920927, 3.1865353624620534],
        vec![4.174158406325235, 3.1611480416571074],
        vec![2.8527683134876978, 3.830187147596906],
        vec![4.5023949748381655, 3.677470024796094],
        vec![3.1927073299205353, 1.9551330147104427],
        vec![3.0922466979663485, 2.0855491835997255],
        vec![2.707032712786994, 3.9455880382897823],
        vec![2.2527686286505206, 1.8939225421290395],
        vec![1.3136316769676093, 1.7154019350664793],
    ];

    let data_matrix_form = Matrix::from_2d_vector(data);

    let mut gmm = GaussianMixtureModel::new(2, 4, vec![0.5, 0.5], 100, 1e-5);

    let distance: Distance<f64> = eucleadian_distance;

    match gmm.fit(&data_matrix_form, distance) {
        Ok(msg) => println!(
            "{}",
            msg.to_owned()
                + "\n Steps: "
                + &gmm.steps.to_string()
                + "\n Final Difference: "
                + &gmm.final_difference.to_string()
        ),
        Err(msg) => {
            let message: String = msg.to_owned()
                + "\n Steps: "
                + &gmm.steps.to_string()
                + "\n Final Difference: "
                + &gmm.final_difference.to_string();
            panic!("{}", message)
        }
    }
}

#[test]
fn test_multivariate_gaussian() {
    let mut v: Matrix<f32> = Matrix::from_2d_vector(vec![
        vec![2.0, 4.0],
        vec![5.0, 3.0],
        vec![6.0, 7.0],
        vec![8.0, 5.0],
        vec![9.0, 6.0],
    ]);

    let res: Vec<f32> = vec![0.01464371, 0.0182322, 0.01437868, 0.02985463, 0.02354461];

    let zs = vec![1u8, 1u8, 1u8, 1u8, 1u8];

    let mut cov = slow_covariance(&v, &zs);
    println!("{:?}", cov);

    let means = v.mean(0).unwrap();
    println!("Means Matrix: {:?}", means);

    match slow_multivariate_gaussian(&mut v, &mut cov, &means.content) {
        Ok(matrix) => {
            println!("{:?}", matrix);
            let mse = mean_squared_error(&matrix, &res).unwrap();
            assert!(mse < 1e-5, "Error: {}", mse);
        }
        Err(e) => panic!("{}", e),
    }

    println!("Gaussian computed");
}

#[test]
fn test_1d_gmm() {
    let v: Matrix<f64> = Matrix::from_1d_vector(vec![2.0, 4.0, 5.0, 3.0, 6.0, 7.0, 8.0, 5.0], 8, 1);

    let mut gmm = GaussianMixtureModel::new(2, 4, vec![0.5, 0.5], 100, 1e-5);

    let distance: Distance<f64> = eucleadian_distance;

    match gmm.fit(&v, distance) {
        Ok(msg) => println!(
            "{}",
            msg.to_owned()
                + "\n Steps: "
                + &gmm.steps.to_string()
                + "\n Final Difference: "
                + &gmm.final_difference.to_string()
        ),
        Err(msg) => {
            let message: String = msg.to_owned()
                + "\n Steps: "
                + &gmm.steps.to_string()
                + "\n Final Difference: "
                + &gmm.final_difference.to_string();
            panic!("{}", message)
        }
    }
}

#[test]
fn test_clustvarsel() {
    let data = vec![
        vec![1.0754941903392263, -0.23128656563065886],
        vec![0.4574473936178802, 0.9360309725246923],
        vec![0.32934947297877976, 1.38548750306268],
        vec![0.5282807898527796, -1.2790598727596125],
        vec![0.7229444360398654, 0.051849483195611334],
        vec![-0.9711088688749532, 0.5972057096698321],
        vec![0.059956970208871474, -0.7683195725500758],
        vec![0.06507523188571362, 0.5100866099687131],
        vec![2.621390670414035, 1.1260363315127664],
        vec![-0.8716491907937076, 0.024930994116899585],
        vec![0.16012622229024365, -0.5051931551551303],
        vec![-0.14341732338231827, 1.2396922302991598],
        vec![0.4608632513583871, -0.2355004832808132],
        vec![0.8396452500894849, -0.25630691287117346],
        vec![0.8002203851111599, 2.163947737335655],
        vec![1.3294368435849129, 0.8596105028558736],
        vec![1.8505390645568442, -0.1667868269013153],
        vec![1.0665709281054623, -0.276013437371877],
        vec![1.1246308563933862, 0.7426135129709652],
        vec![0.12585120896214103, 0.4982598386969306],
        vec![2.329942004958563, 3.645549413834696],
        vec![2.9552284431469484, 2.394856391293376],
        vec![4.188921565261378, 2.9422653443714863],
        vec![3.880344690692337, 1.228392170139145],
        vec![2.4887039201402814, 4.659172805066865],
        vec![2.111935788965111, 3.4608191188214916],
        vec![3.170814852951174, 2.9878325954472973],
        vec![2.5262548875177777, 1.2021936440575964],
        vec![0.4320207439021613, 1.786987791429834],
        vec![3.3682528645305965, 4.626355282338605],
        vec![4.761362496567665, 3.9651351147534957],
        vec![3.0308191293920927, 3.1865353624620534],
        vec![4.174158406325235, 3.1611480416571074],
        vec![2.8527683134876978, 3.830187147596906],
        vec![4.5023949748381655, 3.677470024796094],
        vec![3.1927073299205353, 1.9551330147104427],
        vec![3.0922466979663485, 2.0855491835997255],
        vec![2.707032712786994, 3.9455880382897823],
        vec![2.2527686286505206, 1.8939225421290395],
        vec![1.3136316769676093, 1.7154019350664793],
    ];

    let data_matrix_form = Matrix::from_2d_vector(data);

    let mut cvs = CLUSTVARSEL::new(2, 2, 1e-5, 100, vec![0.5, 0.5], true, 2);

    match cvs.fit(data_matrix_form) {
        Ok(_) => assert_eq!(cvs.final_selection, vec![0]),
        Err(msg) => panic!("{msg:?}"),
    }

    println!("Final: {:?}", cvs.best_bic);
}

fn test_clustvarsel_cats_and_dogs() {
    let (_, data_matrix_form) = parser::read_parse("CATSnDOGS.csv").unwrap();

    let mut cvs = CLUSTVARSEL::new(2, 2, 1e-5, 10000, vec![0.5, 0.5], true, 2);
    cvs.fit(data_matrix_form);

    println!("Final: {:?}", cvs.best_bic);
}
