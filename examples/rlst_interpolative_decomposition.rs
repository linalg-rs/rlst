//! Demo the inverse of a matrix

pub use rlst::prelude::*;
use rlst::dense::linalg::interpolative_decomposition::{MatrixIdDecomposition, Accuracy};

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
fn low_rank_matrix(n: usize, arr: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>){
    //Obtain n equally distributed angles 0<phi<pi and n equally distributed angles 0<theta<2pi
    let pi: f64 = std::f64::consts::PI;

    let mut angles1 = rlst_dynamic_array2!(f64, [n, 1]);
    angles1.fill_from_seed_equally_distributed(0);
    angles1.scale_inplace(pi);

    let mut angles2 = rlst_dynamic_array2!(f64, [n, 1]);
    angles2.fill_from_seed_equally_distributed(1);
    angles2.scale_inplace(2.0*pi);


    //Calculate n points on a sphere given phi and theta
    let mut points = rlst_dynamic_array2!(f64, [n, 3]);

    for i in 0..n{
        let phi: f64 = angles1.get_value([i, 0]).unwrap();
        let theta: f64 = angles2.get_value([i, 0]).unwrap();
        *points.get_mut([i, 0]).unwrap() = phi.sin()*theta.cos();
        *points.get_mut([i, 1]).unwrap() = phi.sin()*theta.sin();
        *points.get_mut([i, 2]).unwrap() = phi.cos();

    }
    
    for i in 0..arr.shape()[0]{
        let point1 = points.view().into_subview([i, 0], [1, 3]);
        for j in 0..n{
            if i!=j{
                let point2 = points.view().into_subview([j, 0], [1, 3]);
                //Calculate distance between all combinations of different points. 
                let res = point1.view().sub(point2.view());
                //Calculate kernel e^{-|points1-points2|^2}
                *arr.get_mut([i, j]).unwrap() = 1.0/((res.view_flat().norm_2().pow(2.0)).exp());
            }
            else{
                //If points are equal, set the value to 1
                *arr.get_mut([i, j]).unwrap() = 1.0;
            }
        }
    }

}

pub fn main() {
    let n:usize = 100;
    let slice: usize = 50;
    //Tolerance given for the 
    let tol: f64 = 1e-5;

    //Create a low rank matrix
    let mut arr: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [slice, n]);
    low_rank_matrix(n, &mut arr);

    let res: IdDecomposition<f64> = arr.view_mut().into_id_alloc(Accuracy::Tol(tol)).unwrap();

    println!("The skeleton of the matrix is given by");
    res.skel.pretty_print();

    println!("The interpolative decomposition matrix is:");
    res.id_mat.pretty_print();

    println!("The rank of this matrix is {}\n", res.rank);

    //We extract the residuals of the matrix
    let mut perm_mat: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [slice, slice]);
    res.get_p(perm_mat.view_mut());
    let perm_arr: DynamicArray<f64, 2> = empty_array::<f64, 2>()
        .simple_mult_into_resize(perm_mat.view_mut(), arr.view());

    let mut a_rs: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [slice-res.rank, n]);
    a_rs.fill_from(perm_arr.into_subview([res.rank, 0], [slice-res.rank, n]));

    //We compute an approximation of the residual columns of the matrix
    let a_rs_app: DynamicArray<f64, 2> = empty_array().simple_mult_into_resize(res.id_mat.view(), res.skel);
    
    let error: f64 = a_rs.view().sub(a_rs_app.view()).view_flat().norm_2();
    println!("Interpolative Decomposition L2 absolute error: {}", error);
}
