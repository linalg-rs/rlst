//! Interpolative decomposition.

pub use rlst::prelude::*;

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
fn low_rank_matrix(n: usize, arr: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>) {
    //Obtain n equally distributed angles 0<phi<pi and n equally distributed angles 0<theta<2pi
    let pi: f64 = std::f64::consts::PI;

    let mut angles1 = rlst_dynamic_array2!(f64, [n, 1]);
    angles1.fill_from_seed_equally_distributed(0);
    angles1.scale_inplace(pi);

    let mut angles2 = rlst_dynamic_array2!(f64, [n, 1]);
    angles2.fill_from_seed_equally_distributed(1);
    angles2.scale_inplace(2.0 * pi);

    //Calculate n points on a sphere given phi and theta
    let mut points = rlst_dynamic_array2!(f64, [n, 3]);

    for i in 0..n {
        let phi: f64 = angles1.get_value([i, 0]).unwrap();
        let theta: f64 = angles2.get_value([i, 0]).unwrap();
        *points.get_mut([i, 0]).unwrap() = phi.sin() * theta.cos();
        *points.get_mut([i, 1]).unwrap() = phi.sin() * theta.sin();
        *points.get_mut([i, 2]).unwrap() = phi.cos();
    }

    for i in 0..arr.shape()[0] {
        let point1 = points.r().into_subview([i, 0], [1, 3]);
        for j in 0..n {
            if i != j {
                let point2 = points.r().into_subview([j, 0], [1, 3]);
                //Calculate distance between all combinations of different points.
                let res = point1.r().sub(point2.r());
                //Calculate kernel e^{-|points1-points2|^2}
                *arr.get_mut([i, j]).unwrap() = 1.0 / ((res.view_flat().norm_2().pow(2.0)).exp());
            } else {
                //If points are equal, set the value to 1
                *arr.get_mut([i, j]).unwrap() = 1.0;
            }
        }
    }
}

pub fn main() {
    let n: usize = 100;
    let slice: usize = 50;
    //Tolerance given for the
    let tol: f64 = 1e-5;

    //Create a low rank matrix
    let mut arr = rlst_dynamic_array2!(f64, [slice, n]);
    low_rank_matrix(n, &mut arr);

    let mut res = arr.r_mut().into_id_alloc(tol, None).unwrap();

    println!("The permuted matrix is:");
    res.arr.pretty_print();

    println!("The interpolative decomposition matrix is:");
    res.id_mat.pretty_print();

    println!("The rank of this matrix is {}\n", res.rank);

    let a_rs_app = empty_array().simple_mult_into_resize(
        res.id_mat.r(),
        res.arr.r_mut().into_subview([0, 0], [res.rank, n]),
    );
    let a_rs = res
        .arr
        .r_mut()
        .into_subview([res.rank, 0], [slice - res.rank, n]);

    let error: f64 = a_rs.r().sub(a_rs_app.r()).view_flat().norm_2();
    println!("Interpolative Decomposition L2 absolute error: {}", error);
}
