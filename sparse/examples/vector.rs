//! Example file for creating vectors.

use rlst_operator::Element;
use rlst_operator::LinearSpace;
use rlst_operator::NormedSpace;
use rlst_sparse::local::indexable_space::LocalIndexableVectorSpace;
use rlst_sparse::traits::indexable_vector::*;

fn main() {
    let space = LocalIndexableVectorSpace::<f64>::new(10);
    let mut vec = space.create_element();

    *vec.view_mut().view_mut().unwrap().get_mut(0).unwrap() = 2.0;

    let n = vec.view().view().unwrap().len();
    println!("The dimension of the vector is {}", n);
    println!("The norm of the vector is {}", vec.view().norm_2());

    println!("The norm of the vector is {}", space.norm(&vec.view()));
}
