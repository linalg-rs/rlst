//! Example file for creating vectors.

use rlst_sparse::local::indexable_space::LocalIndexableVectorSpace;
use rlst_traits::linalg::Norm2;
use rlst_traits::linalg::*;
use rlst_traits::Element;
use rlst_traits::LinearSpace;
use rlst_traits::NormedSpace;

fn main() {
    let space = LocalIndexableVectorSpace::<f64>::new(10);
    let mut vec = space.create_element();

    *vec.view_mut().view_mut().unwrap().get_mut(0).unwrap() = 2.0;

    let n = vec.view().view().unwrap().len();
    println!("The dimension of the vector is {}", n);
    println!("The norm of the vector is {}", vec.view().norm_2());

    println!("The norm of the vector is {}", space.norm(&vec.view()));
}
