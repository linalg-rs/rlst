//! Example file for creating vectors.

#[cfg(not(feature = "mpi"))]
fn main() {}

#[cfg(feature = "mpi")]
fn main() {
    use mpi::traits::*;
    use rlst_operator::Element;
    use rlst_operator::LinearSpace;
    use rlst_sparse::index_layout::DefaultMpiIndexLayout;
    use rlst_sparse::operator_interface::mpi_default_function_space::DistributedIndexableVectorSpace;
    use rlst_sparse::traits::indexable_vector::*;
    use rlst_sparse::vector::DefaultSerialVector;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let n = 100;

    // We first create an index layout.
    let index_layout = DefaultMpiIndexLayout::new(n, &world);
    let space = DistributedIndexableVectorSpace::<'_, f64, _>::new(&index_layout);
    let mut vec = space.create_element();

    let vec_impl = vec.view_mut();

    let mut local_vec: DefaultSerialVector<f64>;

    if rank == 0 {
        local_vec = DefaultSerialVector::<f64>::new(n);
        let mut view = local_vec.view_mut().unwrap();

        for index in 0..n {
            *view.get_mut(index).unwrap() = index as f64;
        }

        println!("Local inf norm: {}", local_vec.norm_infty());
        println!("Local inner: {}", local_vec.inner(&local_vec).unwrap());

        let _ = vec_impl.fill_from_root(&Some(local_vec));
    } else {
        let _ = vec_impl.fill_from_root(&None);
    }

    let inner = vec.view().inner(&vec.view()).unwrap();
    let inf_norm = vec.view().norm_infty();

    if rank == 0 {
        println!("Inner: {}", &inner);
        println!("Inf norm: {}", &inf_norm);
    }
}
