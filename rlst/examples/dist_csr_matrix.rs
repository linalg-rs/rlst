//! Working with a distributed CSR matrix.

use std::rc::Rc;

use mpi::traits::Root;
use mpi::{self, traits::Communicator};
use rlst::dense::array::DynArray;
use rlst::{
    self,
    io::matrix_market::{read_array_mm, read_coordinate_mm},
    Shape,
};
use rlst::{
    assert_array_relative_eq, AijIteratorByValue, AsMatrixApply, FromAijDistributed, GatherToOne,
    ScatterFromOne,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let (dist_mat, dist_x) = {
        if rank == 0 {
            println!("Loading CSR matrix example on process {rank}");

            // Read the sparse matrix in matrix market format.
            let sparse_mat = read_coordinate_mm::<f64>("examples/mat_507_313.mm").unwrap();

            // Communicate the shape of the matrix to all processes.

            let mut shape = sparse_mat.shape();

            world.this_process().broadcast_into(shape.as_mut_slice());

            // We create the domain layout and range layout.

            let domain_layout = Rc::new(
                rlst::distributed_tools::IndexLayout::from_equidistributed_chunks(
                    shape[1], 1, &world,
                ),
            );
            let range_layout = Rc::new(
                rlst::distributed_tools::IndexLayout::from_equidistributed_chunks(
                    shape[0], 1, &world,
                ),
            );

            // We now distribute the matrix across processes.
            let dist_mat = rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::from_aij_iter(
                domain_layout.clone(),
                range_layout.clone(),
                sparse_mat.iter_aij_value(),
            );

            //  We now read the vector x and distribute it across.

            let x = read_array_mm::<f64>("examples/x_313.mm").unwrap();
            let dist_x = x.slice(1, 0).scatter_from_one_root(domain_layout.clone());

            (dist_mat, dist_x)
        } else {
            // Other processes just need to receive the shape and the distributed matrix.

            let mut shape = [0, 0];

            world
                .process_at_rank(0)
                .broadcast_into(shape.as_mut_slice());

            let domain_layout = Rc::new(
                rlst::distributed_tools::IndexLayout::from_equidistributed_chunks(
                    shape[1], 1, &world,
                ),
            );
            let range_layout = Rc::new(
                rlst::distributed_tools::IndexLayout::from_equidistributed_chunks(
                    shape[0], 1, &world,
                ),
            );

            let dist_mat =
                rlst::sparse::distributed_csr_mat::DistributedCsrMatrix::<'_, f64, _>::from_aij(
                    domain_layout.clone(),
                    range_layout.clone(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                );

            let dist_x = DynArray::<f64, 1>::scatter_from_one(0, domain_layout.clone());

            (dist_mat, dist_x)
        }
    };

    // We now apply the  distributed matrix vector multiplication.

    let mut dist_y = rlst::dist_vec!(f64, dist_mat.range_layout().clone());
    dist_mat.apply(1.0, &dist_x, 0.0, &mut dist_y);

    // We now gather the result on process 0 and compare to the expected result.

    if rank == 0 {
        let result = dist_y.gather_to_one_root();

        // Read the expected result vector in matrix market format and slice to one dimension.
        let y_expected = read_array_mm::<f64>("examples/y_507.mm")
            .unwrap()
            .slice(1, 0);

        assert_array_relative_eq!(result, y_expected, 1E-10);

        println!("Test passed.")
    } else {
        dist_y.gather_to_one(0);
    }
}
