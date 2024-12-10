//! Example of a distributed CG iteration across several MPI ranks.

use mpi::topology::Communicator;
use rand::Rng;
use rlst::operator::interface::distributed_sparse_operator::DistributedCsrMatrixOperator;
use rlst::operator::interface::DistributedArrayVectorSpace;
use rlst::{
    CgIteration, DefaultDistributedIndexLayout, DistributedCsrMatrix, Element, LinearSpace,
    NormedSpace,
};

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();

    // We setup a diagonal sparse matrix on the first rank, send this
    // around and then wrap this in an operator to run CG.

    // The matrix dimension.
    let n = 500;
    let tol = 1E-5;

    let index_layout = DefaultDistributedIndexLayout::new(n, 1, &world);

    let space = DistributedArrayVectorSpace::<f64, _>::new(&index_layout);
    let mut residuals = Vec::<f64>::new();

    let mut rng = rand::thread_rng();

    let mut rows = Vec::<usize>::new();
    let mut cols = Vec::<usize>::new();
    let mut data = Vec::<f64>::new();

    if rank == 0 {
        for index in 0..n {
            rows.push(index);
            cols.push(index);
            data.push(rng.gen_range(1.0..=2.0));
        }
    }

    // The constructor takes care of the fact that the aij entries are only defined on rank 0.
    // It sends the entries around according to the index layout and constructs the parallel
    // distributed matrix.
    let distributed_mat =
        DistributedCsrMatrix::from_aij(&index_layout, &index_layout, &rows, &cols, &data, &world);

    // We can now wrap the matrix into an operator.
    let op = DistributedCsrMatrixOperator::new(&distributed_mat, &space, &space);

    // Let's create a right-hand side.
    let mut rhs = space.zero();
    rhs.view_mut()
        .local_mut()
        .fill_from_equally_distributed(&mut rng);
    // We need the vector x as well.

    // We can now run the CG iteration.
    let cg = (CgIteration::new(&op, &rhs))
        .set_callable(|_, res| {
            let res_norm = space.norm(res);
            residuals.push(res_norm);
        })
        .set_tol(tol);

    let (_sol, res) = cg.run();

    if rank == 0 {
        println!("The residual is: {}", res);
    }
}
