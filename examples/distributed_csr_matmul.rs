//! Demonstrate the creation of a distributed CSR matrix and its multiplication with a vector.

use mpi::traits::Communicator;
use rlst::{assert_array_relative_eq, prelude::*};

use rlst::io::matrix_market::{read_array_mm, read_coordinate_mm};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let dist_mat;

    let domain_layout = EquiDistributedIndexLayout::new(313, 1, &world);

    let mut dist_x = DistributedVector::<f64, _>::new(&domain_layout);

    let range_layout = EquiDistributedIndexLayout::new(507, 1, &world);

    let mut dist_y = DistributedVector::<f64, _>::new(&range_layout);

    if rank == 0 {
        // Read the sparse matrix in matrix market format.
        let sparse_mat = read_coordinate_mm::<f64>("mat_507_313.mm").unwrap();

        // Read the vector x. Note that the matrix market format mandates two dimensions for arrays.
        // So the vector is returned as two-dimensional array with the column dimension being 1.
        let x = read_array_mm::<f64>("x_313.mm").unwrap();

        // Read the expected result vector in matrix market format.
        let y_expected = read_array_mm::<f64>("y_507.mm").unwrap();

        // Create a new vector to store the actual matrix-vector product.
        let mut y_actual = rlst_dynamic_array1!(f64, [507]);

        // Execute the matrix-vector product on just the first rank.
        sparse_mat.matmul(1.0, x.data(), 0.0, y_actual.data_mut());

        // Make sure that the actual result matches the expected result.
        // We need to slice the expected result to remove the column dimension.
        assert_array_relative_eq!(y_actual, y_expected.r().slice(1, 0), 1E-12);

        let mut rows = Vec::<usize>::new();
        let mut cols = Vec::<usize>::new();
        let mut data = Vec::<f64>::new();
        sparse_mat.iter_aij().for_each(|(row, col, item)| {
            rows.push(row);
            cols.push(col);
            data.push(item);
        });

        dist_mat = DistributedCsrMatrix::from_aij(
            &domain_layout,
            &range_layout,
            &rows,
            &cols,
            &data,
            &world,
        );

        // dist_mat = DistributedCsrMatrix::from_serial_root(
        //     sparse_mat,
        //     &domain_layout,
        //     &range_layout,
        //     &world,
        // );

        dist_x.scatter_from_root(x.r().slice(1, 0));

        dist_mat.matmul(1.0, &dist_x, 0.0, &mut dist_y);

        // It is not needed to set y_actual to zero. But we want to make sure that
        // the correct data is not already contained in the vector for testing purpose.
        y_actual.set_zero();
        dist_y.gather_to_rank_root(y_actual.r_mut());

        // Make sure that the actual result matches the expected result.
        // We need to slice the expected result to remove the column dimension.
        assert_array_relative_eq!(y_actual, y_expected.r().slice(1, 0), 1E-12);

        println!("Distributed matrix-vector product successfully executed.");
    } else {
        // Create a distributed matrix on the non-root node (compare to `from_serial_root`).
        //dist_mat = DistributedCsrMatrix::from_serial(0, &domain_layout, &range_layout, &world);
        dist_mat = DistributedCsrMatrix::from_aij(
            &domain_layout,
            &range_layout,
            &Vec::default(),
            &Vec::default(),
            &Vec::default(),
            &world,
        );

        // Distribute the vector x.
        dist_x.scatter_from(0);

        // Execute the distributed matmul.
        dist_mat.matmul(1.0, &dist_x, 0.0, &mut dist_y);

        // Send the information back to root.
        dist_y.gather_to_rank(0);
    }
}
