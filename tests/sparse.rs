use rlst::prelude::*;

#[test]
fn test_csc_from_aij() {
    // Test the matrix [[1, 2], [3, 4]]
    let rows = vec![0, 0, 1, 1, 0];
    let cols = vec![0, 1, 0, 1, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

    let csc = CscMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

    assert_eq!(csc.data().len(), 4);
    assert_eq!(csc.indices().len(), 4);
    assert_eq!(csc.indptr().len(), 3);
    assert_eq!(csc.data()[2], 8.0);

    // Test the matrix [[0, 2.0, 0.0], [0, 0, 0], [0, 0, 0]]
    let rows = vec![0, 2, 0];
    let cols = vec![1, 2, 1];
    let data = vec![2.0, 0.0, 3.0];

    let csc = CscMatrix::from_aij([3, 3], &rows, &cols, &data).unwrap();

    assert_eq!(csc.indptr()[0], 0);
    assert_eq!(csc.indptr()[1], 0);
    assert_eq!(csc.indptr()[2], 1);
    assert_eq!(csc.indptr()[3], 1);
    assert_eq!(csc.data()[0], 5.0);
}

#[test]
fn test_csc_matmul() {
    // Test the matrix [[1, 2], [3, 4]]
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let csc = CscMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

    // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
    // Expected result is [35, 79].

    let x = vec![3.0, 4.0];
    let mut res = vec![1.0, 2.0];

    csc.matmul(3.0, &x, 2.0, &mut res);

    assert_eq!(res[0], 35.0);
    assert_eq!(res[1], 79.0);
}

#[test]
fn test_csc_aij_iterator() {
    let rows = vec![2, 3, 4, 4, 6];
    let cols = vec![1, 1, 3, 3, 4];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // This matrix has a zero col at the beginning one in between and several zero cols at the end.
    let csr = CscMatrix::from_aij([10, 20], &rows, &cols, &data).unwrap();

    let aij_data: Vec<(usize, usize, f64)> = csr.iter_aij().collect();

    assert_eq!(aij_data.len(), 4);

    assert_eq!(aij_data[0], (2, 1, 1.0));
    assert_eq!(aij_data[1], (3, 1, 2.0));
    assert_eq!(aij_data[2], (4, 3, 7.0));
    assert_eq!(aij_data[3], (6, 4, 5.0));
}
#[test]
fn test_csr_from_aij() {
    // Test the matrix [[1, 2], [3, 4]]
    let rows = vec![0, 0, 1, 1, 0];
    let cols = vec![0, 1, 0, 1, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

    let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

    assert_eq!(csr.data().len(), 4);
    assert_eq!(csr.indices().len(), 4);
    assert_eq!(csr.indptr().len(), 3);
    assert_eq!(csr.data()[1], 8.0);

    //Test the matrix [[0, 0, 0], [2.0, 0, 0], [0, 0, 0]]
    let rows = vec![1, 2, 1];
    let cols = vec![0, 2, 0];
    let data = vec![2.0, 0.0, 3.0];

    let csr = CsrMatrix::from_aij([3, 3], &rows, &cols, &data).unwrap();

    assert_eq!(csr.indptr()[0], 0);
    assert_eq!(csr.indptr()[1], 0);
    assert_eq!(csr.indptr()[2], 1);
    assert_eq!(csr.indptr()[3], 1);
    assert_eq!(csr.data()[0], 5.0);
}

#[test]
fn test_csr_matmul() {
    // Test the matrix [[1, 2], [3, 4]]
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

    // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
    // Expected result is [35, 79].

    let x = vec![3.0, 4.0];
    let mut res = vec![1.0, 2.0];

    csr.matmul(3.0, &x, 2.0, &mut res);

    assert_eq!(res[0], 35.0);
    assert_eq!(res[1], 79.0);
}

#[test]
fn test_csr_aij_iterator() {
    let rows = vec![2, 3, 4, 4, 6];
    let cols = vec![0, 1, 0, 2, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // This matrix has a zero row at the beginning one in between and several zero rows at the end.
    let csr = CsrMatrix::from_aij([10, 3], &rows, &cols, &data).unwrap();

    let aij_data: Vec<(usize, usize, f64)> = csr.iter_aij().collect();

    assert_eq!(aij_data.len(), 5);

    assert_eq!(aij_data[0], (2, 0, 1.0));
    assert_eq!(aij_data[1], (3, 1, 2.0));
    assert_eq!(aij_data[2], (4, 0, 3.0));
    assert_eq!(aij_data[3], (4, 2, 4.0));
    assert_eq!(aij_data[4], (6, 1, 5.0));
}

#[cfg(feature = "mpi")]
#[test]
fn test_distributed_index_set() {
    use mpi;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let index_layout = DefaultDistributedIndexLayout::new(14, 1, &world);

    // Test that the range is correct on rank 0
    assert_eq!(index_layout.index_range(0).unwrap(), (0, 14));

    // Test that the number of global indices is correct.
    assert_eq!(index_layout.number_of_global_indices(), 14);

    // Test that map works

    assert_eq!(index_layout.local2global(2).unwrap(), 2);

    // Return the correct process for an index

    assert_eq!(index_layout.rank_from_index(5).unwrap(), 0);
}

#[cfg(feature = "suitesparse")]
#[test]
fn test_csc_umfpack_f64() {
    let n = 5;

    let mut mat = rlst_dynamic_array2!(f64, [n, n]);
    let mut x_exact = rlst_dynamic_array1!(f64, [n]);
    let mut x_actual = rlst_dynamic_array1!(f64, [n]);

    mat.fill_from_seed_equally_distributed(0);
    x_exact.fill_from_seed_equally_distributed(1);

    let rhs = empty_array::<f64, 1>().simple_mult_into_resize(mat.r(), x_exact.r());

    let mut rows = Vec::<usize>::with_capacity(n * n);
    let mut cols = Vec::<usize>::with_capacity(n * n);
    let mut data = Vec::<f64>::with_capacity(n * n);

    for col_index in 0..n {
        for row_index in 0..n {
            rows.push(row_index);
            cols.push(col_index);
            data.push(mat[[row_index, col_index]]);
        }
    }

    let sparse_mat = CscMatrix::from_aij([n, n], &rows, &cols, &data).unwrap();

    sparse_mat
        .into_lu()
        .unwrap()
        .solve(rhs.r(), x_actual.r_mut(), TransMode::NoTrans)
        .unwrap();

    rlst::assert_array_relative_eq!(x_actual, x_exact, 1E-12);
}

#[cfg(feature = "suitesparse")]
#[test]
fn test_csc_umfpack_c64() {
    let n = 5;

    let mut mat = rlst_dynamic_array2!(c64, [n, n]);
    let mut x_exact = rlst_dynamic_array1!(c64, [n]);
    let mut x_actual = rlst_dynamic_array1!(c64, [n]);

    mat.fill_from_seed_equally_distributed(0);
    x_exact.fill_from_seed_equally_distributed(1);

    let rhs = empty_array::<c64, 1>().simple_mult_into_resize(mat.r(), x_exact.r());

    let mut rows = Vec::<usize>::with_capacity(n * n);
    let mut cols = Vec::<usize>::with_capacity(n * n);
    let mut data = Vec::<c64>::with_capacity(n * n);

    for col_index in 0..n {
        for row_index in 0..n {
            rows.push(row_index);
            cols.push(col_index);
            data.push(mat[[row_index, col_index]]);
        }
    }

    let sparse_mat = CscMatrix::from_aij([n, n], &rows, &cols, &data).unwrap();

    sparse_mat
        .into_lu()
        .unwrap()
        .solve(rhs.r(), x_actual.r_mut(), TransMode::NoTrans)
        .unwrap();

    rlst::assert_array_relative_eq!(x_actual, x_exact, 1E-12);
}
