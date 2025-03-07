//! Tools for sparse matrix handling

use crate::dense::types::RlstScalar;
use crate::sparse::sparse_mat::SparseMatType;

/// Normalize an Aij matrix.
///
/// Returns a new tuple (rows, cols, data) that has been normalized.
/// This means that duplicate entries have been summed up, zero entries
/// deleted and the entries sorted row-wise with increasing column indicies
/// (if `sort_mode` is [SparseMatType::Csr]) or column-wise with increasing
/// row indices (if `sort_mode` is [SparseMatType::Csc]).
pub fn normalize_aij<T: RlstScalar>(
    rows: &[usize],
    cols: &[usize],
    data: &[T],
    sort_mode: SparseMatType,
) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    let nelems = data.len();

    assert_eq!(
        rows.len(),
        data.len(),
        "Number of rows {} not equal to number of data entries {}.",
        rows.len(),
        data.len()
    );

    assert_eq!(
        cols.len(),
        data.len(),
        "Number of columns {} not equal to number of data entries {}.",
        cols.len(),
        data.len()
    );

    let mut new_rows = Vec::<usize>::with_capacity(nelems);
    let mut new_cols = Vec::<usize>::with_capacity(nelems);
    let mut new_data = Vec::<T>::with_capacity(nelems);

    let mut sorted: Vec<usize> = (0..nelems).collect();

    // Depending on Sparse Matrix Mode sort first by columns
    // then by rows (Csr) or first by rows and then by columns
    // Csc

    match sort_mode {
        SparseMatType::Csc => {
            sorted.sort_by_key(|&idx| rows[idx]);
            sorted.sort_by_key(|&idx| cols[idx]);
        }
        SparseMatType::Csr => {
            sorted.sort_by_key(|&idx| cols[idx]);
            sorted.sort_by_key(|&idx| rows[idx]);
        }
    }

    // Now sum up equal entries

    let mut count: usize = 0;
    while count < nelems {
        let current_row = rows[sorted[count]];
        let current_col = cols[sorted[count]];
        let mut current_data = T::zero();
        while count < nelems
            && rows[sorted[count]] == current_row
            && cols[sorted[count]] == current_col
        {
            current_data += data[sorted[count]];
            count += 1;
        }
        if current_data != T::zero() {
            new_rows.push(current_row);
            new_cols.push(current_col);
            new_data.push(current_data);
        }
    }
    new_rows.shrink_to_fit();
    new_cols.shrink_to_fit();
    new_data.shrink_to_fit();

    (new_rows, new_cols, new_data)
}
