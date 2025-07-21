//! Definition of CSR matrices.

use std::collections::HashMap;
use std::ops::AddAssign;
use std::rc::Rc;

use itertools::{izip, Itertools};
use mpi::collective::CommunicatorCollectives;
use mpi::traits::{Communicator, Equivalence};

use crate::dense::array::DynArray;
use crate::distributed_tools::{redistribute, sort_to_bins, GhostCommunicator, IndexLayout};
use crate::{
    empty_array, AijIteratorByValue, AijIteratorMut, Array, ArrayIteratorByValue, AsMatrixApply,
    BaseItem, FromAij, FromAijDistributed, Len, Nonzeros, RandomAccessByValue, RawAccess, Shape,
    UnsafeRandomAccessByValue,
};

use super::csr_mat::CsrMatrix;
use super::distributed_array::DistributedArray;
use super::tools::normalize_aij;
use super::SparseMatType;

/// Distributed CSR matrix
pub struct DistributedCsrMatrix<'a, Item, C>
where
    C: Communicator,
{
    mat_type: SparseMatType,
    local_matrix: CsrMatrix<Item>,
    global_indices: DynArray<usize, 1>,
    local_dof_count: usize,
    domain_layout: Rc<IndexLayout<'a, C>>,
    range_layout: Rc<IndexLayout<'a, C>>,
    domain_ghosts: GhostCommunicator<usize>,
    local2global: Vec<usize>,
    global2local: HashMap<usize, usize>,
}

impl<'a, Item, C: Communicator> DistributedCsrMatrix<'a, Item, C>
where
    Item: Copy,
    C: Communicator,
    DynArray<Item, 1>: ArrayIteratorByValue<Item = Item>,
{
    /// Create a new distributed CSR matrix.
    pub fn new(
        indices: DynArray<usize, 1>,
        indptr: DynArray<usize, 1>,
        data: DynArray<Item, 1>,
        domain_layout: Rc<IndexLayout<'a, C>>,
        range_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self {
        // Both layouts must have the same communicator.
        assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

        // The indptr vector must have one more element than the number of local indices.
        assert_eq!(1 + range_layout.number_of_local_indices(), indptr.len());

        let comm = domain_layout.comm();

        let my_rank = comm.rank() as usize;

        let domain_ghost_dofs: Vec<usize> = indices
            .iter_value()
            .unique()
            .filter(|&dof| domain_layout.rank_from_index(dof).unwrap() != my_rank)
            .collect();

        let ranks = domain_ghost_dofs
            .iter()
            .map(|dof| domain_layout.rank_from_index(*dof).unwrap())
            .collect::<Vec<_>>();

        let domain_ghosts = GhostCommunicator::new(&domain_ghost_dofs, &ranks, comm);
        let local_dof_count =
            domain_layout.number_of_local_indices() + domain_ghosts.total_receive_count;
        let mut global2local = HashMap::new();

        // We need to transform the indices vector of the CSR matrix from global indexing to
        // local indexing. For this we assume that the input vector has the format
        // [local_indices..., ghost_indices]. So we map global indices to fit this format.
        // To do this we create a hash map that takes the global indices and maps to the new
        // local indexing.

        let mut count: usize = 0;
        for index in domain_layout.local_range().0..domain_layout.local_range().1 {
            global2local.insert(index, count);
            count += 1;
        }
        for index in &domain_ghosts.receive_indices {
            global2local.insert(*index, count);
            count += 1;
        }

        // The hash map is created. We now iterate through the indices vector of the sparse matrix
        // to change the indexing.

        let mapped_indices = indices
            .iter_value()
            .map(|elem| *global2local.get(&elem).unwrap())
            .collect::<Vec<_>>();

        // Now reverese the  index mapper to get the local to global mapping.

        let local2global = global2local
            .iter()
            .sorted_by_key(|&(_, &local_index)| local_index)
            .map(|(&global_index, _)| global_index)
            .collect::<Vec<_>>();

        Self {
            mat_type: SparseMatType::Csr,
            local_matrix: CsrMatrix::new(
                [
                    range_layout.number_of_local_indices(),
                    domain_layout.number_of_global_indices(),
                ],
                DynArray::from_shape_and_vec([mapped_indices.len()], mapped_indices),
                indptr,
                data,
            ),
            global_indices: indices,
            local_dof_count,
            domain_layout,
            range_layout,
            domain_ghosts,
            local2global,
            global2local,
        }
    }

    /// Return the local sparse matrix
    pub fn local(&self) -> &CsrMatrix<Item> {
        &self.local_matrix
    }

    /// Return a mutable reference to the local sparse matrix
    pub fn local_mut(&mut self) -> &mut CsrMatrix<Item> {
        &mut self.local_matrix
    }

    /// Domain layout
    pub fn domain_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.domain_layout.clone()
    }

    /// Range layout
    pub fn range_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.range_layout.clone()
    }

    /// Communicator
    pub fn comm(&self) -> &C {
        self.domain_layout.comm()
    }

    ///  Map a global domain index to a domain index in the local sparse matrix.
    /// Return None if the index does not exist.
    pub fn domain_global2local(&self, global_index: usize) -> Option<usize> {
        self.global2local.get(&global_index).copied()
    }

    /// Map a local domain index to a  global domain index.
    pub fn domain_local2global(&self, local_index: usize) -> Option<usize> {
        self.local2global.get(local_index).copied()
    }

    /// Map a global range index to a range index in the local sparse matrix.
    pub fn range_global2local(&self, global_index: usize) -> Option<usize> {
        let offset = self.range_layout.local_range().0;
        if global_index < offset
            || global_index >= offset + self.range_layout.number_of_local_indices()
        {
            None
        } else {
            Some(global_index - offset)
        }
    }

    ///  Map a local range index to a global range index.
    pub fn range_local2global(&self, local_index: usize) -> Option<usize> {
        let offset = self.range_layout.local_range().0;
        if local_index < self.range_layout.number_of_local_indices() {
            Some(local_index + offset)
        } else {
            None
        }
    }
}

impl<'a, Item, C: Communicator> Nonzeros for DistributedCsrMatrix<'a, Item, C>
where
    CsrMatrix<Item>: Nonzeros,
    Item: Copy,
{
    fn nnz(&self) -> usize {
        let local_result = self.local_matrix.nnz();
        let mut global_result = 0;

        self.comm().all_reduce_into(
            &local_result,
            &mut global_result,
            mpi::collective::SystemOperation::sum(),
        );

        global_result
    }
}

impl<'a, Item, C: Communicator> BaseItem for DistributedCsrMatrix<'a, Item, C> {
    type Item = Item;
}

impl<'a, Item, C: Communicator> Shape<2> for DistributedCsrMatrix<'a, Item, C>
where
    CsrMatrix<Item>: Shape<2>,
{
    fn shape(&self) -> [usize; 2] {
        [
            self.range_layout.number_of_global_indices(),
            self.domain_layout.number_of_global_indices(),
        ]
    }
}

impl<'a, Item, C: Communicator> AijIteratorByValue for DistributedCsrMatrix<'a, Item, C>
where
    Item: Copy + Default,
    CsrMatrix<Item>: AijIteratorByValue<Item = Item>,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        self.local_matrix.iter_aij_value().map(|(index, value)| {
            let global_row = index[0] + self.range_layout().local_range().0;
            let global_col = self.local2global[index[1]];
            ([global_row, global_col], value)
        })
    }
}

impl<'a, Item, C> FromAijDistributed<'a> for DistributedCsrMatrix<'a, Item, C>
where
    Item: Default + Copy + AddAssign + PartialEq + Equivalence,
    C: Communicator,
{
    type C = C;

    fn from_aij(
        domain_layout: Rc<IndexLayout<'a, Self::C>>,
        range_layout: Rc<IndexLayout<'a, Self::C>>,
        rows: &[usize],
        cols: &[usize],
        data: &[<Self as crate::BaseItem>::Item],
    ) -> Self {
        // Require the communicators to be identical.

        assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

        let comm = domain_layout.comm();

        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

        // We now exchange data across the processes that should not be on our node. We need to send to each node
        // the amount of data that it should get from us.
        // The data is already sorted by rows. So just need to iterate through and work out how much data each process
        // gets and the corresponding displacements.

        // First we create the index bounds

        let index_bounds = (0..comm.size())
            .map(|rank| range_layout.index_range(rank as usize).unwrap().0)
            .collect_vec();

        // Now we compute how many entries each process gets.
        let counts = sort_to_bins(&rows, &index_bounds)
            .iter()
            .map(|&x| x as i32)
            .collect_vec();

        let rows = redistribute(&rows, &counts, comm);
        let cols = redistribute(&cols, &counts, comm);
        let data = redistribute(&data, &counts, comm);

        // We now need to normalize again since processes could now again have elements at the same matrix positions being
        // sent over from different ranks.

        let (rows, cols, data) = normalize_aij(&rows, &cols, &data, SparseMatType::Csr);

        // We now have all the data at the right processes.
        // We can now create the indptr array.

        // First create the special case that there are no rows at our local process.

        if rows.is_empty() {
            // The index pointer of an empty sparse matrix still has one element.
            // It contains the total number of elements, namely zero.
            let indptr = empty_array::<_, 1>();
            let indices = empty_array::<_, 1>();
            let data = empty_array::<_, 1>();

            Self::new(indices, indptr, data, domain_layout, range_layout)
        } else {
            let mut indptr =
                Vec::<usize>::with_capacity(1 + range_layout.number_of_local_indices());
            let nelems = data.len();

            // The actual rows in the aij format start at a nonzero index
            // When we iterate through in the following loop we need to
            // take this into account.
            let first_row = range_layout.local_range().0;

            let mut count: usize = 0;
            for row in first_row..first_row + range_layout.number_of_local_indices() {
                indptr.push(count);
                while count < nelems && row == rows[count] {
                    count += 1;
                }
            }
            indptr.push(count);

            Self::new(
                cols.into(),
                indptr.into(),
                data.into(),
                domain_layout,
                range_layout,
            )
        }
    }
}

impl<'a, Item, C, ArrayImplX, ArrayImplY>
    AsMatrixApply<DistributedArray<'a, C, ArrayImplX, 1>, DistributedArray<'a, C, ArrayImplY, 1>, 1>
    for DistributedCsrMatrix<'a, Item, C>
where
    Item: Default + Copy + AddAssign + PartialEq + Equivalence,
    C: Communicator,
    ArrayImplX: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    Array<ArrayImplX, 1>: ArrayIteratorByValue<Item = Item>,
    CsrMatrix<Item>: AsMatrixApply<DynArray<Item, 1>, Array<ArrayImplY, 1>, 1, Item = Item>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &DistributedArray<'a, C, ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut DistributedArray<'a, C, ArrayImplY, 1>,
    ) {
        // Create a vector that combines local dofs and ghosts

        let my_rank = self.domain_layout.comm().rank() as usize;

        let out_values = {
            let mut out_values =
                Vec::<Self::Item>::with_capacity(self.domain_ghosts.total_send_count());
            let out_buff: &mut [Self::Item] =
                unsafe { std::mem::transmute(out_values.spare_capacity_mut()) };

            for (out, out_index) in
                izip!(out_buff.iter_mut(), self.domain_ghosts.send_indices.iter())
            {
                *out = x
                    .local
                    .get_value([self
                        .domain_layout()
                        .global2local(my_rank, *out_index)
                        .unwrap()])
                    .unwrap();
            }

            unsafe { out_values.set_len(self.domain_ghosts.total_send_count()) };
            out_values
        };

        let local_vec = {
            let mut local_vec = Vec::<Self::Item>::with_capacity(self.local_dof_count);
            local_vec.extend(x.local.iter_value());
            let ghost_data: &mut [Item] =
                unsafe { std::mem::transmute(local_vec.spare_capacity_mut()) };

            // Prepare the values that are sent to other ranks

            self.domain_ghosts
                .forward_send_values(&out_values, ghost_data);
            unsafe { local_vec.set_len(self.local_dof_count) };
            local_vec
        };

        // Compute result
        self.local_matrix
            .apply(alpha, &local_vec.into(), beta, &mut y.local);
    }
}

// impl<'a, T: RlstScalar + Equivalence, C: Communicator> DistributedCsrMatrix<'a, T, C> {
//     /// Create new
//     pub fn new(
//         indices: Vec<usize>,
//         indptr: Vec<usize>,
//         data: Vec<T>,
//         domain_layout: Rc<IndexLayout<'a, C>>,
//         range_layout: Rc<IndexLayout<'a, C>>,
//     ) -> Self {
//         assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));
//         let comm = domain_layout.comm();

//         let my_rank = comm.rank() as usize;

//         let domain_ghost_dofs: Vec<usize> = indices
//             .iter()
//             .unique()
//             .copied()
//             .filter(|&dof| domain_layout.rank_from_index(dof).unwrap() != my_rank)
//             .collect();

//         let ranks = domain_ghost_dofs
//             .iter()
//             .map(|dof| domain_layout.rank_from_index(*dof).unwrap())
//             .collect::<Vec<_>>();

//         let domain_ghosts = GhostCommunicator::new(&domain_ghost_dofs, &ranks, comm);
//         let local_dof_count =
//             domain_layout.number_of_local_indices() + domain_ghosts.total_receive_count;
//         let mut index_mapper = HashMap::new();

//         // We need to transform the indices vector of the CSR matrix from global indexing to
//         // local indexing. For this we assume that the input vector has the format
//         // [local_indices..., ghost_indices]. So we map global indices to fit this format.
//         // To do this we create a hash map that takes the global indices and maps to the new
//         // local indexing.

//         let mut count: usize = 0;
//         for index in domain_layout.local_range().0..domain_layout.local_range().1 {
//             index_mapper.insert(index, count);
//             count += 1;
//         }
//         for index in &domain_ghosts.receive_indices {
//             index_mapper.insert(*index, count);
//             count += 1;
//         }

//         // The hash map is created. We now iterate through the indices vector of the sparse matrix
//         // to change the indexing.

//         let mapped_indices = indices
//             .iter()
//             .map(|elem| *index_mapper.get(elem).unwrap())
//             .collect::<Vec<_>>();

//         Self {
//             mat_type: SparseMatType::Csr,
//             local_matrix: CsrMatrix::new(
//                 [indptr.len() - 1, domain_layout.number_of_global_indices()],
//                 mapped_indices,
//                 indptr,
//                 data,
//             ),
//             global_indices: indices,
//             local_dof_count,
//             domain_layout,
//             range_layout,
//             domain_ghosts,
//         }
//     }

//     /// Matrix type
//     pub fn mat_type(&self) -> &SparseMatType {
//         &self.mat_type
//     }

//     /// Local shape
//     pub fn local_shape(&self) -> [usize; 2] {
//         self.local_matrix.shape()
//     }

//     /// Column indices
//     pub fn indices(&self) -> &[usize] {
//         &self.global_indices
//     }

//     /// Indices at which each row starts
//     pub fn indptr(&self) -> &[usize] {
//         self.local_matrix.indptr()
//     }

//     /// Matrix entries
//     pub fn data(&self) -> &[T] {
//         self.local_matrix.data()
//     }

//     /// Domain layout
//     pub fn domain_layout(&self) -> Rc<IndexLayout<'a, C>> {
//         self.domain_layout.clone()
//     }

//     /// Range layout
//     pub fn range_layout(&self) -> Rc<IndexLayout<'a, C>> {
//         self.range_layout.clone()
//     }

//     /// Create a new distributed CSR matrix from an aij format.
//     pub fn from_aij(
//         domain_layout: Rc<IndexLayout<'a, C>>,
//         range_layout: Rc<IndexLayout<'a, C>>,
//         rows: &[usize],
//         cols: &[usize],
//         data: &[T],
//     ) -> Self {
//         // Require the communicators to be identical.

//         assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

//         let comm = domain_layout.comm();

//         let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

//         // We now exchange data across the processes that should not be on our node. We need to send to each node
//         // the amount of data that it should get from us.
//         // The data is already sorted by rows. So just need to iterate through and work out how much data each process
//         // gets and the corresponding displacements.

//         // First we create the index bounds

//         let index_bounds = (0..comm.size())
//             .map(|rank| range_layout.index_range(rank as usize).unwrap().0)
//             .collect_vec();

//         // Now we compute how many entries each process gets.
//         let counts = sort_to_bins(&rows, &index_bounds)
//             .iter()
//             .map(|&x| x as i32)
//             .collect_vec();

//         let rows = redistribute(&rows, &counts, comm);
//         let cols = redistribute(&cols, &counts, comm);
//         let data = redistribute(&data, &counts, comm);

//         // We now need to normalize again since processes could now again have elements at the same matrix positions being
//         // sent over from different ranks.

//         let (rows, cols, data) = normalize_aij(&rows, &cols, &data, SparseMatType::Csr);

//         // We now have all the data at the right processes.
//         // We can now create the indptr array.

//         // First create the special case that there are no rows at our local process.

//         if rows.is_empty() {
//             // The index pointer of an empty sparse matrix still has one element.
//             // It contains the total number of elements, namely zero.
//             let indptr = vec![0];
//             let indices = Vec::<usize>::new();
//             let data = Vec::<T>::new();

//             Self::new(indices, indptr, data, domain_layout, range_layout)
//         } else {
//             let mut indptr =
//                 Vec::<usize>::with_capacity(1 + range_layout.number_of_local_indices());
//             let nelems = data.len();

//             // The actual rows in the aij format start at a nonzero index
//             // When we iterate through in the following loop we need to
//             // take this into account.
//             let first_row = range_layout.local_range().0;

//             let mut count: usize = 0;
//             for row in first_row..first_row + range_layout.number_of_local_indices() {
//                 indptr.push(count);
//                 while count < nelems && row == rows[count] {
//                     count += 1;
//                 }
//             }
//             indptr.push(count);

//             Self::new(cols, indptr, data, domain_layout, range_layout)
//         }
//     }

//     /// Create from root
//     pub fn from_serial(
//         root: usize,
//         domain_layout: Rc<IndexLayout<'a, C>>,
//         range_layout: Rc<IndexLayout<'a, C>>,
//     ) -> Self {
//         assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));

//         let comm = domain_layout.comm();

//         let root_process = comm.process_at_rank(root as i32);

//         let my_index_range = range_layout.local_range();
//         let my_number_of_rows = my_index_range.1 - my_index_range.0;

//         let mut csr_data;
//         let mut csr_indices;
//         let mut csr_indptr;
//         let mut shape;
//         let mut my_data_count: i32 = 0;

//         shape = vec![0; 2];

//         // Receive the number of data entries.
//         root_process.scatter_into(&mut my_data_count);

//         // Now make space for the matrix data.

//         csr_data = vec![T::zero(); my_data_count as usize];
//         csr_indices = vec![0; my_data_count as usize];
//         csr_indptr = vec![0; 1 + my_number_of_rows];

//         // Get the matrix data
//         root_process.scatter_varcount_into(csr_data.as_mut_slice());
//         root_process.scatter_varcount_into(csr_indices.as_mut_slice());
//         root_process.scatter_varcount_into(&mut csr_indptr.as_mut_slice()[..my_number_of_rows]);
//         root_process.broadcast_into(shape.as_mut_slice());

//         // We need to fix the index pointer as locally it needs to start at 0. But
//         // the communicated data is offset by where the index pointer was on the
//         // root process.
//         let first_elem = csr_indptr[0];
//         csr_indptr[..my_number_of_rows]
//             .iter_mut()
//             .for_each(|elem| *elem -= first_elem);

//         csr_indptr[my_number_of_rows] = my_data_count as usize;

//         // Once everything is received we can finally create the matrix object.

//         Self::new(
//             csr_indices,
//             csr_indptr,
//             csr_data,
//             domain_layout,
//             range_layout,
//         )
//     }

//     /// Create from root
//     pub fn from_serial_root(
//         csr_mat: CsrMatrix<T>,
//         domain_layout: Rc<IndexLayout<'a, C>>,
//         range_layout: Rc<IndexLayout<'a, C>>,
//     ) -> Self {
//         assert!(std::ptr::addr_eq(domain_layout.comm(), range_layout.comm()));
//         let comm = domain_layout.comm();

//         let size = comm.size() as usize;
//         let root_process = comm.this_process();

//         let my_index_range = range_layout.local_range();
//         let my_number_of_rows = my_index_range.1 - my_index_range.0;

//         let mut csr_data;
//         let mut csr_indices;
//         let mut csr_indptr;
//         let mut shape;

//         // Need to compute how much data to send to each process.

//         let mut my_data_count: i32 = 0;

//         let mut counts: Vec<i32> = vec![0; size];

//         for (rank, item) in counts.iter_mut().enumerate() {
//             let local_index_range = range_layout.index_range(rank).unwrap();
//             *item = (csr_mat.indptr()[local_index_range.1] - csr_mat.indptr()[local_index_range.0])
//                 as i32;
//         }

//         // Send around how much data is received by everyone.

//         root_process.scatter_into_root(&counts, &mut my_data_count);

//         // Every process now knows how much data it gets. Now compute the displacements.

//         let mut count = 0;
//         let mut displacements = Vec::<i32>::with_capacity(size);

//         for n in &counts {
//             displacements.push(count);
//             count += n;
//         }

//         csr_data = vec![T::zero(); my_data_count as usize];
//         csr_indices = vec![0; my_data_count as usize];
//         csr_indptr = vec![0; 1 + my_number_of_rows];
//         shape = vec![csr_mat.shape()[0], csr_mat.shape()[1]];

//         // The following code computes the counts and displacements for the indexptr vector.

//         let mut idxptrcount: Vec<i32> = vec![0; size];
//         let mut idxptrdisplacements: Vec<i32> = vec![0; size];

//         for rank in 0..size {
//             let local_index_range = range_layout.index_range(rank).unwrap();
//             idxptrcount[rank] = (local_index_range.1 - local_index_range.0) as i32;
//             idxptrdisplacements[rank] = local_index_range.0 as i32;
//         }

//         // We now scatter the csr matrix data to the processes.

//         let data_partition = mpi::datatype::Partition::new(
//             csr_mat.data(),
//             counts.as_slice(),
//             displacements.as_slice(),
//         );

//         let indices_partition = mpi::datatype::Partition::new(
//             csr_mat.indices(),
//             counts.as_slice(),
//             displacements.as_slice(),
//         );

//         let idx_partition = mpi::datatype::Partition::new(
//             csr_mat.indptr(),
//             idxptrcount.as_slice(),
//             idxptrdisplacements.as_slice(),
//         );

//         // Send everything around
//         root_process.scatter_varcount_into_root(&data_partition, csr_data.as_mut_slice());
//         root_process.scatter_varcount_into_root(&indices_partition, csr_indices.as_mut_slice());
//         root_process.scatter_varcount_into_root(
//             &idx_partition,
//             &mut csr_indptr.as_mut_slice()[..my_number_of_rows],
//         );
//         root_process.broadcast_into(shape.as_mut_slice());

//         // We need to fix the index pointer as locally it needs to start at 0. But
//         // the communicated data is offset by where the index pointer was on the
//         // root process.
//         let first_elem = csr_indptr[0];
//         csr_indptr[..my_number_of_rows]
//             .iter_mut()
//             .for_each(|elem| *elem -= first_elem);

//         csr_indptr[my_number_of_rows] = my_data_count as usize;

//         // Once everything is received we can finally create the matrix object.

//         Self::new(
//             csr_indices,
//             csr_indptr,
//             csr_data,
//             domain_layout,
//             range_layout,
//         )
//     }

//     /// Matrix multiplication
//     pub fn matmul(
//         &self,
//         alpha: T,
//         x: &DistributedVector<'_, C, T>,
//         beta: T,
//         y: &mut DistributedVector<'_, C, T>,
//     ) {
//         // Create a vector that combines local dofs and ghosts

//         let my_rank = self.domain_layout.comm().rank() as usize;

//         let out_values = {
//             let mut out_values = Vec::<T>::with_capacity(self.domain_ghosts.total_send_count());
//             let out_buff: &mut [T] =
//                 unsafe { std::mem::transmute(out_values.spare_capacity_mut()) };

//             for (out, out_index) in
//                 izip!(out_buff.iter_mut(), self.domain_ghosts.send_indices.iter())
//             {
//                 *out = x.local().data()[self
//                     .domain_layout()
//                     .global2local(my_rank, *out_index)
//                     .unwrap()];
//             }

//             unsafe { out_values.set_len(self.domain_ghosts.total_send_count()) };
//             out_values
//         };

//         let local_vec = {
//             let mut local_vec = Vec::<T>::with_capacity(self.local_dof_count);
//             local_vec.extend(x.local().data().iter().copied());
//             let ghost_data: &mut [T] =
//                 unsafe { std::mem::transmute(local_vec.spare_capacity_mut()) };

//             // Prepare the values that are sent to other ranks

//             self.domain_ghosts
//                 .forward_send_values(&out_values, ghost_data);
//             unsafe { local_vec.set_len(self.local_dof_count) };
//             local_vec
//         };

//         // Compute result
//         self.local_matrix
//             .matmul(alpha, local_vec.as_slice(), beta, y.local_mut().data_mut());
//     }
// }

// impl<T: RlstScalar + Equivalence, C: Communicator> Shape<2> for DistributedCsrMatrix<'_, T, C> {
//     fn shape(&self) -> [usize; 2] {
//         [
//             self.range_layout().number_of_global_indices(),
//             self.domain_layout().number_of_global_indices(),
//         ]
//     }
// }
