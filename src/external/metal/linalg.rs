//! Metal based gemm implementation

use crate::{
    external::metal::interface::{MpsDataType, MpsMatrixMut},
    Array, AsRawMetalBuffer, AutoReleasePool, MetalDevice, RawAccess, RlstScalar, Shape, Stride,
    TransMode, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use super::{
    interface::{MpsMatrix, MpsMatrixDescriptor, MpsMatrixMultiplication},
    metal_array::AsRawMetalBufferMut,
};

impl<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = f32>
            + UnsafeRandomAccessMut<2, Item = f32>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = f32>
            + AsRawMetalBufferMut,
    > Array<f32, ArrayImpl, 2>
{
    /// Multiply `arr_a` x `arr_b` into `self`.
    pub fn metal_mult_into<
        ArrayImplA: UnsafeRandomAccessByValue<2, Item = f32>
            + Shape<2>
            + Stride<2>
            + AsRawMetalBuffer
            + RawAccess<Item = f32>,
        ArrayImplB: UnsafeRandomAccessByValue<2, Item = f32>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = f32>
            + AsRawMetalBuffer,
    >(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: f32,
        arr_a: Array<f32, ArrayImplA, 2>,
        arr_b: Array<f32, ArrayImplB, 2>,
        beta: f32,
    ) -> Self {
        assert!(
            is_row_major(&self),
            "Matrix multiplied into must be row major."
        );
        assert!(is_row_major(&arr_a), "Matrix `arr_a` must be row major.");
        assert!(is_row_major(&arr_b), "Matrix `arr_b` must be row major.");

        AutoReleasePool::execute(|| {
            let mat_a = as_mps_matrix(&arr_a);
            let mat_b = as_mps_matrix(&arr_b);

            let result_rows = self.shape()[0];
            let result_columns = self.shape()[1];
            let interior_columns = arr_a.shape()[1];

            let mut mat_c = as_mps_matrix_mut(&mut self);

            let transa = match transa {
                TransMode::NoTrans => false,
                TransMode::Trans => true,
                _ => panic!("Transposition mode not supported."),
            };

            let transb = match transb {
                TransMode::NoTrans => false,
                TransMode::Trans => true,
                _ => panic!("Transposition mode not supported."),
            };

            let device = MetalDevice::from_default();

            let queue = device.command_queue();
            let mut command_buffer = queue.command_buffer();

            let mat_mult = MpsMatrixMultiplication::new(
                &device,
                transa,
                transb,
                result_rows,
                result_columns,
                interior_columns,
                alpha.into(),
                beta.into(),
            );

            mat_mult.encode_to_command_buffer(&mut command_buffer, &mat_a, &mat_b, &mut mat_c);

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self
    }
}

/// Return a Metal Performance Shader Matrix from an array.
pub fn as_mps_matrix<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = f32>
        + Shape<2>
        + Stride<2>
        + RawAccess<Item = f32>
        + AsRawMetalBuffer,
>(
    arr: &Array<f32, ArrayImpl, 2>,
) -> MpsMatrix {
    let stride = arr.stride();
    let shape = arr.shape();
    let offset = arr.offset() * std::mem::size_of::<f32>();
    assert_eq!(stride[0], shape[1]);
    assert_eq!(stride[1], 1);

    let row_bytes = shape[1] * std::mem::size_of::<f32>();

    let desc = MpsMatrixDescriptor::new(
        shape[0],
        shape[1],
        1,
        row_bytes,
        row_bytes * shape[0],
        MpsDataType::F32,
    );

    MpsMatrix::new(arr.metal_buffer(), offset, desc)
}

/// Return a mutable Metal Performance Shader Matrix from an array.
pub fn as_mps_matrix_mut<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = f32>
        + Shape<2>
        + Stride<2>
        + RawAccess<Item = f32>
        + AsRawMetalBufferMut,
>(
    arr: &mut Array<f32, ArrayImpl, 2>,
) -> MpsMatrixMut {
    let stride = arr.stride();
    let shape = arr.shape();
    let offset = arr.offset() * std::mem::size_of::<f32>();
    assert_eq!(stride[0], shape[1]);
    assert_eq!(stride[1], 1);

    let row_bytes = shape[1] * std::mem::size_of::<f32>();

    let desc = MpsMatrixDescriptor::new(
        shape[0],
        shape[1],
        1,
        row_bytes,
        row_bytes * shape[0],
        MpsDataType::F32,
    );

    MpsMatrixMut::new(arr.metal_buffer_mut(), offset, desc)
}

fn is_row_major<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2>,
>(
    arr: &Array<Item, ArrayImpl, 2>,
) -> bool {
    let stride = arr.stride();
    let shape = arr.shape();

    (stride[0] == shape[1]) && (stride[1] == 1)
}

#[cfg(test)]
mod test {

    use rand::SeedableRng;

    use crate::prelude::*;

    #[test]
    fn test_metal_mat_mul() {
        let device = MetalDevice::from_default();

        let mut mat_a = rlst_metal_array3!(&device, f32, [2, 3, 4]);
        let mut mat_b = rlst_metal_array3!(&device, f32, [2, 4, 5]);
        let mut mat_c = rlst_metal_array3!(&device, f32, [2, 3, 5]);

        let mut mat_a_cpu = rlst_dynamic_array2!(f32, [3, 4]);
        let mut mat_b_cpu = rlst_dynamic_array2!(f32, [4, 5]);
        let mut expected = rlst_dynamic_array2!(f32, [3, 5]);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

        mat_a.fill_from_equally_distributed(&mut rng);
        mat_b.fill_from_equally_distributed(&mut rng);
        mat_c.fill_from_equally_distributed(&mut rng);

        mat_a_cpu.fill_from(mat_a.view().slice(0, 1));
        mat_b_cpu.fill_from(mat_b.view().slice(0, 1));
        expected.fill_from(mat_c.view().slice(0, 1));

        mat_c.view_mut().slice(0, 1).metal_mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            1.0,
            mat_a.view().slice(0, 1),
            mat_b.view().slice(0, 1),
            0.0,
        );

        expected.view_mut().mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            1.0,
            mat_a_cpu.view(),
            mat_b_cpu.view(),
            0.0,
        );

        crate::assert_array_relative_eq!(mat_c.view().slice(0, 1), expected, 1E-6);
    }
}
