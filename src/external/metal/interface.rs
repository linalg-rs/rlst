//! Interface routines

use std::ffi::c_void;

use super::raw;

#[derive(Clone, Copy)]
#[repr(u32)]
/// Metal Resource Options.
pub enum ResourceOptions {
    /// The app must prevent hazards when modifying memory.
    HazardTrackingModeUntracked =
        raw::RLSTMtlResourceOptions_RLST_MTL_RESOURCE_HAZARD_TRACKING_MODE_UNTRACKED,
    /// Metal prevents hazards when modifying content.
    HazardTrackingModeTracked =
        raw::RLSTMtlResourceOptions_RLST_MTL_RESOURCE_HAZARD_TRACKING_MODE_TRACKED,
    /// Optimized for resources that the CPU writes into but never reads.
    CpuCacheModeWriteCombined =
        raw::RLSTMtlResourceOptions_RLST_MTL_RESOURCE_CPU_CACHE_MODE_WRITE_COMBINED,
}

#[derive(Clone, Copy)]
#[repr(u32)]
/// Available Metal data types.
pub enum MpsDataType {
    /// Single precision floating point.
    F32 = raw::RLSTMtlMpsDataType_RLST_MTL_MPS_FLOAT32,
}

macro_rules! ptr_not_null {
    ($cmd:stmt, $err_msg:literal) => {{
        let p = unsafe { $cmd };
        if p.is_null() {
            panic!($err_msg);
        }
        p
    }};
}

/// Holds on Objective-C autorelease pool.
pub struct AutoReleasePool {
    pool: *mut raw::rlst_mtl_autorelease_pool_s,
}

impl AutoReleasePool {
    /// Execute `fun` within an autorelease pool.
    pub fn execute(mut fun: impl FnMut()) {
        let pool = ptr_not_null!(
            raw::rlst_mtl_new_autorelease_pool(),
            "Could not create AutoReleasePool"
        );
        fun();
        unsafe { raw::rlst_mtl_autorelease_pool_drain(pool) };
    }

    /// Show available pools
    pub fn show_pools() {
        unsafe { raw::rlst_mtl_autorelease_pool_show_pools() };
    }

    /// Initialize a new autorelease pool.
    pub fn new() -> Self {
        Self {
            pool: ptr_not_null!(
                raw::rlst_mtl_new_autorelease_pool(),
                "Could not create AutoReleasePool"
            ),
        }
    }

    /// Drain the autorelease pool.
    pub fn drain(self) {
        unsafe { raw::rlst_mtl_autorelease_pool_drain(self.pool) };
    }
}

/// Hods a Metal device context.
pub struct MetalDevice {
    device_p: raw::rlst_mtl_device_p,
}

impl std::ops::Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe { raw::rlst_mtl_device_release(self.device_p) }
    }
}

impl MetalDevice {
    /// Create a new device context from the available default device.
    pub fn from_default() -> Self {
        Self {
            device_p: ptr_not_null!(
                raw::rlst_mtl_new_default_device(),
                "Could not create new default device."
            ),
        }
    }

    /// Create a new Metal buffer.
    ///
    /// Create a new buffer with `nbytes` bytes. The options are a combination of options
    /// from [ResourceOptions].
    pub fn new_buffer(&self, nbytes: usize, options: u32) -> MetalBuffer {
        MetalBuffer::new(self, nbytes, options)
    }

    /// Return the name of the device.
    pub fn name(&self) -> String {
        unsafe { std::ffi::CStr::from_ptr(raw::rlst_mtl_device_name(self.device_p)) }
            .to_str()
            .map(|s| s.to_owned())
            .unwrap()
    }

    /// Create a command queue.
    pub fn command_queue(&self) -> MetalCommandQueue {
        MetalCommandQueue {
            queue_p: ptr_not_null!(
                raw::rlst_mtl_device_new_command_queue(self.device_p),
                "Could not create command queue."
            ),
        }
    }
}

/// Metal Command Queue.
pub struct MetalCommandQueue {
    queue_p: raw::rlst_mtl_command_queue_p,
}

impl MetalCommandQueue {
    /// Create a command buffer.
    pub fn command_buffer(&self) -> MetalCommandBuffer {
        MetalCommandBuffer {
            command_buffer_p: ptr_not_null!(
                raw::rlst_mtl_command_queue_command_buffer(self.queue_p),
                "Could not create command buffer."
            ),
        }
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe {
            raw::rlst_mtl_command_queue_release(self.queue_p);
        }
    }
}

/// Holds a Metal buffer.
pub struct MetalBuffer {
    buffer_p: raw::rlst_mtl_buffer_p,
    nbytes: usize,
}

impl MetalBuffer {
    /// Create a new Metal buffer.
    ///
    /// # This function should not be called directly. Instead [MetalDevice::new_buffer] should be used.
    pub fn new(device: &MetalDevice, nbytes: usize, options: u32) -> MetalBuffer {
        Self {
            buffer_p: ptr_not_null!(
                raw::rlst_mtl_device_new_buffer(device.device_p, nbytes as u64, options),
                "Could not create Metal buffer."
            ),
            nbytes,
        }
    }

    /// Get write access to the contents of the buffer.
    ///
    /// The number of bytes in the buffer must by a multiple of the number of bytes
    /// require for the type `T`.
    pub fn contents_mut<T: Sized>(&mut self) -> &mut [T] {
        let ptr = ptr_not_null!(
            raw::rlst_mtl_buffer_contents(self.buffer_p),
            "Could not access buffer contents"
        );
        assert_eq!(
            self.nbytes % std::mem::size_of::<T>(),
            0,
            "Number of bytes ({}) not compatible with type size ({}).",
            self.nbytes,
            std::mem::size_of::<T>()
        );

        unsafe {
            std::slice::from_raw_parts_mut(ptr as *mut T, self.nbytes / std::mem::size_of::<T>())
        }
    }

    /// Get read access to the contents of the buffer.
    ///
    /// The number of bytes in the buffer must by a multiple of the number of bytes
    /// require for the type `T`.
    pub fn contents<T: Sized>(&self) -> &[T] {
        let ptr = ptr_not_null!(
            raw::rlst_mtl_buffer_contents(self.buffer_p),
            "Could not access buffer contents"
        );
        assert_eq!(
            self.nbytes % std::mem::size_of::<T>(),
            0,
            "Number of bytes ({}) not compatible with type size ({}).",
            self.nbytes,
            std::mem::size_of::<T>()
        );

        unsafe {
            std::slice::from_raw_parts(ptr as *const T, self.nbytes / std::mem::size_of::<T>())
        }
    }

    /// Return a raw ptr to the contents.
    pub fn raw_ptr(&self) -> *const c_void {
        ptr_not_null!(
            raw::rlst_mtl_buffer_contents(self.buffer_p),
            "Could not access buffer contents"
        )
    }

    /// Return a mutable raw pointer to the contents.
    pub fn raw_ptr_mut(&mut self) -> *mut c_void {
        ptr_not_null!(
            raw::rlst_mtl_buffer_contents(self.buffer_p),
            "Could not access buffer contents"
        )
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe {
            raw::rlst_mtl_buffer_release(self.buffer_p);
        }
    }
}

/// Helds a Metal command buffer.
pub struct MetalCommandBuffer {
    command_buffer_p: raw::rlst_mtl_command_buffer_p,
}

impl MetalCommandBuffer {
    /// Commit a command buffer.
    pub fn commit(&self) {
        unsafe { raw::rlst_mtl_command_buffer_commit(self.command_buffer_p) };
    }

    /// Wait until a command buffer is completed.
    pub fn wait_until_completed(&self) {
        unsafe { raw::rlst_mtl_command_buffer_wait_until_completed(self.command_buffer_p) };
    }
}

/// Holds a Metal Matrix descriptor.
pub struct MpsMatrixDescriptor {
    desc: raw::rlst_mtl_mps_matrix_descriptor_p,
}

impl MpsMatrixDescriptor {
    /// Create a new matrix descriptor.
    ///
    /// # Arguments
    /// - `rows` - The number of rows in a single matrix.
    /// - `columns` - The number of columns in a single matrix.
    /// - `matrices` - The number of matrices encoded.
    /// - `row_bytes` - The stride from one row to the next in bytes.
    /// - `matrix_bytes` - The stride from one matrix to the next in bytes.
    /// - `data_type` - The data type of the matrix.
    pub fn new(
        rows: usize,
        columns: usize,
        matrices: usize,
        row_bytes: usize,
        matrix_bytes: usize,
        data_type: MpsDataType,
    ) -> Self {
        Self {
            desc: ptr_not_null!(
                raw::rlst_mtl_mps_matrix_descriptor(
                    rows as u64,
                    columns as u64,
                    matrices as u64,
                    row_bytes as u64,
                    matrix_bytes as u64,
                    data_type as u32,
                ),
                "Could not create MPS Matrix descriptor."
            ),
        }
    }

    /// Return the recommended row stride in bytes from the number of columns and the data type.
    pub fn row_bytes_from_columns(columns: usize, data_type: MpsDataType) -> usize {
        unsafe {
            raw::rlst_mtl_mps_matrix_descriptor_row_bytes_from_columns(
                columns as u64,
                data_type as u32,
            )
        }
    }

    /// The number of rows in a matrix.
    pub fn rows(&self) -> usize {
        unsafe { raw::rlst_mtl_mps_matrix_descriptor_rows(self.desc) as usize }
    }

    /// The number of columns in a matrix.
    pub fn columns(&self) -> usize {
        unsafe { raw::rlst_mtl_mps_matrix_descriptor_columns(self.desc) as usize }
    }

    /// The number of matrices.
    pub fn matrices(&self) -> usize {
        unsafe { raw::rlst_mtl_mps_matrix_descriptor_matrices(self.desc) as usize }
    }

    /// The row stride in bytes.
    pub fn row_bytes(&self) -> usize {
        unsafe { raw::rlst_mtl_mps_matrix_descriptor_row_bytes(self.desc) as usize }
    }

    /// The stride from one matrix to the next in bytes.
    pub fn matrix_bytes(&self) -> usize {
        unsafe { raw::rlst_mtl_mps_matrix_descriptor_matrix_bytes(self.desc) as usize }
    }
}

/// Holds a Metal Performance Shaders Matrix.
pub struct MpsMatrix {
    matrix_p: raw::rlst_mtl_mps_matrix_p,
    buffer: MetalBuffer,
    descriptor: MpsMatrixDescriptor,
}

impl MpsMatrix {
    /// Initialize a new Metal Performance Shaders Matrix.
    pub fn new(buffer: MetalBuffer, descriptor: MpsMatrixDescriptor) -> Self {
        Self {
            matrix_p: ptr_not_null!(
                raw::rlst_mtl_mps_matrix(buffer.buffer_p, descriptor.desc),
                "Could not create MpsMatrix."
            ),
            buffer,
            descriptor,
        }
    }

    /// Get the underlying buffer for the matrix.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    /// Get the descriptor of the matrix.
    pub fn descriptor(&self) -> &MpsMatrixDescriptor {
        &self.descriptor
    }

    /// Get a mutable slice of the contents of the matrix.
    pub fn contents_mut<T: Sized>(&mut self) -> &mut [T] {
        self.buffer.contents_mut()
    }

    /// Get a slice of the contents of the matrix.
    pub fn contents<T: Sized>(&self) -> &[T] {
        self.buffer.contents()
    }
}

impl Drop for MpsMatrix {
    fn drop(&mut self) {
        unsafe {
            raw::rlst_mtl_mps_matrix_release(self.matrix_p);
        }
    }
}

/// Holds a Metal Performance Shaders matrix multiplication object.
pub struct MpsMatrixMultiplication {
    matrix_mult_p: raw::rlst_mtl_mps_matrix_multiplication_p,
}

impl MpsMatrixMultiplication {
    /// Create a new Metal Performance Shaders matri multiplication object.
    ///
    /// This structure encodes the operation `C = alpha * op(A) * op(B) + beta C`,
    /// where `op` can be the transpose of the matrix.
    ///
    /// # Arguments
    /// `device` - The underlying Metal device.
    /// `transpose_left` - If true the left matrix is transposed.
    /// `transpose_rigth` - If true the right matrix is transposed.
    /// `result_rows` - The number of rows in the result matrix.
    /// `interior_columns` - The number of columns of `op(A)`.
    /// `alpha` - A double precision parameter.
    /// `beta` - A double precision parameter.
    pub fn new(
        device: &MetalDevice,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Self {
        Self {
            matrix_mult_p: ptr_not_null!(
                raw::rlst_mtl_mps_matrix_multiplication(
                    device.device_p,
                    transpose_left,
                    transpose_right,
                    result_rows as u64,
                    result_columns as u64,
                    interior_columns as u64,
                    alpha,
                    beta,
                ),
                "Could not create MpsMatrixMultiplication"
            ),
        }
    }

    /// Encode a matrix multiplication to a command buffer.
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &mut MetalCommandBuffer,
        left_matrix: &MpsMatrix,
        right_matrix: &MpsMatrix,
        result_matrix: &mut MpsMatrix,
    ) {
        unsafe {
            raw::rlst_mtl_mps_matrix_multiplication_encode_to_command_buffer(
                self.matrix_mult_p,
                command_buffer.command_buffer_p,
                left_matrix.matrix_p,
                right_matrix.matrix_p,
                result_matrix.matrix_p,
            );
        }
    }
}

impl Drop for MpsMatrixMultiplication {
    fn drop(&mut self) {
        unsafe {
            raw::rlst_mtl_mps_matrix_multiplication_release(self.matrix_mult_p);
        }
    }
}
