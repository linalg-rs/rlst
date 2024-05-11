//! Implementation of an RLST Array over a Metal Buffer.

use super::interface::{MetalBuffer, MetalDevice};

/// A container built around Metal buffers.
pub struct MetalDataBuffer {
    #[allow(dead_code)]
    metal_buff: MetalBuffer,
    number_of_elements: usize,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

/// Return a raw metal buffer
pub trait AsRawMetalBuffer {
    /// Return reference to raw metal buffer.
    fn metal_buffer(&self) -> &MetalBuffer;
}

/// Return a mutable raw metal buffer
pub trait AsRawMetalBufferMut: AsRawMetalBuffer {
    /// Return mutable reference to raw metal buffer.
    fn metal_buffer_mut(&mut self) -> &mut MetalBuffer;
}

impl MetalDataBuffer {
    /// Initialize a new Metal Data Buffer.
    pub fn new(device: &MetalDevice, number_of_elements: usize, options: u32) -> Self {
        let buff = device.new_buffer(number_of_elements * std::mem::size_of::<f32>(), options);
        Self {
            metal_buff: buff,
            number_of_elements,
        }
    }
}

impl AsRawMetalBuffer for MetalDataBuffer {
    fn metal_buffer(&self) -> &MetalBuffer {
        &self.metal_buff
    }
}

impl AsRawMetalBufferMut for MetalDataBuffer {
    fn metal_buffer_mut(&mut self) -> &mut MetalBuffer {
        &mut self.metal_buff
    }
}

impl MetalDataBuffer {
    /// Return a reference to the metal buffer
    pub fn metal_buffer(&self) -> &MetalBuffer {
        &self.metal_buff
    }

    /// Return a mutable reference to the metal buffer
    pub fn metal_buffer_mut(&mut self) -> &mut MetalBuffer {
        &mut self.metal_buff
    }
}

impl crate::dense::data_container::DataContainer for MetalDataBuffer {
    type Item = f32;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        self.data()[index]
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        &self.data()[index]
    }

    fn number_of_elements(&self) -> usize {
        self.number_of_elements
    }

    fn data(&self) -> &[Self::Item] {
        unsafe {
            std::slice::from_raw_parts(
                self.metal_buff.raw_ptr() as *const f32,
                self.number_of_elements,
            )
        }
    }
}

impl crate::dense::data_container::DataContainerMut for MetalDataBuffer {
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        &mut self.data_mut()[index]
    }

    fn data_mut(&mut self) -> &mut [Self::Item] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.metal_buff.raw_ptr_mut() as *mut f32,
                self.number_of_elements,
            )
        }
    }
}

/// Create a new one dimensional metal array.
#[macro_export]
macro_rules! rlst_metal_array1 {
    ($device:expr, f32, $shape:expr) => {{
        let container = $crate::external::metal::metal_array::MetalDataBuffer::new(
            $device,
            $shape[0],
            $crate::external::metal::interface::ResourceOptions::HazardTrackingModeUntracked as u32,
        );
        $crate::dense::array::Array::new($crate::dense::base_array::BaseArray::new(
            container, $shape,
        ))
    }};
}

/// Create a new two dimensional metal array.
///
/// Note, the array uses a row oriented C-style ordering of elements
/// as opposed to the column oriented Fortray-Style ordering of default
/// RLST arrays. The reason is the default ordering of matrices for Apple Metal.
#[macro_export]
macro_rules! rlst_metal_array2 {
    ($device:expr, f32, $shape:expr) => {{
        let container = $crate::external::metal::metal_array::MetalDataBuffer::new(
            $device,
            $shape.iter().product(),
            $crate::external::metal::interface::ResourceOptions::HazardTrackingModeUntracked as u32,
        );
        $crate::dense::array::Array::new($crate::dense::base_array::BaseArray::new_with_stride(
            container,
            $shape,
            [$shape[1], 1],
        ))
    }};
}
