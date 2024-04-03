//! Implementation of an RLST Array over a Metal Buffer.

use super::interface::{MetalBuffer, MetalDevice};

/// A container built around Metal buffers.
pub struct MetalDataContainer {
    metal_buff: MetalBuffer,
    number_of_elements: usize,
    data: *mut f32,
}

impl MetalDataContainer {
    pub fn new(device: &MetalDevice, number_of_elements: usize, options: u32) -> Self {
        let mut buff = device.new_buffer(number_of_elements * std::mem::size_of::<f32>(), options);
        let data = (buff.raw_ptr_mut() as *mut f32);
        Self {
            metal_buff: buff,
            data,
            number_of_elements,
        }
    }
}

impl crate::dense::data_container::DataContainer for MetalDataContainer {
    type Item = f32;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        *self.data.add(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        &(*self.data.add(index))
    }

    fn number_of_elements(&self) -> usize {
        self.number_of_elements
    }

    fn data(&self) -> &[Self::Item] {
        unsafe { std::slice::from_raw_parts(self.data, self.number_of_elements) }
    }
}

impl crate::dense::data_container::DataContainerMut for MetalDataContainer {
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        &mut (*self.data.add(index))
    }

    fn data_mut(&mut self) -> &mut [Self::Item] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.number_of_elements) }
    }
}

/// Create a new one dimensional metal array.
#[macro_export]
macro_rules! rlst_metal_array1 {
    ($device:expr, f32, $shape:expr) => {{
        let container = $crate::external::metal::metal_array::MetalDataContainer::new(
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
        let container = $crate::external::metal::metal_array::MetalDataContainer::new(
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
