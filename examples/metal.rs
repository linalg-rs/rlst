//! Using RLST together with Apple Metal acceleration

use rlst::dense::tools::PrettyPrint;
use rlst::external::metal::{AutoReleasePool, MetalDevice};
use rlst::{prelude::*, rlst_metal_array2};

pub fn main() {
    AutoReleasePool::execute(|| {
        let device = MetalDevice::from_default();

        let mut arr = rlst_metal_array2!(&device, f32, [2, 3]);

        for (index, elem) in arr.iter_mut().enumerate() {
            *elem = index as f32;
        }

        arr.pretty_print();

        println!("Name: {}", device.name());
    });
}
