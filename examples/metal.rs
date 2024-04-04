//! Using RLST together with Apple Metal acceleration

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rlst::external::metal::{AutoReleasePool, MetalDevice};

pub fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    AutoReleasePool::execute(|| {
        let device = MetalDevice::from_default();
        println!("Name: {}", device.name());
    });
}
