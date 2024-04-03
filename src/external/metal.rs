pub mod batched_gemm;
pub mod interface;
pub mod metal_array;
pub mod raw;

pub use interface::AutoReleasePool;
pub use interface::MetalDevice;
pub use metal_array::MetalDataContainer;
