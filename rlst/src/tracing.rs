//! Simple structures to trace execution times of functions

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
    time::Duration,
};

/// Output only on root rank 0
#[macro_export]
macro_rules! println_mpi {
    ($rank:expr, $($arg:tt)*) => {
        if $rank == 0 {
            println!($($arg)*);
        }
    };
}

/// A simple global tracing structure.
pub struct Tracing;

impl Tracing {
    fn get_map() -> &'static LazyLock<Mutex<HashMap<String, Duration>>> {
        static MAP: LazyLock<Mutex<HashMap<String, Duration>>> =
            LazyLock::new(|| Mutex::new(HashMap::new()));
        &MAP
    }

    /// Add a duration
    pub fn add_duration(identifier: &str, duration: Duration) {
        *Tracing::get_map()
            .lock()
            .unwrap()
            .entry(identifier.into())
            .or_insert(Duration::ZERO) += duration;
    }

    /// Clear the data
    pub fn clear() {
        Tracing::get_map().lock().unwrap().clear();
    }

    /// Copy values out as HashMap
    pub fn durations() -> HashMap<String, Duration> {
        Tracing::get_map().lock().unwrap().clone()
    }
}

#[cfg(feature = "enable_tracing")]
/// Trace execution time of a block of code
pub fn trace_call<T>(identifier: &str, mut fun: impl FnMut() -> T) -> T {
    let now = std::time::Instant::now();
    let res = fun();
    let duration = now.elapsed();
    let duration_in_secs = duration.as_secs_f64();
    log::info!("Id: {identifier} - {duration_in_secs}s");
    Tracing::add_duration(identifier, duration);
    res
}
#[cfg(not(feature = "enable_tracing"))]
/// Trace execution time of a block of code
pub fn trace_call<T>(_identifier: &str, mut fun: impl FnMut() -> T) -> T {
    fun()
}
