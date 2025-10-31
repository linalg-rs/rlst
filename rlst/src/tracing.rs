//! Simple structures to trace execution times of functions.
//!
//! This module defines a simple tracing system for RLST. Tracing information is stored in a global hashmap that maps
//! a [String] to a [Duration]. To access the map from any thread call [Tracing::get_map]. The returned map is a global
//! that is lazily instantiated on the first call.
//!
//! Moreover, a function [trace_call] is provided that can be used as a wrapper around other calls to measure duration and
//! store the duration in the global map. An example is shown below.
//!
//! ```
//! use rlst::tracing::trace_call;
//!
//! pub fn adder(x: usize) -> usize  {
//!   1 + x
//! }
//!
//! let a = trace_call("Add 1", || {adder(2) });
//! assert_eq!(a, 3);
//! ```
//! Alternatively, one can annotate a function directly with the macro [measure_duration](crate::measure_duration).
//! ```
//! use rlst::measure_duration;
//!
//! #[measure_duration(id ="Add 1")]
//! pub fn adder(x: usize) -> usize {
//!   1 + x
//! }
//!
//! let a = adder(1);
//! ```
//! In both cases the result is stored in the global map under the string id "Add 1".
//! Use of tracing requires the feature flag `enable_tracing` to be set. Without this
//! feature flag the above examples still execute but do not store tracing information.
//!
//! Both, the function `trace_call` and the macro `measure_duration` also emit a logging message
//! with the duration printed in seconds.
//!
//! Repeated tracing calls with the same identifier add up the corresponding durations, so that the
//! stored duration is always the sum of the individual durations of the same identifier.
//!
//! The module also provides a simple helper [println_mpi](crate::println_mpi). This is useful for MPI environments.
//! It acts exactly like [println] but only executes on rank 0.
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
    time::Duration,
};

/// A println function that is MPI aware and outputs only on rank 0.
#[macro_export]
macro_rules! println_mpi {
    ($rank:expr, $($arg:tt)*) => {
        if $rank == 0 {
            println!($($arg)*);
        }
    };
}

/// A simple global tracing structure. This struct is not meant to be instantiated directly.
/// It only provides a namespace for functions that operate on the global durations map.
///
/// The following functions are provided.
/// - [Tracing::get_map] (Get the HashMap that stores the durations for all ids)
/// - [Tracing::add_duration] (Add a duration to the HashMap)
/// - [Tracing::clear] (Clear all durations from the map)
/// - [Tracing::durations] (Clone the map of all existing durations)
pub struct Tracing;

impl Tracing {
    /// Get the HashMap that stores the durations for all ids.
    pub fn get_map() -> &'static LazyLock<Mutex<HashMap<String, Duration>>> {
        static MAP: LazyLock<Mutex<HashMap<String, Duration>>> =
            LazyLock::new(|| Mutex::new(HashMap::new()));
        &MAP
    }

    /// Add a duration to the HashMap.
    pub fn add_duration(identifier: &str, duration: Duration) {
        *Tracing::get_map()
            .lock()
            .unwrap()
            .entry(identifier.into())
            .or_insert(Duration::ZERO) += duration;
    }

    /// Clear all durations from the map.
    pub fn clear() {
        Tracing::get_map().lock().unwrap().clear();
    }

    /// Clone the map of all existing durations.
    pub fn durations() -> HashMap<String, Duration> {
        Tracing::get_map().lock().unwrap().clone()
    }
}

#[cfg(feature = "enable_tracing")]
/// Trace execution time of a block of code and store under the name `identifier`.
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
/// Trace execution time of a block of code and store under the name `identifier`.
pub fn trace_call<T>(_identifier: &str, mut fun: impl FnMut() -> T) -> T {
    fun()
}
