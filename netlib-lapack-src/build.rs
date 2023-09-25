fn main() {
    // // Build Lapack static

    let blis_root = std::path::PathBuf::from(std::env::var("DEP_BLIS_ROOT").unwrap());

    let dst = cmake::Config::new("lapack")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LAPACKE", "ON")
        .define("BLA_VENDOR", "FLAME")
        .define("CMAKE_PREFIX_PATH", blis_root.display().to_string())
        .define("USE_OPTIMIZED_BLAS", "ON")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("TEST_FORTRAN_COMPILER", "ON")
        .build();

    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=static=lapacke");
}
