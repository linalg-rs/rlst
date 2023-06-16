use cmake::Config;

macro_rules! build_dep {
    ($name:literal, $blas_lib:expr, $lapack_lib:expr) => {{
        let out_dir = std::env::var("OUT_DIR").unwrap();
        Config::new(format!("suitesparse/{}", $name))
            .define("CMAKE_PREFIX_PATH", out_dir)
            .define("LAPACK_LIBRARIES", $lapack_lib)
            .define("BLAS_LIBRARIES", $blas_lib)
            .define("BLA_STATIC", "TRUE")
            .build()
    }};
}

fn main() {
    // These are only needed since the build scripts for Suitesparse insist
    // on also building shared library versions that require resolution of
    // the Blas/Lapack symbol names.
    let blas_lib = std::env::var("DEP_BLIS_ROOT").unwrap() + "/lib/libblis.a";
    let lapack_lib = std::env::var("DEP_NETLIBLAPACK_ROOT").unwrap() + "/lib/liblapack.a";

    let suitesparse = build_dep!("SuiteSparse_config", &blas_lib, &lapack_lib);
    let _amd = build_dep!("AMD", &blas_lib, &lapack_lib);
    let _camd = build_dep!("CAMD", &blas_lib, &lapack_lib);
    let _colamd = build_dep!("COLAMD", &blas_lib, &lapack_lib);
    let _ccolamd = build_dep!("CCOLAMD", &blas_lib, &lapack_lib);
    let _cholmod = build_dep!("CHOLMOD", &blas_lib, &lapack_lib);
    let _umfpack = build_dep!("UMFPACK", &blas_lib, &lapack_lib);

    println!("cargo:root={}", std::env::var("OUT_DIR").unwrap());

    println!(
        "cargo:rustc-link-search={}",
        suitesparse.join("lib").display()
    );

    println!("cargo:rustc-link-lib=static=suitesparseconfig");
    println!("cargo:rustc-link-lib=static=amd");
    println!("cargo:rustc-link-lib=static=camd");
    println!("cargo:rustc-link-lib=static=colamd");
    println!("cargo:rustc-link-lib=static=ccolamd");
    println!("cargo:rustc-link-lib=static=cholmod");
    println!("cargo:rustc-link-lib=static=umfpack");

    // On Linux OpenMP is automatically enabled. Need to link against
    // gomp library.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}
