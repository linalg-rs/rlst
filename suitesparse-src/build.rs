use cmake::Config;

macro_rules! build_dep {
    ($name:literal) => {{
        let out_dir = std::env::var("OUT_DIR").unwrap();
        Config::new(format!("suitesparse/{}", $name))
            .define("CMAKE_PREFIX_PATH", out_dir)
            .build()
    }};
}

fn main() {
    // These are only needed since the build scripts for Suitesparse insist
    // on also building shared library versions that require resolution of
    // the Blas/Lapack symbol names.

    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:warning=Out Dir {}", out_dir);

    let suitesparse = build_dep!("SuiteSparse_config");
    let _amd = build_dep!("AMD");
    let _camd = build_dep!("CAMD");
    let _colamd = build_dep!("COLAMD");
    let _ccolamd = build_dep!("CCOLAMD");
    let _cholmod = build_dep!("CHOLMOD");
    let _umfpack = build_dep!("UMFPACK");

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
