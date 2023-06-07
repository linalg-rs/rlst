use cmake::Config;

macro_rules! build_dep {
    ($name:literal, $root:expr) => {
        Config::new(format!("suitesparse/{}", $name))
            .define("BLA_VENDOR", "FLAME")
            .define("CMAKE_PREFIX_PATH", $root)
            .build()
    };
}

fn main() {
    // Build Suitesparse config

    let blis_root = std::env::var("DEP_BLIS_ROOT").unwrap();

    let suitesparse = build_dep!("Suitesparse_config", &blis_root);
    let _amd = build_dep!("AMD", &blis_root);
    let _camd = build_dep!("CAMD", &blis_root);
    let _colamd = build_dep!("COLAMD", &blis_root);
    let _ccolamd = build_dep!("CCOLAMD", &blis_root);
    let _cholmod = build_dep!("CHOLMOD", &blis_root);
    let _umfpack = build_dep!("UMFPACK", &blis_root);

    println!(
        "cargo:rustc-link-search={}",
        suitesparse.join("lib").display()
    );
    println!("cargo:rustc-link-lib=suitesparseconfig");
    println!("cargo:rustc-link-lib=amd");
    println!("cargo:rustc-link-lib=camd");
    println!("cargo:rustc-link-lib=colamd");
    println!("cargo:rustc-link-lib=ccolamd");
    println!("cargo:rustc-link-lib=cholmod");
    println!("cargo:rustc-link-lib=umfpack");
}
