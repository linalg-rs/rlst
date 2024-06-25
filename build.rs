use std::env;
use std::path::PathBuf;

use cmake::Config;

macro_rules! build_dep {
    ($name:literal) => {{
        let out_dir = std::env::var("OUT_DIR").unwrap();
        Config::new(format!("suitesparse/{}", $name))
            .define("CMAKE_PREFIX_PATH", out_dir)
            .build()
    }};
}

fn build_lapack() {
    use glob::glob;

    let mut build = cc::Build::new();

    for path in glob("lapack/*.c").unwrap() {
        match path {
            Ok(path) => {
                build.file(path);
            }
            Err(e) => {
                println!("cargo:warning={:#?}", e);
            }
        }
    }

    build.compile("lapack");
    println!("cargo:rustc-link-lib=static=lapack");

    // cc::Build::new().files(glob("lapack/*.c")).compile("lapack");
}

fn build_metal(out_dir: String) {
    cc::Build::new()
        .file("metal/rlst_metal.m")
        .compile("rlst_metal");

    let sdk_path = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";

    let bindings = bindgen::Builder::default()
        .header("metal/rlst_metal.h")
        .clang_args(["-x", "objective-c"])
        .clang_args(&["-isysroot", sdk_path])
        .allowlist_function("rlst.*")
        .allowlist_type("RLST.*")
        .allowlist_var("RLST.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(out_dir.clone());

    bindings
        .write_to_file(out_path.join("metal.rs"))
        .expect("Could not write bindings.");

    println!("cargo:warning={}", out_dir);
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=rlst_metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
}

fn build_umfpack(out_dir: String) {
    let out_path = PathBuf::from(out_dir.clone());

    let suitesparse = build_dep!("SuiteSparse_config");
    let _amd = build_dep!("AMD");
    let _camd = build_dep!("CAMD");
    let _colamd = build_dep!("COLAMD");
    let _ccolamd = build_dep!("CCOLAMD");
    let _cholmod = build_dep!("CHOLMOD");
    let _umfpack = build_dep!("UMFPACK");

    // Only needed if we want to let other libraries know where we
    // compiled suitesparse.
    // println!("cargo:root={}", std::env::var("OUT_DIR").unwrap());

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

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/external/umfpack/wrapper.h")
        // Add an include path
        .clang_arg(format!("-I{}", out_path.join("include").display()))
        .allowlist_function("umfpack.*")
        .allowlist_type("UMFPACK.*")
        .allowlist_var("UMFPACK.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:warning={}", out_dir);
    //println!("Out path: {}", out_path.to_str().unwrap());
    bindings
        .write_to_file(out_path.join("umfpack.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let use_metal = target_os == "macos" && target_arch == "aarch64";

    if use_metal {
        build_metal(out_dir.clone());
    }

    if std::env::var("CARGO_FEATURE_SUITESPARSE").is_ok() {
        build_umfpack(out_dir.clone());
    }

    if std::env::var("CARGO_FEATURE_LAPACK").is_ok() {
        build_lapack();
    }

    println!("cargo:rerun-if-changed=metal/rlst_metal.m");
    println!("cargo:rerun-if-changed=build.rs");
}
