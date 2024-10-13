use git2::Repository;
use std::env;
use std::path::PathBuf;

use cmake::Config;

macro_rules! build_dep {
    ($name:literal) => {{
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        Config::new(out_dir.join("suitesparse").join($name))
            .define("CMAKE_PREFIX_PATH", out_dir)
            .build()
    }};
}

fn build_sleef() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sleef_dir = out_dir.join("sleef");

    if !sleef_dir.exists() {
        // From https://stackoverflow.com/questions/55141013/how-to-get-the-behaviour-of-git-checkout-in-rust-git2
        let repo = Repository::clone("https://github.com/shibatch/sleef", sleef_dir.clone())
            .expect("Could not clone Sleef repository.");
        let refname = "3.6.1";
        let (object, reference) = repo.revparse_ext(refname).expect("Object not found");
        repo.checkout_tree(&object, None)
            .expect("Failed to checkout");

        match reference {
            // gref is an actual reference like branches or tags
            Some(gref) => repo.set_head(gref.name().unwrap()),
            // this is a commit, not a reference
            None => repo.set_head_detached(object.id()),
        }
        .expect("Failed to set HEAD");
    }

    Config::new(sleef_dir.clone())
        .define("CMAKE_PREFIX_PATH", out_dir.clone())
        .profile("Release")
        .build();

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_arch == "aarch64" {
        cc::Build::new()
            .file("sleef_interface/sleef_neon.c")
            .include(out_dir.join("include"))
            .flag("-march=native")
            .compile("sleef_interface");
    } else if target_arch == "x86_64" {
        cc::Build::new()
            .file("sleef_interface/sleef_avx.c")
            .include(out_dir.join("include"))
            .flag("-march=native")
            .compile("sleef_interface");
    }

    println!("cargo:rustc-link-lib=static=sleef");
    println!("cargo:rustc-link-lib=static=sleef_interface");
}

fn build_umfpack(out_dir: String) {
    let out_path = PathBuf::from(out_dir.clone());

    let suitesparse_dir = out_path.join("suitesparse");

    if !suitesparse_dir.exists() {
        // From https://stackoverflow.com/questions/55141013/how-to-get-the-behaviour-of-git-checkout-in-rust-git2
        let repo = Repository::clone(
            "https://github.com/DrTimothyAldenDavis/SuiteSparse.git",
            suitesparse_dir.clone(),
        )
        .expect("Could not clone Suitesparse repository.");
        let refname = "v7.7.0";
        let (object, reference) = repo.revparse_ext(refname).expect("Object not found");
        repo.checkout_tree(&object, None)
            .expect("Failed to checkout");

        match reference {
            // gref is an actual reference like branches or tags
            Some(gref) => repo.set_head(gref.name().unwrap()),
            // this is a commit, not a reference
            None => repo.set_head_detached(object.id()),
        }
        .expect("Failed to set HEAD");
    }

    let _suitesparse_config = build_dep!("SuiteSparse_config");
    let _amd = build_dep!("AMD");
    let _camd = build_dep!("CAMD");
    let _colamd = build_dep!("COLAMD");
    let _ccolamd = build_dep!("CCOLAMD");
    let _cholmod = build_dep!("CHOLMOD");
    let _umfpack = build_dep!("UMFPACK");

    // Only needed if we want to let other libraries know where we
    // compiled suitesparse.
    // println!("cargo:root={}", std::env::var("OUT_DIR").unwrap());

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
        println!("cargo:rustc-link-lib=dylib=lapack");
        println!("cargo:rustc-link-lib=dylib=blas");
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
    println!("cargo:rustc-link-search={}", out_dir.clone() + "/lib");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let mut use_system_blas_lapack = std::env::var("CARGO_FEATURE_DISABLE_SYSTEM_BLAS_LAPACK")
        .is_err()
        && std::env::var("CARGO_FEATURE_INTERNAL_BLIS").is_err();

    if target_os == "macos" && !use_system_blas_lapack {
        println!("cargo:warning=Reverting to Accelerate as BLAS/Lapack provider on Mac OS.");
        use_system_blas_lapack = true
    }

    if use_system_blas_lapack {
        if target_os == "macos" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }

        if target_os == "linux" {
            println!("cargo:rustc-link-lib=blas");
            println!("cargo:rustc-link-lib=lapack");
        }
    }

    // if std::env::var("CARGO_FEATURE_INTERNAL_BLIS").is_ok() && target_os != "macos" {
    //     build_internal_blis();
    // }

    if std::env::var("CARGO_FEATURE_SUITESPARSE").is_ok() {
        build_umfpack(out_dir.clone());
    }

    if std::env::var("CARGO_FEATURE_SLEEF").is_ok() {
        build_sleef()
    }

    // println!("// cargo:rern-if-changed=sleef_interface/sleef_avx.c");
    // println!("cargo:rerun-if-changed=sleef_interface/sleef_neon.c");
    println!("cargo:rerun-if-changed=build.rs");
}
