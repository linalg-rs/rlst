use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use cmake::Config;

macro_rules! build_dep {
    ($name:literal) => {{
        let out_dir = std::env::var("OUT_DIR").unwrap();
        Config::new(format!("suitesparse/{}", $name))
            .define("CMAKE_PREFIX_PATH", out_dir)
            .build()
    }};
}

fn build_internal_blis() {
    let dst = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build = dst.join("build_blis");
    let _ = fs::create_dir(&build);

    let configure = PathBuf::from("blis")
        .join("configure")
        .canonicalize()
        .unwrap();
    let mut config_command = Command::new("sh");
    config_command
        .args(["-c", "exec \"$0\" \"$@\""])
        .arg(configure);
    config_command.arg(format!("--prefix={}", dst.display()));
    config_command.arg("--enable-threading=pthreads");
    config_command.args(["--enable-cblas", "auto"]);
    let status = match config_command.current_dir(&build).status() {
        Ok(status) => status,
        Err(e) => panic!("Could not execute configure command with error {}", e),
    };
    if !status.success() {
        panic!("Configure command failed with error {}", status);
    }

    let make_flags = env::var_os("CARGO_MAKEFLAGS").unwrap();
    let mut build_command = Command::new("sh");
    build_command
        .args(["-c", "exec \"$0\" \"$@\""])
        .args(["make", "install"]);
    build_command.env("MAKEFLAGS", make_flags);

    let status = match build_command.current_dir(&build).status() {
        Ok(status) => status,
        Err(e) => panic!("Could not execute build command with error {}", e),
    };
    if !status.success() {
        panic!("Build command failed with error {}", status);
    }

    let dst = cmake::Config::new("lapack")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LAPACKE", "OFF")
        .define("BLA_VENDOR", "FLAME")
        .define("CMAKE_PREFIX_PATH", dst.display().to_string())
        .define("USE_OPTIMIZED_BLAS", "ON")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();

    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=dylib=blis");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/opt/homebrew/opt/gfortran/lib/gcc/current");
    }
    println!("cargo:rustc-link-lib=dylib=gfortran");
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

    if use_metal {
        build_metal(out_dir.clone());
    }

    if std::env::var("CARGO_FEATURE_INTERNAL_BLIS").is_ok() && target_os != "macos" {
        build_internal_blis();
    }

    if std::env::var("CARGO_FEATURE_SUITESPARSE").is_ok() {
        build_umfpack(out_dir.clone());
    }

    println!("cargo:rerun-if-changed=metal/rlst_metal.m");
    println!("cargo:rerun-if-changed=build.rs");
}
