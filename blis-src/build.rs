use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Build BLIS

    let dst = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build = dst.join("build");
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

    println!("cargo:root={}", dst.display());
    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=blis");
}
