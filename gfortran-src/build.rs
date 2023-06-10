fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/opt/homebrew/opt/gfortran/lib/gcc/current");
    }
    println!("cargo:rustc-link-lib=dylib=gfortran");
}
