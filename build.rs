use std::fs;

fn main() {
    // Keep test/temp artifacts inside the workspace instead of /tmp to avoid filling system tmpfs.
    let _ = fs::create_dir_all(".tmp");
    println!("cargo:rustc-env=TMPDIR=.tmp");
}
