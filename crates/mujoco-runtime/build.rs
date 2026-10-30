fn main() {
    let library = std::path::PathBuf::from(
        std::env::var_os("MUJOCO_LIB").expect("enter nix develop: MUJOCO_LIB is required"),
    );
    let include = library
        .parent()
        .expect("MuJoCo lib directory has no parent")
        .join("include");
    cc::Build::new()
        .include(include)
        .file("src/model_access.c")
        .compile("robot_sim_model_access");
    println!("cargo:rerun-if-env-changed=MUJOCO_LIB");
    println!("cargo:rerun-if-changed=src/model_access.c");
}
