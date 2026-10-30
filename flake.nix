{
  description = "Generic Rust/Python robotics development";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    mujoco-pkgs.url = "github:NixOS/nixpkgs/775bdd0467f3de2fb5017830d00fa1f573b2ecda";
    menagerie = { url = "github:google-deepmind/mujoco_menagerie"; flake = false; };
  };
  outputs = { self, nixpkgs, mujoco-pkgs, menagerie }: let
    system = "aarch64-darwin";
    pkgs = import nixpkgs { inherit system; };
    physics = import mujoco-pkgs { inherit system; };
    python = pkgs.python312;
    # Upstream marks Darwin broken. Build only the library, and explicitly
    # select ar/ranlib to fix CMake's Darwin IPO tool detection.
    # Our optional Rust viewer supplies GLFW separately.
    mujoco = physics.mujoco.overrideAttrs (old: {
      propagatedBuildInputs = [];
      cmakeFlags = old.cmakeFlags ++ [
        "-DMUJOCO_BUILD_TESTS=OFF" "-DMUJOCO_BUILD_EXAMPLES=OFF"
        "-DMUJOCO_BUILD_SIMULATE=OFF"
        "-DCMAKE_C_COMPILER_AR=${physics.stdenv.cc.bintools}/bin/ar"
        "-DCMAKE_CXX_COMPILER_AR=${physics.stdenv.cc.bintools}/bin/ar"
        "-DCMAKE_C_COMPILER_RANLIB=${physics.stdenv.cc.bintools}/bin/ranlib"
        "-DCMAKE_CXX_COMPILER_RANLIB=${physics.stdenv.cc.bintools}/bin/ranlib"
      ];
      meta = old.meta // { broken = false; };
    });
  in {
    devShells.${system}.default = assert mujoco.version == "3.10.0"; pkgs.mkShell {
      packages = [ pkgs.rustc pkgs.cargo pkgs.rustfmt pkgs.clippy pkgs.clang pkgs.pkg-config
        python pkgs.uv pkgs.maturin mujoco physics.glfw3 pkgs.ffmpeg ];
      MUJOCO_LIB = "${mujoco}/lib";
      SO101_MODEL_DIR = "${menagerie}/robotstudio_so101";
      MENAGERIE_DIR = "${menagerie}";
      NIX_PYTHON = "${python}/bin/python3.12";
      PYO3_PYTHON = "${python}/bin/python3.12";
      UV_PYTHON_DOWNLOADS = "never";
      UV_VENV_SEED = "1";
      shellHook = ''
        export DYLD_LIBRARY_PATH="${mujoco}/lib:${physics.glfw3}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
        export RUSTFLAGS="''${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,${mujoco}/lib -C link-arg=-Wl,-rpath,${physics.glfw3}/lib"
        if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
      '';
    };
  };
}
