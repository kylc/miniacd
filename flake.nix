{
  inputs = {
    fenix.url = "github:nix-community/fenix/monthly";
    naersk.url = "github:nix-community/naersk/master";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, fenix, naersk, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        naersk-lib = pkgs.callPackage naersk { };
        fenix-pkgs = fenix.packages.${system};
      in {
        packages.default = naersk-lib.buildPackage ./.;
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [
              # Rust toolchain
              fenix-pkgs.latest.toolchain
              fenix-pkgs.rust-analyzer
              maturin
              mold

              # Development utilities
              cargo-flamegraph
              cargo-nextest
              cargo-outdated
              f3d
              pre-commit
              watchexec

              # Python packages
              python3Packages.click
              python3Packages.numpy
              python3Packages.pip
              python3Packages.trimesh
              python3Packages.venvShellHook
            ];
            RUST_SRC_PATH =
              "${fenix-pkgs.latest.rust-src}/lib/rustlib/src/rust/library";
            PYO3_PYTHON = "${python3}/bin/python3";
            venvDir = "./.venv";
          };
      });
}
