{
  description = "LLM Orch — single-host LLM orchestrator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      # ROCm overlays: rocmPackages_10_0 (TheRock prebuilt binaries).
      # See nix/rocm.nix. Apply via nixpkgs.overlays or use directly:
      #   (self.overlays.rocm null nixpkgs.legacyPackages.x86_64-linux)
      overlays.rocm = import ./nix/rocm.nix;
      overlays.default = self.overlays.rocm;

      packages.${system} = {
        default = pkgs.rustPlatform.buildRustPackage {
        pname = "llm-orch";
        version = "0.1.0";

        src = ./.;

        cargoLock.lockFile = ./Cargo.lock;

        meta = with pkgs.lib; {
          description = "Single-host LLM orchestrator with hot-reload, API key auth, and llama.cpp backend management";
          license = licenses.mit;
          mainProgram = "llm-orch";
          platforms = platforms.linux;
        };
      };

        # Compile+run a HIP kernel against the TheRock binary ROCm to prove
        # the toolchain works on the current machine (needs a gfx1201 GPU):
        #   nix run .#hip-smoke
        hip-smoke =
          (import nixpkgs {
            inherit system;
            # TheRock is AMD's redistributable binary dist; consumers apply
            # overlays.rocm to their own (unfree-enabled) package set.
            config = { allowUnfreePredicate = p: pkgs.lib.elem (pkgs.lib.getName p) [ "rocm-core-bin-gfx120x" ]; };
            overlays = [ self.overlays.rocm ];
          }).rocmPackages_10_0.hip-smoke;
      };

      nixosModules.llm-orch = { pkgs, lib, ... }: {
        imports = [ ./nix/nixos.nix ];
        services.llm-orch.package =
          lib.mkDefault self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      };
      nixosModules.default = self.nixosModules.llm-orch;

      devShells.${system}.default = pkgs.mkShell {
        name = "llm-orch-devshell";

        buildInputs = with pkgs; [
          cargo
          rustc
          clippy
          rustfmt
          rust-analyzer
          cargo-expand
          jq
          # scripts/benchmark-llm.py
          python3
          python3Packages.httpx
        ];

        # Ensure cargo-expand can find the nightly rustfmt it needs, or
        # fall back gracefully. The default rustfmt from stable is fine
        # for most use; cargo-expand will still work.
        RUSTC_VERSION = pkgs.rustc.version;
      };
    };
}