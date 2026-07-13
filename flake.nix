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
        ];

        # Ensure cargo-expand can find the nightly rustfmt it needs, or
        # fall back gracefully. The default rustfmt from stable is fine
        # for most use; cargo-expand will still work.
        RUSTC_VERSION = pkgs.rustc.version;
      };
    };
}