{
  inputs = {
    nixpkgs.url="github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url="github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {nixpkgs, flake-utils, poetry2nix, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv defaultPoetryOverrides;

    in rec {
      devShell = pkgs.mkShell {
	buildInputs = [
    ( mkPoetryEnv {
        projectDir = ./.;
        preferWheels = true;
        overrides = defaultPoetryOverrides;
        })
      pkgs.poetry
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glib
          pkgs.copier
        ];
        # set jupyter notebook password
        shellHook = ''export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.glib}/lib:$LD_LIBRARY_PATH
        potery --version'';
        
      };
    }
  );
}
