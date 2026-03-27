{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.pkg-config
    pkgs.zlib
    pkgs.libjpeg
  ];
}
