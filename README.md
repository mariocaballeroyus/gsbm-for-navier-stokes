# A Generalized Shifted Boundary Method with Divergence-Conforming B-Splines for the Navier-Stokes Equations on Moving Domains

Benchmarks of the GSBM [1] in the IgaApplication of Kratos Multiphysics.

## Installation

Kratos, branch `iga/generalized-sbm`, with the IgaApplication and LinearSolversApplication:

```bash
git clone -b iga/generalized-sbm https://github.com/KratosMultiphysics/Kratos.git && cd Kratos
export KRATOS_APPLICATIONS="$PWD/applications/IgaApplication;$PWD/applications/LinearSolversApplication;"
cmake -S "$PWD" -B "$PWD/build/Release" -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE="$(which python3)"
cmake --build build/Release --target install -j"$(nproc)"
export PYTHONPATH="$PWD/bin/Release" LD_LIBRARY_PATH="$PWD/bin/Release/libs"
```

## References

[1] O. Colomés, J. Modderman, G. Scovazzi. The generalized shifted boundary method for geometry-parametric PDEs and time-dependent domains. *Comput. Methods Appl. Mech. Engrg.* 452 (2026) 118748.
