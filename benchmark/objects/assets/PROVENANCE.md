The cardiac anatomy comes from the user-supplied file
`cardiac_10xz_shortaxis_clean.phantom`.
Its exact original bytes are pinned as `cardiac_source.phantom`; `../anatomy.json`
records its SHA-256 and the hashes of the derived label assets. Evaluation has no
external filesystem dependency. This is supplied project data; no third-party
license has been inferred or substituted.

The source identifies itself as “Cardiac 3 T case 01, static 10x-z, fat shift,
no B0 map.” It contains 3,466,834 occupied grid cells, with a regular
90 × 89 × 1090 lattice and cell sizes 1.5 × 1.5 × 0.15 mm. There are three
source contrast tuples: fat (T1 0.382 s, T2 0.068 s), myocardium (T1 1.122 s,
T2 0.052 s), and blood (T1 1.932 s, T2 0.275 s). Their supplied equilibrium
density 0.1 is retained. T2* is 10^6 s in the source and therefore supplies no
additional susceptibility decay. Fat has intrinsic chemical shift
−2728.726318359375 rad/s = −434.2902819118434 Hz at 3 T; it scales linearly
with actual scanner field strength. These tissue names identify the three
contrast classes in the supplied cardiac anatomy.

The 3D object retains every occupied source cell, without cropping, resizing or
shape deformation. Axis midpoints are subtracted, translating the source's z
centre from approximately +20 mm to zero. Rounded Float32 coordinates are
replaced with the declared uniform lattice spacing, removing only coordinate
roundoff. The physical grid extends ±67.5 mm, ±66.75 mm and ±81.75 mm and fits
the existing 200 mm field of view. The 2D object is native z plane 545
(zero-based), immediately above the volume centre, with the same x/y cell
geometry and material properties. It is the short-axis view of the same anatomy.

The source has no motion or spatial B0 map. The benchmark adds a separate,
static spatial B0 profile; this profile and intrinsic fat chemical shift remain
separate, and both scale with scanner field. No motion is synthesized.

`../generate_cardiac.py` regenerates the label assets and provenance manifest
from the pinned source. The derived HDF5 label arrays use 0 for empty cells and
1/2/3 for fat/myocardium/blood. Python and Julia use identical cell centres and
material properties. The judge treats each occupied cell as a piecewise-constant
continuum: exact cell Fourier integrals provide the independent target, and
independent scrambled Sobol samples choose occupied cells within each tissue and
uniform positions inside them for encoded Bloch integration. Each tissue volume
is exact in every replicate. Image masks and voxel-centre phase fractions use
exact overlaps between anatomical cells and image voxels. Standard reconstruction
and quality thresholds are unchanged. Historical analytic toy-phantom
calibration does not establish acceptance on this cardiac anatomy.

The explicit `055t`, `15t` and `3t` object references share these same geometry
bytes. They select discrete field-specific rho/T1/T2 profiles in `../anatomy.json`;
T1 and T2 are not obtained by linear field scaling. The fourth property column
always stores fat chemical shift in Hz at 3 T, and the simulator applies the
actual scanner field factor once. See the sourced table and its measurement
limitations in [CARDIAC_PHANTOM.md](../../../docs/CARDIAC_PHANTOM.md) and the
pinned `../tissue_relaxation_sources.json`.
