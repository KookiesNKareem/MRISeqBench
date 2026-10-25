# Static cardiac phantom

All sequence tiers use `cardiac_10xz_shortaxis_clean.phantom` anatomy: fat,
myocardium and blood, with no motion. 2D uses its central short-axis slice;
3D uses the complete volume. Coordinates are recentered without cropping or
rescaling; nominal source spacing replaces floating-point coordinate roundoff.
The supplied input manifest pins the exported anatomy and fields by SHA-256.

Each scanner strength selects an explicit cardiac tissue profile. Values below
are T1 / T2 in milliseconds; relaxation is not linearly scaled or interpolated.

| Field | Myocardium | Blood | Fat |
|---|---:|---:|---:|
| 0.55 T | 701 / 58 | 1122 / 263 | 187 / 93 |
| 1.5 T | 955 / 49 | 1441 / 290 | 343 / 58 |
| 3 T | 1122 / 52 | 1932 / 275 | 382 / 68 |

Sources: 0.55 T [Campbell-Washburn 2019, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6823617/);
myocardium [Varghese 2020](https://doi.org/10.1002/nbm.4317) and [Roy 2017](https://doi.org/10.1186/s12968-017-0371-5);
blood [Stanisz 2005, Table 1](https://doi.org/10.1002/mrm.20605);
fat [de Bazelaire 2004, Tables 1–2](https://doi.org/10.1148/radiol.2303021331).

These are canonical simulation parameters combining different studies. Higher-field
blood is an in vitro reference and fat uses a subcutaneous-fat proxy. Both cardiac
chambers share one blood compartment. The 3 T profile and common proton density
(approximately 0.1) retain the supplied phantom's stored values.

The source has no B0 map. A synthetic static field adds 1000 Hz/m along x and,
for 3D, 300 Hz/m along z at 3 T; amplitudes scale by scanner field / 3 T.
Fat uses the source's single peak: −434.2902819 Hz at 3 T, also scaled by field.
The central myocardial B0 SD is 33 Hz at 3 T.
Chemical shift and scanner B0 are added separately, once. The richer fat spectrum
is outside this approximation; see [Hamilton 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3860876/).

Object references use `055t`, `15t` or `3t`; resolution selects the matching
scanner profile and rejects mismatches. Public briefs expose actual T1/T2,
proton density and chemical shift. Reconstruction remains standard and fixed.
Independent full-grid targets hold finite-RF tissue response at the central echo,
including intrinsic fat. The phase oracle adds only declared B0; readout-induced
fat displacement and blur remain errors under the fixed reconstruction.
S4 additionally tests ECG windows, preparation efficiency, bSSFP and mapping.
