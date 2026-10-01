# Porting notes: tubulaton → rusty_tube

Source: `~/projects/pavement/external/tubulaton`, branch
`improve-correctness-and-english` (commit `c7da0c3`, 2026-09-22).

## What maps where

| C++ | Rust |
|---|---|
| `Parameters` | `src/params.rs`: every predeclared key and default, English aliases, numbered per-surface keys, `Update` blocks, per-second rate calibration |
| `Random` | `src/rng.rs` (xoshiro256++; `seed=` pins it, the seed in use is printed) |
| `Contour` | `src/sim/membrane.rs`: multi-surface loading, area normalisation of nucleation, per-area nucleation CDF, mesh-resolution guard, `AdaptElement` |
| `ElementPool` | `src/sim/mod.rs` (`ElementPool`, neighbour searches) and `src/sim/grid.rs` |
| `Microtubule` | `Sim::grow_plus`, `grow_contact_neighbour`, `grow_contact_mb`, `grow_free`, `shrink_plus`, `shrink_moins`, `sever` |
| `MicrotubulePool` | `Sim::evolve`, `initiate`, `initiate_midway`, `choose_new_mt`, the R3/R4 stress-biased rates |
| `main.cpp` loop, `rapport` | `Sim::tick`, `Sim::write_report` |
| `drawScene::pool2vtk` | `Sim::write_vtk`: same binary legacy-VTK layout, so `pavesim.tubulaton.vtk` reads it |
| VTK reader | `src/vtk.rs` (ASCII and binary, file versions up to 5.1) |

## Validation

The C++ (`build-cat/programme`) and the port were run on the same mesh and
config (`configs/pavement_calibrated.ini`, a procedural pavement cell),
24 seeds each, 1000 steps:

| | C++ | Rust |
|---|---|---|
| microtubules | 371.4 ± 3.1 | 374.8 ± 3.0 |
| subunits | 85 078 ± 1 007 | 85 337 ± 1 143 |
| bundling encounters | 242.9 ± 6.5 | 238.2 ± 4.7 |
| encounter catastrophes | 407.4 ± 3.6 | 410.6 ± 4.2 |
| membrane catastrophes | 568.3 ± 3.7 | 565.3 ± 4.5 |
| microtubules destroyed | 889.9 ± 2.9 | 893.2 ± 3.6 |
| detachments | 230.0 ± 2.1 | 229.3 ± 2.5 |

(mean ± standard error.) Cortical fraction 0.995 and escapes ~0.6 % in both.
The port runs about 35× faster (1.1 s against 37 s for 3000 steps).
`tests/regression.rs` keeps a coarse version of this check.

The random streams differ (mt19937 vs xoshiro), so runs are equal in
distribution, not bit for bit.

## Deliberate departures

All remove undefined behaviour or a crash path; none changes a
well-defined result.

* `angleVec` clamps its `acos` argument. (Tested: on the validation runs
  the unclamped version never produced a NaN, so results were unaffected.)
* A NaN growth direction (zero-length normalisation) is replaced by a
  random direction tangent to the membrane wherever it arises; the C++
  only did this in one branch and otherwise let the NaN into the element.
* The stress-steering step skips a field with no component in the membrane
  plane instead of normalising a zero vector.
* Dead elements are removed from the spatial grid at once, rather than at the
  next garbage collection. They were already skipped by every search.
* Crossover references carry a generation, so a reference to a freed element
  reads as dead instead of dangling.
* The microtubule–microtubule neighbour grid is sized to the bundling radius
  `D_bundle/taille_microtubule`, not to the contour grid. The C++ README
  lists this as a limitation; results are identical because the search only
  ever uses neighbours within that radius.
* Grid keys cannot alias: the C++ packs cell indices into bit fields sized
  from the bounds, so a filament leaving the box can land in another cell.
* The `stop` parameter is not reset by the membrane-angle rule (a leftover
  debugging hook in `grow_plus_contact_mb`).

## Not ported yet

* Moving and deforming contours: `Movement_Speed`, `UpdateNextTime`,
  `Update_file`, `MoveValues`, `MoveNucleus`, `UpdateUnconnected` (and so an
  `Update` block that changes `nom_input_vtk`). `Sim::check_supported`
  refuses these configs instead of ignoring them.
* `drawScene::Contour2vtk` (the per-vertex force output, `Output_Contour`).
* `Element::getProperty` for arbitrary mesh arrays: only `strain_0`,
  `attachement` and `Nucleation` are read, which are the only ones the
  dynamics use.

## Finding along the way

In real, strongly lobed pavement cells the calibrated parameter set produces
a sparse array (mean length 1.3 µm, ~45 000 subunits on a 1 280 µm² cell).
The `Angle_mb_limite` rule (catastrophe when the membrane demands a turn of
more than 40°) causes 78 % of all catastrophes there, against 43 % on a
smooth procedural cell. Raising the limit to 57° alone removes membrane
catastrophes entirely (mean length 2.1 µm, 2.8× the polymer); adding
encounter catastrophe at 0.4 instead of 1 (Dixit & Cyr 2004) gives 2.6 µm. The result is
therefore very sensitive to this one parameter in lobed geometry. The
`dense` preset uses that setting for illustration; it is not a calibration.
