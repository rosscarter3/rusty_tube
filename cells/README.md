# Cell outlines

`rusty_tube --outline` reads a closed outline in micrometres from here (or
anywhere): a CSV of `x,y` rows, or a pavesim JSON with an `outline` array of
`{"x_um": .., "y_um": ..}`. The viewer lists every `.csv` and `.json` in
`cells/real/`, `cells/sim/` and `cells/`.

Nothing is committed here because the outlines used so far come from
unpublished data:

* `real/`: segmented WT leaf epidermis, exported from pavesim's leaf data
  (`pavesim.leaf.segmentation.cell_outlines`, 0.30 um/px, interior cells only);
* `sim/`: cell outlines from pavesim tissue-model runs (`cell_to_vtk.py` JSON).

Without any outlines, use `--outline procedural:SEED` for a generated
pavement cell.
