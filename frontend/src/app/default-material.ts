/**
 * default-material.ts — the program the viewer boots with.
 *
 * The black slate flooring material shown in the paper's teaser row: a bright
 * sand-mortar base under dark slate tiles (Bricks coverage + fBm slate tones
 * and relief), with all noise seeds baked in so the viewer reproduces the same
 * surface at any resolution. The format is the shared material string
 * (material-io.ts), parseable by both this app and the Python engine.
 */
export const DEFAULT_MATERIAL = `View(0, 0, 4, 4)
Define(tiles, Bricks(brick_width=1.6, brick_height=0.8, mortar=0.06, feather=0.004))
Define(grout_sand, fBm(octaves=4, base_freq=35, to_01=True, seed=1736150699))
Define(slate_macro, fBm(octaves=3, base_freq=1.2, to_01=True, seed=72383839))
Define(slate_grain, fBm(octaves=5, base_freq=8, to_01=True, seed=92547767))
Define(slate_detail, fBm(octaves=4, base_freq=12, to_01=True, seed=1587625347))
Material(
  Layer(1)
    .basecolor((215 + (grout_sand * 20)), (215 + (grout_sand * 20)), (210 + (grout_sand * 15)))
    .roughness((0.9 + (grout_sand * 0.1)))
    .height((grout_sand * 0.004)),
  Layer(tiles)
    .basecolor(((22 + (slate_macro * 25)) + (slate_grain * 10)), ((24 + (slate_macro * 25)) + (slate_grain * 10)), ((24 + (slate_macro * 25)) + (slate_grain * 12)))
    .roughness(((0.38 + (slate_grain * 0.3)) + (slate_detail * 0.12)))
    .height(((0.035 + (slate_grain * 0.02)) + (slate_macro * 0.015)))
)`;
