# Round 6 smoke pass/fail (540×960, 8 samples, `pilot-demic-r6`)

| Frame | Arms attached | Hands on bar | Bar overhead (stretch) | Cable → pulley | Glow discrete / DEMIC-like | No bleed | No floating parts | Machine readable |
|---|---|---|---|---|---|---|---|---|
| **wrong · front · stretch** | FAIL (shoulder gap / limbs read separate) | PARTIAL | **PASS** meta (wrist_z 1.40 > head 1.14) | FAIL (cable behind stack, not centred on bar) | **PASS** (small arm patches, ~1.6% mean glow) | **PASS** (no bodysuit) | FAIL | PARTIAL (stack behind, blocky) |
| **wrong · side · contract** | PARTIAL | PASS | N/A | PARTIAL | **PASS** (bi/delt/trap stripes) | PASS | PARTIAL | PASS (side silhouette + stack) |
| **right · front · stretch** | FAIL (crossed forearms, shoulder gap) | PARTIAL | **PASS** meta (1.42 > 1.18) | FAIL | **PASS** (minimal front glow) | PASS | FAIL | PARTIAL |
| **right · side · contract** | PARTIAL | PASS | N/A | PARTIAL | **PASS** (green lat patch on back) | PASS | PARTIAL (block in front of hands) | **PASS** (seated pull read, stack behind, no face plates) |

### vs Round 5 smoke

- Glow bodysuit → **fixed** (discrete masks).
- Plates floating at face → **fixed** (handle-only bar).
- Stack layout → **improved** (behind seat).
- Arms crossed / shoulder gap → **still open** (needs Mixamo/Rigify or MPFB rig + mocap, not more bpy IK).
