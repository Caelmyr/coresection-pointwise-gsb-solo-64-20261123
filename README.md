# CoreSection

CoreSection is a local image workbench for photographed, polished rock-core cross-sections. Register repeat exposures against the same measured plane, inspect visible grain, lamination, veins and surface marks, and retain the source alongside georeferenced observations.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19982`. Set `CORESECTION_PORT` for another local port. The named volume holds source photos, records and their evidence. With Python 3.11 and the pinned packages installed, run `./start.sh`.

## Inspect a surface

1. Load the reference sections or import your own. The reference sections are generated practice images. Set physical dimensions and use four registration points for each capture. Compare registered views to check alignment and mark an inspection window or excluded patch.
2. In **Measure**, use **Dark opening inventory** to follow dark areas bounded by observed pixels. Amber traces reach a window edge, an exclusion or unavailable image, so only the visible footprint is measured. Green traces are closed within the observed plane. Use **Dark grain census** for local dark grain size and elongation, **Lamina spacing and angle** to compare approximately orthogonal directions of repeated texture, **Vein width** for marked transects, or **Fine-line network** for a branching trace and enclosed regions.
3. Other tools cover color variation, two-reference vein coverage, spot observations, paired polarized glare and raking contrast, texture, line scans, margin depth profiles, scribe visibility, seam differences, and pale mineral flecks. Save geometry measurements, reuse retained patch references, or compare comparable observations across cores.
4. Pin a report and optionally include original photos in the downloadable ZIP. The local **Received evidence** page verifies received members and lists the sources it could verify.

## Reassemble a broken core string

The **Core assembly** desk joins photographed fracture faces back into one archived string.

1. Create an assembly with three segments (A is fixed at the uphole end), each with its measured length and the direction of the downhole arrow. Duplicated legacy numbers are flagged rather than renumbered silently. Each segment may carry one documented, unphotographed gap of known length.
2. Record the two field orderings; both must fit the same shelf. The workbench scores every fracture join on visible lamina continuity (edge tone, bedding angle and edge-profile correlation) and, where legacy depth books exist, on depth-record agreement.
3. Exactly one ordering is archived. The decision record states which evidence won, and the defeated ordering keeps the reason it lost — for example, legacy-depth continuity set aside because the duplicated old number makes it unreliable and the laminae do not continue across the fracture. An operator can override the automatic order, and the override is retained with its own rationale.
4. The adopted string opens as a derived specimen: documented gaps remain transparent bands of the recorded length and are never filled with another segment's imagery. Run the existing **Line scan** across several segments. Stations falling on a gap carry no color value, while every measured station is clickable back to the exact point on the original fracture-face photo it came from, including depth along the string.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A section holds up to ten captures and 24 regions. The registered plane is limited to 1,800 pixels on its longest side. Dark opening intensity and apparent edge area are imaging measurements. They do not establish pore depth, porosity in three dimensions, or mineral identity. The opening map and counts cover the whole selected window. The list and CSV retain the largest 500 qualifying openings when more are detected.

The workbench reuses photo, registration, storage, and established image tools from the local WeaveDesk project. CoreSection adds a rock-core reference series, a dark-opening analysis with support-aware component topology, and corresponding browser, evidence and archive behavior.
