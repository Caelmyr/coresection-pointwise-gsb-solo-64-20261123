# CoreSection

CoreSection is a local image workbench for photographed, polished rock-core cross-sections. Register repeat exposures against the same measured plane, inspect visible grain, lamination, veins and surface marks, and retain the source alongside georeferenced observations.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19982`. Set `CORESECTION_PORT` for another local port. The named volume holds source photos, records and their evidence. With Python 3.11 and the pinned packages installed, run `./start.sh`.

## Inspect a surface

1. Load the reference sections or import your own. The reference sections are generated practice images. Set physical dimensions and use four registration points for each capture. Compare registered views to check alignment and mark an inspection window or excluded patch.
2. In **Measure**, use **Dark opening inventory** to follow dark areas bounded by observed pixels. Amber traces reach a window edge, an exclusion or unavailable image, so only the visible footprint is measured. Green traces are closed within the observed plane. Use **Dark grain census** for local dark grain size and elongation, **Lamina spacing and angle** to compare approximately orthogonal directions of repeated texture, **Vein width** for marked transects, or **Fine-line network** for a branching trace and enclosed regions.
3. Other tools cover color variation, two-reference vein coverage, spot observations, paired polarized glare and raking contrast, texture, line scans, margin depth profiles, scribe visibility, seam differences, and pale mineral flecks. Save geometry measurements, reuse retained patch references, or compare comparable observations across cores.
4. When two rulered plates share one capture, **Geometry → Plane scale evidence** records taped check distances and the second plate's corners, reads every candidate scale against every taped distance, and recommends the scale with the smallest worst-case residual. Adopting pins one report scale with its residuals annotated on the original photo. A saved trace can be re-measured under another stored scale without moving the line on the photo or altering the saved reading; the new record cross-checks which segments are visible in each plane.
5. Pin a report and optionally include original photos in the downloadable ZIP. The local **Received evidence** page verifies received members and lists the sources it could verify.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A section holds up to ten captures and 24 regions. The registered plane is limited to 1,800 pixels on its longest side. Dark opening intensity and apparent edge area are imaging measurements. They do not establish pore depth, porosity in three dimensions, or mineral identity. The opening map and counts cover the whole selected window. The list and CSV retain the largest 500 qualifying openings when more are detected.

The workbench reuses photo, registration, storage, and established image tools from the local WeaveDesk project. CoreSection adds a rock-core reference series, a dark-opening analysis with support-aware component topology, and corresponding browser, evidence and archive behavior.
