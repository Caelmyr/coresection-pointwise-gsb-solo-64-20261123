from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id
from .photos import store_capture


DEMO_KEY = "coresection.training.v2"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.894, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
}


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple


SERIES = (
    DemoSpecimen(173, "Rill sandstone · R17", "Cross-bedded sandstone", "Dry cut", "Upper terrace", "Rack A",
                ((69, 51, 43), (150, 115, 89), (220, 185, 140))),
    DemoSpecimen(284, "Shale · B03", "Pyritic shale", "Light polish", "Lower bank", "Rack B",
                ((35, 38, 42), (86, 94, 95), (155, 165, 154))),
    DemoSpecimen(391, "Calcite seam · E06", "Veined limestone", "Dry cut", "Upper terrace", "Rack A",
                ((64, 61, 55), (146, 139, 121), (228, 221, 197))),
    DemoSpecimen(475, "Oxidized core · K11", "Ferruginous siltstone", "Light polish", "Old bench", "Rack C",
                ((72, 43, 35), (155, 83, 55), (212, 152, 104))),
)


def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class ProofField:
    diffuse: np.ndarray
    height: np.ndarray
    ink_fraction: np.ndarray
    keyline_mask: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def neutral_card(diffuse, roughness, x, y):
    sample = (x >= 0.07) & (x <= 0.165) & (y >= 0.844) & (y <= 0.961)
    border = (x >= 0.063) & (x <= 0.172) & (y >= 0.835) & (y <= 0.968) & ~sample
    diffuse[sample] = np.array([0.194, 0.172, 0.155])
    diffuse[border] *= 0.70
    roughness[sample] = 0.76


def folio_marks(x, y, rng):
    lower_tag = (x > .70) & (x < .94) & (y > .785) & (y < .943)
    lines = ((np.floor(y * 235).astype(int) % 6) == 0) & lower_tag & (x > .717) & (x < .919)
    scale = (x > .48) & (x < .66) & (y > .886) & (y < .938)
    dark_ticks = (np.floor((x - .48) * 225).astype(int) % 2 == 0) & scale
    folio_number = (x > .763) & (x < .875) & (y > .810) & (y < .818)
    tape_1 = (x > .250) & (x < .276) & (y > .372) & (y < .517)
    tape_2 = (x > .613) & (x < .629) & (y > .475) & (y < .635)
    sheet = np.broadcast_to(np.array([.68, .62, .49], dtype=np.float32), (*x.shape, 3)).copy()
    paper = rng.normal(0, .014, x.shape).astype(np.float32)
    warm = (.027 * np.sin(4*x+3*y) + .042*x) * (1 - y)
    sheet += (paper + warm)[..., None]
    sheet[lower_tag | scale] += [.12, .13, .14]
    sheet[lines | dark_ticks | folio_number] = [.14, .15, .12]
    return sheet, tape_1 | tape_2


def polyline_band(x, y, vertices, half_width):
    """Mask of normalized points within half_width of a polyline."""
    band = np.zeros(x.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:]):
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        ratio = np.clip(((x - x0) * dx + (y - y0) * dy) / length2, 0.0, 1.0)
        nearest_x = x0 + ratio * dx
        nearest_y = y0 + ratio * dy
        band |= np.hypot(x - nearest_x, y - nearest_y) <= half_width
        band &= (x >= min(x0, x1) - half_width) & (x <= max(x0, x1) + half_width)
        band &= (y >= min(y0, y1) - half_width) & (y <= max(y0, y1) + half_width)
    return band


# A hand-drawn grey rule forks: one strike keeps the pencil continuity,
# the competing strike is a relief fissure that only flips under raking light.
PENCIL_TRUNK = ((.205, .335), (.300, .350), (.398, .382))
PENCIL_CONTINUATION = ((.398, .382), (.560, .407), (.745, .438))
PENCIL_FORK_STRIKE = ((.398, .382), (.505, .505), (.648, .596))
FISSURE_TRACE = ((.398, .382), (.505, .505), (.648, .596))


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = (height, width)
    generator = np.random.default_rng(specimen.seed)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 47, 68)
    mottling = random_field(generator, shape, 11, 19)
    grain = generator.normal(size=shape).astype(np.float32)
    sheet, tape = folio_marks(x, y, generator)
    px, py = x * width, y * height
    contour = .006 * np.sin(20 * y) + .004 * mottling
    section = (x > .12 + contour) & (x < .895 + contour) & (y > .205) & (y < .776)
    # Piecewise mineral color follows irregular grain boundaries rather than the optical grid.
    seeds = generator.uniform((64, 86), (574, 380), size=(1700, 2))
    distance, indices = cKDTree(seeds).query(np.stack((px, py), axis=-1), k=2)
    origin_luma = generator.normal(0, .48, len(seeds))
    mineral = origin_luma[indices[..., 0]]
    boundaries = np.exp(-np.maximum(distance[..., 1]-distance[...,0], 0)*1.75)
    bend = 5.6 * np.sin(5 * x + specimen.seed) + 2.2 * broad
    bedding = np.sin(2 * np.pi * (py - 0.155 * px + bend) / 18.5)
    joints = np.sin(2 * np.pi * (px + 0.13 * py + 1.8 * np.sin(12 * y)) / 28.1)
    strata = .52*bedding + .30*joints + .36*broad + .30*mottling + .62*mineral + .11*grain
    rock = surface_palette(specimen.colors, strata)
    rock *= (1 - .125 * boundaries)[...,None]
    # A segmented pale vein and an old dark scribe cross the natural grain field.
    vein_axis = .707 + .013*np.sin(15*x) + .007*np.cos(35*x)
    vein_width = .012 + .005*np.cos(41*x)
    vein = section & (np.abs(y-vein_axis) < vein_width)
    vein_tone = np.clip(.60 + .13*mottling + .07*joints, .34, .96)
    rock[vein] = vein_tone[vein,None] * [.97, .87, .74]
    for i in range(55):
        cx, cy = generator.uniform((.23,.275),(.84,.671))
        sx, sy = generator.uniform(.002,.014), generator.uniform(.004,.019)
        c, s = np.cos(i*.49), np.sin(i*.49)
        dx, dy = x-cx, y-cy
        kernel = ((c*dx+s*dy)/sx)**2 + ((-s*dx+c*dy)/sy)**2
        hole = (kernel < 1 + .16*mottling) & section
        rock[hole] *= np.clip(.075 + .11*kernel[hole,None], .02, .24)
    scribe = section & (np.abs(y-(.435+.014*np.sin(16*x)))<.0024) & (x>.275)&(x<.762)
    rock[scribe] *= .27
    # Forking hand-drawn grey pencil rule: a surface pigment that darkens every capture.
    pencil_trunk = polyline_band(x, y, PENCIL_TRUNK, .0060)
    pencil_kept = polyline_band(x, y, PENCIL_CONTINUATION, .0056)
    pencil_rule = section & (pencil_trunk | pencil_kept)
    # A graphite grey that stays clearly below the surrounding rock lightness in every lighting.
    pencil_tone = np.clip(0.07 + 0.02 * mottling, 0.04, 0.12)
    rock[pencil_rule] = rock[pencil_rule] * 0.28 + pencil_tone[pencil_rule, None] * [0.94, 0.95, 0.98]
    # The competing strike is a genuine thin fissure: negligible pigment, real relief.
    fissure_line = polyline_band(x, y, FISSURE_TRACE, .0016) & section
    rock[fissure_line] *= .985
    cross = np.zeros(shape, dtype=bool)
    for cx in (.086,.919):
        for cy in (.171,.797):
            cross |= ((abs(x-cx)<.014)&(abs(y-cy)<.0025))|((abs(x-cx)<.002)&(abs(y-cy)<.019))
    diffuse = np.where(section[...,None], rock, sheet)
    diffuse[cross] = [.055, .059, .052]
    relief = (.042 * boundaries + .020 * bedding + .032 * broad + .011*mottling) * section
    # Raised rim along the fissure so the raking pair flips sign while diffuse stays flat.
    fissure_relief = ndimage.gaussian_filter(fissure_line.astype(np.float32), 0.8)
    relief = relief + .34 * fissure_relief
    roughness = np.where(section, np.clip(.59 + .12*mottling + .1*boundaries, .31, .92), .79)
    neutral_card(diffuse,roughness,x,y)
    return ProofField(np.clip(diffuse,0,1),relief,np.where(section,.48+.28*vein,.035),
                      (tape|cross|scribe|pencil_rule).astype(np.float32),roughness,x,y)


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.ink_fraction * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.keyline_mask * 0.58)


def light_field(field, role):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def surface_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = surface_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions():
    return [
        rectangle("Complete core window", "inspection", 0.13, 0.22, 0.881, 0.766),
        rectangle("Vein reference patch", "glazed", 0.414, 0.700, 0.435, 0.719),
        rectangle("Matrix reference", "bare", 0.34, 0.38, 0.39, 0.41),
        rectangle("Neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Image scale", "ignore", 0.478, 0.883, 0.663, 0.941),
        rectangle("Central lamina window", "inspection", .232, .272, .786, .625),
    ]


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Synthetic geological core slice, calcite band, scribed line, scale and neutral reference.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 112, 84)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic rock-core cross-section. Mineral texture and reference marks occupy the same physical plane in each generated view.",
        "width_mm": 112,
        "height_mm": 84,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    return store.seed_tiles(DEMO_KEY, rows), True


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
