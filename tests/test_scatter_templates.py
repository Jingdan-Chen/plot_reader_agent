"""Tests use known image coordinates, independently of a model's predictions."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from plot_reader.scatter_templates import _subpixel, scan_templates, validate_profile, legend_components


def fixture(tmp_path, shape="square", line=True):
    image = Image.new("L", (320, 240), "white")
    d = ImageDraw.Draw(image)

    def glyph(x, y):
        if shape == "square":
            d.rectangle((x-5, y-5, x+5, y+5), outline=0, width=2)
        elif shape == "triangle":
            d.polygon([(x, y-5), (x-5, y+5), (x+5, y+5)], outline=0, width=2)
        elif shape == "inverted":
            d.polygon([(x-5, y-5), (x+5, y-5), (x, y+5)], outline=0, width=2)
        elif shape == "diamond":
            d.polygon([(x, y-5), (x+5, y), (x, y+5), (x-5, y)], outline=0, width=2)
        elif shape == "cross":
            d.line((x-5, y-5, x+5, y+5), fill=0, width=1)
            d.line((x+5, y-5, x-5, y+5), fill=0, width=1)
        elif shape == "plus":
            d.line((x-5, y, x+5, y), fill=0, width=2)
            d.line((x, y-5, x, y+5), fill=0, width=2)

    glyph(40, 200)  # Clean template outside the data rectangle.
    if line:
        d.line((15, 80, 300, 80), fill=0, width=1)
        d.line((15, 140, 300, 140), fill=0, width=2)  # Marker-free nuisance line.
    truth = [[70, 80], [150, 80], [250, 80]]
    for x, y in truth:
        glyph(x, y)
    source = tmp_path/f"{shape}.png"
    image.save(source)
    components = [{"id": "G1", "bounds": [35, 195, 46, 206]}]
    profile = {"applicable": True, "templates": [{"series": shape, "component_id": "G1"}]}
    layout = {"plot_bounds": [10, 10, 310, 165], "exclude_regions": []}
    return source, components, profile, layout, truth


@pytest.mark.parametrize("shape", ["square", "triangle", "inverted", "diamond", "cross", "plus"])
def test_repeated_markers_on_line_with_no_ghosts(tmp_path, shape):
    source, components, profile, layout, truth = fixture(tmp_path, shape)
    candidates, _ = scan_templates(source, components, profile, layout)
    assert len(candidates) == len(truth)
    for point in truth:
        assert min(math.dist(point, c["pixel"]) for c in candidates) < .85
    assert all(c["offset_limit"] == 0 for c in candidates)


def test_line_only_source_abstains_and_blank_map_is_finite(tmp_path):
    source, components, profile, layout, _ = fixture(tmp_path)
    im = Image.open(source); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 319, 170), fill=255)
    d.line((10, 80, 310, 80), fill=0, width=2)
    im.save(source)
    candidates, maps = scan_templates(source, components, profile, layout)
    assert not candidates
    assert all(np.isfinite(m).all() for m in maps.values())


def test_model_cannot_use_foreign_or_duplicate_glyph(tmp_path):
    _, components, profile, _, _ = fixture(tmp_path)
    profile["templates"][0]["component_id"] = "foreign"
    with pytest.raises(ValueError, match="Unknown"):
        validate_profile(profile, components, ["square"])
    profile["templates"] = [{"series": s, "component_id": "G1"} for s in ["one", "two"]]
    with pytest.raises(ValueError, match="multiple"):
        validate_profile(profile, components, ["one", "two"])
    profile["applicable"] = False
    with pytest.raises(ValueError, match="not applicable"):
        validate_profile(profile, components, ["one", "two"])


def test_exclusions_and_center_response_coordinates(tmp_path):
    source, components, profile, layout, truth = fixture(tmp_path)
    layout["exclude_regions"] = [[140, 65, 160, 95]]
    candidates, maps = scan_templates(source, components, profile, layout)
    assert len(candidates) == 2
    response = maps["square"]
    assert response[80, 150] == 0
    assert response[80, 70] > .7
    assert min(math.dist([70, 80], p["pixel"]) for p in candidates) < .5


def test_cross_series_competition_remains_explicit(tmp_path):
    source, components, profile, layout, _ = fixture(tmp_path)
    # Two visually identical legends cannot establish series identity.
    components.append({"id": "G2", "bounds": components[0]["bounds"]})
    profile["templates"].append({"series": "other", "component_id": "G2"})
    candidates, _ = scan_templates(source, components, profile, layout)
    assert len(candidates) == 6
    assert all(c["status"] == "needs_review" and c["alternatives"] for c in candidates)


def test_subpixel_peak_and_flat_response():
    y, x = np.mgrid[:9, :9]
    response = -((x-4.3)**2+2*(y-3.8)**2)
    offset, stable = _subpixel(response, 4, 4)
    assert stable and offset == pytest.approx([.3, -.2])
    assert _subpixel(np.ones((9, 9)), 4, 4) == ([0., 0.], False)
    assert _subpixel(response, 0, 0) == ([0., 0.], False)


def test_legend_ids_are_deterministic_and_native(tmp_path):
    source, _, _, _, _ = fixture(tmp_path)
    components = legend_components(source, [[15, 175, 65, 225]])
    assert components == [{"id": "G1", "bounds": [35, 195, 46, 206]}]
    assert components == legend_components(source, [[15, 175, 65, 225]])


def test_antialiased_translated_centers_do_not_follow_ink_centroid(tmp_path):
    source, components, profile, layout, truth = fixture(tmp_path, "triangle", line=False)
    arr = np.array(Image.open(source))
    shifted = cv2.warpAffine(arr[:170], np.float32([[1, 0, .35], [0, 1, -.25]]),
                            (320, 170), flags=cv2.INTER_LINEAR, borderValue=255)
    arr[:170] = shifted
    Image.fromarray(arr).save(source)
    candidates, _ = scan_templates(source, components, profile, layout, scales=(1.,))
    assert len(candidates) == 3
    for x, y in truth:
        assert min(math.dist([x+.35, y-.25], p["pixel"]) for p in candidates) < .3


def test_semantic_review_cannot_drop_or_invent_candidates():
    from experiments.scatter_templates import validate_review
    candidates = [{"id": "T1"}, {"id": "T2"}]
    valid = {"items": [{"candidate_id": "T1", "verdict": "supported"},
                       {"candidate_id": "T2", "verdict": "uncertain"}]}
    validate_review(valid, candidates)
    for items in [valid["items"][:1], valid["items"]*2,
                  [{"candidate_id": "T1", "verdict": "supported"}, {"candidate_id": "T3", "verdict": "supported"}]]:
        with pytest.raises(ValueError, match="every candidate"):
            validate_review({"items": items}, candidates)
