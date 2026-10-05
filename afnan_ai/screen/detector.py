"""Visual element detection from pixels.

Where DOM/accessibility data exists it always wins (confidence
1.0); this detector is the fallback for everything else —
canvas apps, remote-desktop windows, custom-drawn UI.  It finds
prominent rectangular regions that stand out from the
background (buttons, cards, panels, input boxes) and scores
each one.

The score is honest about what it measures: rectangularity
(how completely the shape fills its bounding box) and contrast
(how strongly it stands out from the background).  A crisp,
solid button scores high; faint or fragmented shapes score
low, and low scores are exactly what the approval/verify tiers
in :class:`~afnan_ai.screen.observer.ScreenObserver` exist for.

Pure stdlib, deterministic, and safe on any platform.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from afnan_ai.screen.base import VisualRegion
from afnan_ai.screen.image import PixelImage

#: how far a pixel must be from the background to count as shape
_CONTRAST_THRESHOLD = 36
#: detection runs on a grid no larger than this (speed)
_MAX_GRID = 320


@dataclass
class RawDetection:
    region: VisualRegion
    kind: str
    confidence: float


class ElementDetector(ABC):
    """Turns a decoded screenshot into candidate UI elements."""

    @abstractmethod
    def detect(self, image: PixelImage) -> list[RawDetection]:
        """Return candidate elements found in *image*."""


def _distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / 3.0


class RegionDetector(ElementDetector):
    """Connected-component detector over a contrast mask."""

    def detect(self, image: PixelImage) -> list[RawDetection]:
        factor = max(1, -(-max(image.width, image.height) // _MAX_GRID))
        gw = max(1, image.width // factor)
        gh = max(1, image.height // factor)
        background = image.pixel(0, 0)

        mask = [[False] * gw for _ in range(gh)]
        strength = [[0.0] * gw for _ in range(gh)]
        for gy in range(gh):
            for gx in range(gw):
                px = image.pixel(gx * factor, gy * factor)
                d = _distance(px, background)
                if d > _CONTRAST_THRESHOLD:
                    mask[gy][gx] = True
                    strength[gy][gx] = d

        detections: list[RawDetection] = []
        min_cells = max(6, (gw * gh) // 4000)
        for gy in range(gh):
            for gx in range(gw):
                if not mask[gy][gx]:
                    continue
                cells, bounds, total = self._flood(mask, strength, gx, gy)
                if cells < min_cells:
                    continue
                x0, y0, x1, y1 = bounds
                region = VisualRegion(
                    x=x0 * factor,
                    y=y0 * factor,
                    width=(x1 - x0 + 1) * factor,
                    height=(y1 - y0 + 1) * factor,
                )
                # ignore shapes that are basically the whole screen
                if region.area > 0.92 * image.width * image.height:
                    continue
                bbox_cells = (x1 - x0 + 1) * (y1 - y0 + 1)
                fill = cells / bbox_cells
                contrast = min(1.0, (total / cells) / 160.0)
                confidence = max(
                    0.05,
                    min(0.98, 0.10 + 0.35 * fill + 0.55 * contrast),
                )
                aspect = region.width / max(1, region.height)
                if 1.5 <= aspect <= 7 and 10 <= region.height <= 110:
                    kind = "button_like"
                else:
                    kind = "region"
                detections.append(
                    RawDetection(region, kind, round(confidence, 3))
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:50]

    @staticmethod
    def _flood(mask, strength, sx, sy):
        """Breadth-first component fill; clears visited cells."""
        queue = deque([(sx, sy)])
        mask[sy][sx] = False
        cells = 0
        total = 0.0
        x0 = x1 = sx
        y0 = y1 = sy
        while queue:
            x, y = queue.popleft()
            cells += 1
            total += strength[y][x]
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (
                    0 <= ny < len(mask)
                    and 0 <= nx < len(mask[0])
                    and mask[ny][nx]
                ):
                    mask[ny][nx] = False
                    queue.append((nx, ny))
        return cells, (x0, y0, x1, y1), total
