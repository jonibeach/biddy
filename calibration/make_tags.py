"""Generate a printable A4 sheet of AprilTags (tag36h11, IDs 0-15).

Print at 100% scale (no fit-to-page). Each black tag should measure 4.0 cm
edge to edge; check with a ruler before use.
"""
from pathlib import Path

import cv2
import numpy as np

DPI = 300
CM = DPI / 2.54
TAG_CM = 4.0
CELL_W_CM, CELL_H_CM = 5.2, 6.0
COLS, ROWS = 4, 4
PAGE_W, PAGE_H = int(21.0 * CM), int(29.7 * CM)

# Tag ID -> what it is attached to. Keep this in sync with tag_check.py.
ASSIGNMENTS = {
    **{i: "table" for i in range(0, 4)},             # work-area corners, seen by top camera
    **{i: "arm_base" for i in range(4, 8)},          # on the arm base plate, seen by top camera
    **{i: "inspection_stand" for i in range(8, 10)}, # on the inspection camera stand, seen by top camera
    10: "good_bowl",                                 # fixed to the bowl, seen by top camera
    11: "defective_bowl",
    **{i: "inspection_view" for i in range(12, 16)}, # fixed background in the inspection camera's view
}

def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    sheet = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    tag_px = int(TAG_CM * CM)
    cell_w, cell_h = int(CELL_W_CM * CM), int(CELL_H_CM * CM)
    margin_x = (PAGE_W - COLS * cell_w) // 2
    margin_y = int(1.5 * CM)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tag_id in range(16):
        r, c = divmod(tag_id, COLS)
        x0 = margin_x + c * cell_w
        y0 = margin_y + r * cell_h
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_px, borderBits=1)
        tx = x0 + (cell_w - tag_px) // 2
        ty = y0 + int(0.3 * CM)
        sheet[ty:ty + tag_px, tx:tx + tag_px] = tag
        label = f"ID {tag_id}: {ASSIGNMENTS[tag_id].replace('_', ' ')}"
        cv2.putText(sheet, label, (x0 + 10, ty + tag_px + int(0.8 * CM)),
                    font, 0.6, 0, 2, cv2.LINE_AA)

    # cv2.imwrite fails silently on non-ASCII paths on Windows, so encode and write ourselves.
    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        raise SystemExit("PNG encoding failed")
    out = Path(__file__).with_name("apriltags_sheet.png")
    out.write_bytes(buf.tobytes())
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
