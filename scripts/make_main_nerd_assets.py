"""Make the two picture assets of paper 1's pages on the Side Nerd Blog from the owner's mascot GIF. Usage:

    python scripts/make_main_nerd_assets.py "PATH/TO/Sleepy Cow's Morning Stretch.gif"

Writes docs/public/mainnerd/assets/01thecurledtorusburps/sleepy-cow.gif (the same two frames with the empty margin cropped away,
so the drawing fills its box) and og-image.png (the 1200 x 630 card that link previews show). Needs Pillow.
The GIF is the owner's; nothing is redrawn.
"""
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parents[1] / "docs/public/mainnerd/assets/01thecurledtorusburps"
MIDNIGHT, NAVY, AQUA, AQUA_LIGHT, WHITE = "#0d1419", "#192933", "#34b1e8", "#5bc4f0", "#ffffff"


def frames(path):
    im = Image.open(path)
    out, durations = [], []
    for k in range(getattr(im, "n_frames", 1)):
        im.seek(k)
        out.append(im.convert("RGB"))
        durations.append(im.info.get("duration", 2400))
    return out, durations


def content_box(images, margin=14):
    """The box that holds the drawing in every frame."""
    box = None
    for im in images:
        diff = ImageChops.difference(im, Image.new("RGB", im.size, WHITE)).convert("L").point(lambda v: 255 * (v > 24))
        b = diff.getbbox()
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    w, h = images[0].size
    return max(0, box[0] - margin), max(0, box[1] - margin), min(w, box[2] + margin), min(h, box[3] + margin)


def font(size, bold=False, serif=True):
    names = (["timesbd.ttf", "georgiab.ttf"] if bold else ["times.ttf", "georgia.ttf"]) if serif else \
            (["arialbd.ttf"] if bold else ["arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/" + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(source):
    ASSETS.mkdir(parents=True, exist_ok=True)
    images, durations = frames(source)
    box = content_box(images)
    cropped = [im.crop(box) for im in images]
    cropped[0].save(ASSETS / "sleepy-cow.gif", save_all=True, append_images=cropped[1:], duration=durations, loop=0)
    print("sleepy-cow.gif", cropped[0].size, [d for d in durations])

    card = Image.new("RGB", (1200, 630), MIDNIGHT)
    draw = ImageDraw.Draw(card)
    for k, color in enumerate((AQUA, AQUA_LIGHT, NAVY)):
        draw.rectangle([10 + 6 * k, 10 + 6 * k, 1189 - 6 * k, 619 - 6 * k], outline=color, width=6)
    cow = cropped[-1]
    scale = 470 / cow.height
    cow = cow.resize((round(cow.width * scale), 470), Image.LANCZOS)
    x0 = 1200 - cow.width - 70
    draw.rectangle([x0 - 10, 70, x0 + cow.width + 10, 560], fill=WHITE, outline=AQUA, width=4)
    card.paste(cow, (x0, 80))
    draw.text((70, 84), "THE SIDE NERD BLOG  ·  POST No. 01", font=font(30, bold=True, serif=False), fill=AQUA)
    y = 150
    for line in ("The curled", "torus burps"):
        draw.text((70, y), line, font=font(104, bold=True), fill=WHITE)
        y += 112
    y += 22
    for line in ("A small push, a large release,", "and the question of what survives"):
        draw.text((70, y), line, font=font(36), fill="#e2e8f0")
        y += 46
    draw.text((70, 540), "Emily Smith  ·  sidenerdapps.com/mainnerd", font=font(26, serif=False), fill=AQUA_LIGHT)
    card.save(ASSETS / "og-image.png", optimize=True)
    print("og-image.png", card.size)


if __name__ == "__main__":
    main(sys.argv[1])
