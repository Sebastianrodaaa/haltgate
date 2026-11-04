"""1280x400 black-and-white banner: pixel blackletter name, mono tagline, dithered public-domain engravings at the edges."""
import sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 400
FONT = sys.argv[1] if len(sys.argv) > 1 else "UnifrakturMaguntia-Book.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def plate(path, box, side):
    im = Image.open(path).convert("L")
    im = im.crop(box).resize((360, H), Image.LANCZOS)  # box is 0.9:1 so nothing is squashed
    px = im.load()
    for x in range(im.width):  # fade toward the centre so the plate dissolves into the black
        f = (1 - x / im.width) if side == "l" else x / im.width
        f = min(1.0, max(0.0, f * 1.5 - 0.1)) ** 1.6
        for y in range(H):
            px[x, y] = int(px[x, y] * f * 0.85)
    for y in range(H):  # soft top/bottom edges too
        g = min(1.0, y / 30, (H - 1 - y) / 30)
        for x in range(im.width):
            px[x, y] = int(px[x, y] * g)
    return im.convert("1", dither=Image.FLOYDSTEINBERG).convert("L")


img = Image.new("L", (W, H), 0)
img.paste(plate("docs/assets/dore_gate.jpg", (150, 40, 730, 684), "l"), (0, 0))
img.paste(plate("docs/assets/durer_mel.jpg", (250, 60, 760, 627), "r"), (W - 360, 0))
small = Image.new("L", (300, 60), 0)  # render tiny, upscale nearest: pixel blackletter
ImageDraw.Draw(small).text((150, 30), "haltgate", font=ImageFont.truetype(FONT, 46), fill=255, anchor="mm")
small = small.point(lambda v: 255 if v > 110 else 0).resize((900, 180), Image.NEAREST)
img.paste(235, (190, 70), small)
d = ImageDraw.Draw(img)
mono = ImageFont.truetype(MONO, 22)
for i, line in enumerate(["a calibrated gate for looped transformers", "continue, halt or escalate"]):
    d.text((W // 2, 268 + 34 * i), line, font=mono, fill=170, anchor="mm")
img.save("docs/screenshots/banner.png", optimize=True)
