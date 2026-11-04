"""Demo GIF from a real run: real seed-0 test items stepping through the loop gate (results/seed0.json -> docs/screenshots/demo.gif)."""
import json
from PIL import Image, ImageDraw, ImageFont

R = json.load(open("results/seed0.json"))
items, N = R["items"][:30], R["nmax"]
W, H, X0, Y0, CW, RH = 960, 540, 150, 92, 64, 12
mono = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", s)
F, FS, FB = mono(13), mono(11), mono(20)


def frame(loop):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.text((24, 20), "haltgate", font=FB, fill=255)
    d.text((24, 48), f"real run · seed {R['seed']} · SYNTH test items · α={R['alpha']} δ={R['delta']}", font=FS, fill=120)
    for i in range(N):
        d.text((X0 + i * CW + CW // 2 - 4, Y0 - 18), str(i + 1), font=FS, fill=200 if i < loop else 70)
    alive = spent = right = 0
    for r, it in enumerate(items):
        y = Y0 + r * RH
        d.text((24, y - 1), it["task"], font=FS, fill=110)
        upto = min(loop, it["halt"])
        spent += upto
        for i in range(upto):
            c = it["correct"][i]
            d.rectangle([X0 + i * CW + 2, y, X0 + (i + 1) * CW - 3, y + RH - 3], fill=235 if c else 0, outline=150)
        if it["halt"] <= loop:
            x = X0 + it["halt"] * CW
            d.line([x, y - 1, x, y + RH - 2], fill=255, width=2)
            d.text((X0 + N * CW + 10, y - 1), it["decision"], font=FS, fill=200)
            right += it["correct"][it["halt"] - 1]
        else:
            alive += 1
            d.text((X0 + N * CW + 10, y - 1), "…", font=FS, fill=90)
    done = len(items) - alive
    d.text((24, H - 46), f"loop {loop}/{N}   alive {alive:2d}   halted {done:2d}   correct {right:2d}/{done:2d}", font=F, fill=230)
    d.text((24, H - 26), f"loops spent {spent} vs fixed-{N} {loop * len(items)}  ■ correct □ wrong │ gate decision", font=FS, fill=120)
    return im


frames = [frame(i) for i in range(1, N + 1)]
frames[0].save("docs/screenshots/demo.gif", save_all=True, append_images=frames[1:] + [frames[-1]] * 3, duration=700, loop=0)
