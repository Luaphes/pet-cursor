# 注意：给同事用的换皮已搬进 App（mac-app/SkinMaker.swift，菜单「切换宠物 → 添加宠物…」直接选总图）。
# 本脚本只留一个用途：从橘猫的旧版切图（--sprites/--idle）生成 skins/橘猫 文件夹。总图模式与 Swift 版逻辑相同，改动以 Swift 版为准。
"""宠物指针换皮：一张 4×5 姿势总图 → 一个宠物文件夹，在「宠物指针」App 的「切换宠物」菜单里用。

用法：
  python3 make_skin.py <总图.png> --name 团子 [--out 输出目录] [--install]
  python3 make_skin.py --sprites <切图目录> --idle <待机切图目录> --name 橘猫 --out <输出目录>   # 已有切图，只生成宠物文件夹
--install：直接放进本机 App 的宠物文件夹（~/Library/Application Support/OrangeCatCursor/Skins/），菜单里马上能选。

总图要求见 App 菜单「切换宠物 → 复制出图提示词」：同一只宠物 20 个姿势，4 列 5 行，纯色背景。
  1–16 格：16 种指针状态的宠物姿势（一只前爪举向左上）
  17–19 格：3 个跳舞姿势；20 格：蜷着睡觉
指针符号（箭头、I 形、双向箭头…）由本脚本画，点击点固定，不靠 AI 画，换什么动物都对得准。
依赖：python3 + Pillow、numpy、scipy（pip3 install pillow numpy scipy）。
"""
import argparse, json, os, plistlib, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["01_正常", "02_文字输入", "03_链接", "04_不可用", "05_忙碌", "06_移动", "07_左右调整", "08_上下调整",
         "09_斜向调整_左上右下", "10_斜向调整_右上左下", "11_帮助", "12_十字瞄准", "13_复制", "14_张开爪", "15_抓紧爪", "16_放大"]
OUTLINE = (58, 32, 20, 255)
WHITE = (255, 255, 255, 255)
S = 4                                    # 符号 4 倍画再缩小，边缘平滑
OW = 5                                   # 描边粗细（256 画布像素）
PET_H = 162                              # 「正常」姿势宠物的高度（256 画布）
FOOT_Y, PET_CX, PET_MAXW = 251, 166, 170 # 宠物脚底、身体中线、最大宽度
FONT = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
NOTE_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
TEXT = dict(fill=(70, 38, 22, 255), stroke_width=3, stroke_fill=WHITE)   # 音符和 z：深棕字白描边，亮暗背景都看得见


# ---------- 1. 抠图、切格 ----------
def remove_bg(im):
    """背景色取四边像素的中位数；按颜色距离抠掉，边缘半透明像素反混合掉背景色。"""
    a = np.array(im.convert("RGB")).astype(float)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((a - bg) ** 2).sum(-1))
    alpha = np.clip((dist - 60) / 80, 0, 1)                     # 距离 <60 全透明，>140 全不透明
    al = np.maximum(alpha, 1e-3)[..., None]
    rgb = np.clip((a - (1 - al) * bg) / al, 0, 255)              # 去掉混进边缘的背景色
    out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    return Image.fromarray(out), bg


def split_cells(im, cols=4, rows=5):
    """按连通块归格：每块按重心落在哪一格归到哪一格，不怕图里的格子画得不齐。"""
    a = np.array(im)
    solid = a[..., 3] > 100
    lab, n = ndimage.label(ndimage.binary_dilation(solid, iterations=3))   # 稍微膨胀，汗滴这类小件并进身体
    H, W = solid.shape
    cells = [np.zeros_like(solid) for _ in range(cols * rows)]
    sizes = ndimage.sum(solid, lab, range(1, n + 1))
    coms = ndimage.center_of_mass(solid, lab, range(1, n + 1))
    for k in range(n):
        if sizes[k] < 200: continue                                       # 零星杂点
        cy, cx = coms[k]
        idx = min(int(cy / H * rows), rows - 1) * cols + min(int(cx / W * cols), cols - 1)
        cells[idx] |= lab == k + 1
    out = []
    for i, m in enumerate(cells):
        ys, xs = np.nonzero(m)
        if not len(xs): sys.exit(f"第 {i + 1} 格是空的：总图要 4 列 5 行、每格一只，宠物之间留够空白")
        b = a.copy(); b[~ndimage.binary_dilation(m, iterations=2)] = 0
        out.append(Image.fromarray(b).crop((xs.min() - 2, ys.min() - 2, xs.max() + 3, ys.max() + 3)))
    return out


def place(pet, sc, canvas=256):
    """宠物按统一比例缩放，脚底对齐、放在画布右下；太宽的姿势单独缩小。"""
    w, h = round(pet.width * sc), round(pet.height * sc)
    if w > PET_MAXW: h, w = round(h * PET_MAXW / w), PET_MAXW
    if h > 200: w, h = round(w * 200 / h), 200
    p = pet.resize((w, h), Image.LANCZOS)
    x = min(round(PET_CX - w / 2), canvas - 2 - w)
    layer = Image.new("RGBA", (canvas, canvas))
    layer.alpha_composite(p, (x, FOOT_Y - h))
    return layer


# ---------- 2. 指针符号（256 画布坐标，返回 符号层, 点击点） ----------
def new():
    return Image.new("RGBA", (256 * S, 256 * S))


def done(img):
    return img.resize((256, 256), Image.LANCZOS)


def P(pts):
    return [(x * S, y * S) for x, y in pts]


def stroke_fill(img, shapes, fill=WHITE):
    """先把所有形状加粗画成描边色，再统一填白：多个形状合成一个外轮廓。shapes = [(kind, args)]"""
    d = ImageDraw.Draw(img)
    for pass_ in (0, 1):
        for kind, arg in shapes:
            col = OUTLINE if pass_ == 0 else (arg.get("fill") or fill)
            w = OW * S * 2 if pass_ == 0 else 0
            if kind == "poly":
                d.polygon(P(arg["pts"]), fill=col, outline=OUTLINE if pass_ == 0 else None, width=w or 1)
                if pass_ == 0:
                    d.line(P(arg["pts"] + arg["pts"][:1]), fill=OUTLINE, width=w, joint="curve")
            elif kind == "ellipse":
                x0, y0, x1, y1 = arg["box"]; e = OW if pass_ == 0 else 0
                d.ellipse([(x0 - e) * S, (y0 - e) * S, (x1 + e) * S, (y1 + e) * S], fill=col)
            elif kind == "rrect":
                x0, y0, x1, y1 = arg["box"]; e = OW if pass_ == 0 else 0
                d.rounded_rectangle([(x0 - e) * S, (y0 - e) * S, (x1 + e) * S, (y1 + e) * S], (arg.get("r", 3) + e) * S, fill=col)
    return img


def arrow_pts(tip=(8, 6), k=1.15):
    base = [(0, 0), (0, 96), (23, 75), (38, 109), (54, 102), (39, 69), (69, 69)]
    return [(tip[0] + x * k, tip[1] + y * k) for x, y in base]


def double_arrow(c, ang, L=96, sw=8, hw=22, hl=26):
    pts = [(L / 2, 0), (L / 2 - hl, hw), (L / 2 - hl, sw), (-L / 2 + hl, sw), (-L / 2 + hl, hw), (-L / 2, 0),
           (-L / 2 + hl, -hw), (-L / 2 + hl, -sw), (L / 2 - hl, -sw), (L / 2 - hl, -hw)]
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    return [(c[0] + x * ca - y * sa, c[1] + x * sa + y * ca) for x, y in pts]


def paw(c, k=1.0, toes=True, point=False):
    """爪子：掌垫 + 4 个趾头。point=True 时中间一个趾头伸长当「指」。"""
    cx, cy = c
    sh = [("ellipse", {"box": (cx - 30 * k, cy - 8 * k, cx + 30 * k, cy + 40 * k)})]
    if toes:
        for dx, dy, r in [(-34, -16, 12), (-13, -34, 13), (13, -34, 13), (34, -16, 12)]:
            if point and dx == -13:
                sh.append(("rrect", {"box": (cx + (dx - 12) * k, cy - 76 * k, cx + (dx + 12) * k, cy - 10 * k), "r": 12 * k}))
            else:
                sh.append(("ellipse", {"box": (cx + (dx - r) * k, cy + (dy - r) * k, cx + (dx + r) * k, cy + (dy + r) * k)}))
    return sh


def pink_pads(img, c, k=1.0, point=False):
    d = ImageDraw.Draw(img); cx, cy = c; pink = (255, 170, 185, 255)
    d.ellipse(P([(cx - 16 * k, cy + 6 * k), (cx + 16 * k, cy + 30 * k)]), fill=pink)
    for dx, dy, r in [(-34, -16, 6), (-13, -34, 7), (13, -34, 7), (34, -16, 6)]:
        if point and dx == -13: dy = -62
        d.ellipse(P([(cx + (dx - r) * k, cy + (dy - r) * k), (cx + (dx + r) * k, cy + (dy + r) * k)]), fill=pink)


def symbols():
    out = {}
    tip = (8, 6)
    out["01_正常"] = (done(stroke_fill(new(), [("poly", {"pts": arrow_pts(tip)})])), tip)

    img = stroke_fill(new(), [("rrect", {"box": (32, 26, 40, 126), "r": 3}),
                              ("rrect", {"box": (18, 20, 54, 30), "r": 4}), ("rrect", {"box": (18, 122, 54, 132), "r": 4})])
    out["02_文字输入"] = (done(img), (36, 76))

    img = stroke_fill(new(), paw((52, 92), 0.95, point=True)); pink_pads(img, (52, 92), 0.95, point=True)
    out["03_链接"] = (done(img), (40, 20))

    img = new(); d = ImageDraw.Draw(img); c, r = (60, 62), 46
    d.ellipse(P([(c[0] - r - OW, c[1] - r - OW), (c[0] + r + OW, c[1] + r + OW)]), fill=OUTLINE)
    d.ellipse(P([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]), fill=(225, 45, 40, 255))
    d.ellipse(P([(c[0] - r + 13, c[1] - r + 13), (c[0] + r - 13, c[1] + r - 13)]), fill=OUTLINE)
    d.ellipse(P([(c[0] - r + 13 + OW, c[1] - r + 13 + OW), (c[0] + r - 13 - OW, c[1] + r - 13 - OW)]), fill=WHITE)
    a = np.radians(45); dx, dy = (r - 8) * np.cos(a), (r - 8) * np.sin(a)
    d.line(P([(c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy)]), fill=OUTLINE, width=(13 + 2 * OW) * S)
    d.line(P([(c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy)]), fill=(225, 45, 40, 255), width=13 * S)
    out["04_不可用"] = (done(img), c)

    c = (60, 64)
    out["06_移动"] = (done(stroke_fill(new(), [("poly", {"pts": double_arrow(c, 0, 104)}), ("poly", {"pts": double_arrow(c, 90, 104)})])), c)
    out["07_左右调整"] = (done(stroke_fill(new(), [("poly", {"pts": double_arrow((66, 44), 0, 118)})])), (66, 44))
    out["08_上下调整"] = (done(stroke_fill(new(), [("poly", {"pts": double_arrow((40, 74), 90, 118)})])), (40, 74))
    out["09_斜向调整_左上右下"] = (done(stroke_fill(new(), [("poly", {"pts": double_arrow(c, 45, 116)})])), c)
    out["10_斜向调整_右上左下"] = (done(stroke_fill(new(), [("poly", {"pts": double_arrow(c, -45, 116)})])), c)

    c = (54, 54)
    img = stroke_fill(new(), [("ellipse", {"box": (c[0] - 40, c[1] - 40, c[0] + 40, c[1] + 40)})])
    f = ImageFont.truetype(FONT, 60 * S)
    ImageDraw.Draw(img).text((c[0] * S, (c[1] + 2) * S), "?", font=f, fill=OUTLINE, anchor="mm")
    out["11_帮助"] = (done(img), c)

    c = (60, 64); img = new(); d = ImageDraw.Draw(img)
    for col, w in ((WHITE, 9 + 2 * OW // 2 * 2), (OUTLINE, 6)):
        d.ellipse(P([(c[0] - 34, c[1] - 34), (c[0] + 34, c[1] + 34)]), outline=col, width=w * S)
        for (x0, y0, x1, y1) in [(c[0], c[1] - 56, c[0], c[1] - 14), (c[0], c[1] + 14, c[0], c[1] + 56),
                                 (c[0] - 56, c[1], c[0] - 14, c[1]), (c[0] + 14, c[1], c[0] + 56, c[1])]:
            d.line(P([(x0, y0), (x1, y1)]), fill=col, width=w * S)
    out["12_十字瞄准"] = (done(img), c)

    img = stroke_fill(new(), [("poly", {"pts": arrow_pts(tip)})]); d = ImageDraw.Draw(img); c, r = (84, 100), 22
    d.ellipse(P([(c[0] - r - OW, c[1] - r - OW), (c[0] + r + OW, c[1] + r + OW)]), fill=OUTLINE)
    d.ellipse(P([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]), fill=(52, 168, 83, 255))
    d.rounded_rectangle(P([(c[0] - 13, c[1] - 3.5), (c[0] + 13, c[1] + 3.5)]), 2 * S, fill=WHITE)
    d.rounded_rectangle(P([(c[0] - 3.5, c[1] - 13), (c[0] + 3.5, c[1] + 13)]), 2 * S, fill=WHITE)
    out["13_复制"] = (done(img), tip)

    c = (58, 66)
    img = stroke_fill(new(), paw(c, 1.0)); pink_pads(img, c, 1.0)
    out["14_张开爪"] = (done(img), (c[0], c[1] + 8))
    img = stroke_fill(new(), [("ellipse", {"box": (c[0] - 34, c[1] - 30, c[0] + 34, c[1] + 34)})])
    d = ImageDraw.Draw(img)
    for dx in (-14, 0, 14):
        d.line(P([(c[0] + dx, c[1] - 30), (c[0] + dx, c[1] - 12)]), fill=OUTLINE, width=4 * S)
    out["15_抓紧爪"] = (done(img), c)

    c = (50, 50)
    img = stroke_fill(new(), [("poly", {"pts": double_arrow((c[0] + 46, c[1] + 46), 45, 40, sw=8, hw=8, hl=1)}),
                              ("ellipse", {"box": (c[0] - 36, c[1] - 36, c[0] + 36, c[1] + 36)})])
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(P([(c[0] - 20, c[1] - 5), (c[0] + 20, c[1] + 5)]), 3 * S, fill=OUTLINE)
    d.rounded_rectangle(P([(c[0] - 5, c[1] - 20), (c[0] + 5, c[1] + 20)]), 3 * S, fill=OUTLINE)
    out["16_放大"] = (done(img), c)
    return out


def busy_frames(n=12, c=(74, 70), R=54, W=20):
    frames = []
    base_col = np.array([255, 160, 50])
    for f in range(n):
        img = new(); d = ImageDraw.Draw(img)
        cx, cy = c[0] * S, c[1] * S
        ro, ri = (R + W / 2) * S, (R - W / 2) * S
        d.ellipse([cx - ro - 10, cy - ro - 10, cx + ro + 10, cy + ro + 10], fill=OUTLINE)
        for k in range(n):
            t = ((f - k) % n) / (n - 1)
            col = tuple(int(v) for v in base_col * (1 - t) + np.array([255, 236, 210]) * t) + (255,)
            a0 = -90 + k * 360 / n + 3
            d.pieslice([cx - ro, cy - ro, cx + ro, cy + ro], a0, a0 + 360 / n - 6, fill=col)
        d.ellipse([cx - ri - 10, cy - ri - 10, cx + ri + 10, cy + ri + 10], fill=OUTLINE)
        d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=(0, 0, 0, 0))
        frames.append(done(img))
    return frames, c


def on_top(pet_layer, sym):
    out = pet_layer.copy(); out.alpha_composite(sym); return out


# ---------- 3. 待机动画 ----------
def overlay_text(frame, xy, s, font, alpha):
    l = Image.new("RGBA", frame.size)
    ImageDraw.Draw(l).text(xy, s, font=font, **TEXT)
    l.putalpha(Image.fromarray((np.array(l)[..., 3].astype(int) * alpha // 255).astype(np.uint8)))
    frame.alpha_composite(l)


def dance_frames(arrow, poses, sc, per_beat=8, hop=14, tilt=5):
    """3 个舞姿各跳一拍：最高点 → 落下 → 落地压扁 → 弹起；最高点换舞姿。脚底中点对齐「正常」姿势。"""
    note_font = ImageFont.truetype(NOTE_FONT, 46)

    def hop_at(p):
        if p < 0.3: return hop * (1 - (p / 0.3) ** 2), 0.98, 1.03
        if p < 0.5: q = np.sin(np.pi * (p - 0.3) / 0.2); return 0, 1 + 0.08 * q, 1 - 0.1 * q
        return hop * (1 - ((1 - p) / 0.5) ** 2), 0.98, 1.03

    frames = []
    for b, pet in enumerate(poses):
        base = place(pet, sc)                                    # 已经脚底对齐的 256 层
        ys, xs = np.nonzero(np.array(base)[..., 3] > 0)
        mid, top = (xs.min() + xs.max()) / 2, ys.min()
        sign = 1 if b % 2 == 0 else -1
        for j in range(per_beat):
            p = j / per_beat
            y, kx, ky = hop_at(p)
            big = Image.new("RGBA", (512, 512)); big.alpha_composite(base, (256 - round(mid), 256 - FOOT_Y))
            # 以脚底中点 (256,256) 为原点压扁/拉长、旋转
            w2, h2 = round(512 * kx), round(512 * ky)
            sq = big.resize((w2, h2), Image.LANCZOS)
            big = Image.new("RGBA", (512, 512)); big.alpha_composite(sq, (round(256 - 256 * kx), round(256 - 256 * ky)))
            big = big.rotate(sign * tilt * np.sin(np.pi * p), resample=Image.BICUBIC, center=(256, 256))
            frame = Image.new("RGBA", (256, 256))
            ox, oy = round(256 - mid), round(256 - (FOOT_Y - y))
            frame.alpha_composite(big, (0, 0), (ox, oy, ox + 256, oy + 256))
            frame.alpha_composite(arrow)
            for k in range(2):                                       # 本拍和上一拍的音符同时在飘
                age = (p + k) / 2
                side = 1 if (b - k) % 2 == 0 else -1
                nx, ny = mid + 50 + side * 10 + age * 8, top - 40 - age * 30
                if nx < 226 and ny > -10:
                    overlay_text(frame, (nx, ny), "♪" if (b - k) % 2 == 0 else "♫", note_font, int(255 * (1 - age) ** 0.8))
            frames.append(frame)
    return frames


def sleep_frames(arrow, pet, sc, n=6):
    base = place(pet, sc)
    ys, xs = np.nonzero(np.array(base)[..., 3] > 0)
    head = (xs.max() - 60, ys.min() - 4)
    out = []
    for f in range(n):
        t = f / n
        breathe = 1 + 0.025 * np.sin(2 * np.pi * t)
        h = round(256 * breathe)
        b = base.resize((256, h), Image.LANCZOS).crop((0, h - 256, 256, h))
        frame = Image.new("RGBA", (256, 256)); frame.alpha_composite(b); frame.alpha_composite(arrow)
        for k, (size, phase) in enumerate([(34, 0.0), (42, 0.33), (50, 0.66)]):
            p = (t + phase) % 1
            overlay_text(frame, (head[0] + k * 8 + p * 16, head[1] - 8 - p * 44), "z",
                         ImageFont.truetype(FONT, size), int(255 * (1 - p) ** 0.7))
        out.append(frame)
    return out


# ---------- 4. 生成宠物文件夹 ----------
def make_skin_dir(sprites, idle, name, out):
    """切图 → 宠物文件夹 <out>/<name>/：theme.cursor（37 个系统指针）+ idle.plist（跳舞/睡觉）+ MenuIcon.png。"""
    tmp = tempfile.mkdtemp()
    env = dict(os.environ, MAC_POINTS="40", IDLE_DIR=idle)
    subprocess.run([sys.executable, os.path.join(HERE, "build.py"), sprites, tmp, name], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    skin = os.path.join(out, name)
    os.makedirs(skin, exist_ok=True)
    shutil.copy(os.path.join(tmp, "Mac", f"{name}.cursor"), os.path.join(skin, "theme.cursor"))
    shutil.copy(os.path.join(tmp, "Mac", "idle.plist"), os.path.join(skin, "idle.plist"))
    src = os.path.join(sprites, "pet.png")                  # 菜单栏图标：宠物本身；已有切图没有 pet.png 时用张开爪
    icon = Image.open(src if os.path.exists(src) else os.path.join(sprites, "14_张开爪.png")).convert("RGBA")
    icon = icon.crop(icon.getbbox()); s = max(icon.size)
    sq = Image.new("RGBA", (s, s)); sq.alpha_composite(icon, ((s - icon.width) // 2, (s - icon.height) // 2))
    sq.resize((36, 36), Image.LANCZOS).save(os.path.join(skin, "MenuIcon.png"))
    shutil.rmtree(tmp)
    return skin


USER_SKINS = os.path.expanduser("~/Library/Application Support/OrangeCatCursor/Skins")


# ---------- 主流程 ----------
def from_sheet(sheet, work):
    im, bg = remove_bg(Image.open(sheet))
    pets = split_cells(im)
    sc = PET_H / np.median([p.height for p in pets[:16]])
    sprites, idle = os.path.join(work, "切图"), os.path.join(work, "待机切图")
    os.makedirs(sprites, exist_ok=True); os.makedirs(idle, exist_ok=True)
    syms = symbols(); hs = {}
    for i, n in enumerate(NAMES):
        layer = place(pets[i], sc)
        if n == "05_忙碌":
            frames, hs[n] = busy_frames()
            for f, fr in enumerate(frames):
                on_top(layer, fr).save(f"{sprites}/{n}_f{f:02d}.png")
            on_top(layer, frames[0]).save(f"{sprites}/{n}.png")
            continue
        sym, hs[n] = syms[n]
        on_top(layer, sym).save(f"{sprites}/{n}.png")
    json.dump({k: [int(v[0]), int(v[1])] for k, v in hs.items()}, open(f"{sprites}/hotspots.json", "w"), ensure_ascii=False)
    place(pets[0], sc).save(f"{sprites}/pet.png")
    arrow = syms["01_正常"][0]
    for f, fr in enumerate(dance_frames(arrow, pets[16:19], sc)):
        fr.save(f"{idle}/跳舞_f{f:02d}.png")
    for f, fr in enumerate(sleep_frames(arrow, pets[19], sc)):
        fr.save(f"{idle}/睡觉_f{f:02d}.png")
    json.dump({"dance": 0.06, "sleep": 0.25}, open(f"{idle}/timing.json", "w"))
    print(f"背景色 {tuple(int(v) for v in bg)}，切出 20 个姿势")
    return sprites, idle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", nargs="?", help="4 列 5 行姿势总图")
    ap.add_argument("--name", required=True, help="宠物名字，如 团子（会显示在「切换宠物」菜单里）")
    ap.add_argument("--install", action="store_true", help="做完直接放进本机「宠物指针」的宠物文件夹")
    ap.add_argument("--out", help="输出目录（默认：总图所在文件夹）")
    ap.add_argument("--sprites"); ap.add_argument("--idle")
    a = ap.parse_args()
    out = os.path.abspath(a.out or os.path.dirname(os.path.abspath(a.sheet or ".")))
    os.makedirs(out, exist_ok=True)
    if a.sheet:                                   # 切图等中间文件放在总图旁边，宠物文件夹放 --out
        sprites, idle = from_sheet(a.sheet, os.path.dirname(os.path.abspath(a.sheet)))
    elif a.sprites and a.idle:
        sprites, idle = os.path.abspath(a.sprites), os.path.abspath(a.idle)
    else:
        sys.exit("要么给总图，要么给 --sprites 和 --idle")
    skin = make_skin_dir(sprites, idle, a.name, out)
    print("宠物文件夹：", skin)
    if a.install:
        dst = os.path.join(USER_SKINS, a.name)
        os.makedirs(USER_SKINS, exist_ok=True)
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(skin, dst)
        print(f"已放进宠物文件夹：{dst}\n点菜单栏的宠物小图标 → 切换宠物 → {a.name}")
