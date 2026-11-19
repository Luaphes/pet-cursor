"""把切图目录里 256px 透明 PNG 打包成 Windows 指针方案（.cur/.ani + install.inf）和 Mac 主题（MaCursor 原生 .cursor）。

用法：MAC_POINTS=40 python3 build.py <切图目录> <输出目录> [主题名]
切图目录要有 hotspots.json（文件名 -> 256 画布上的点击点）；忙碌动画帧命名为 05_忙碌_f00.png …
"""
import glob, io, json, os, plistlib, struct, sys, uuid
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else "橘猫指针"
HOTSPOTS = {k: tuple(v) for k, v in json.load(open(os.path.join(SRC, "hotspots.json"), encoding="utf-8")).items()}
WIN_SIZES = [32, 48, 64, 96, 128]
MAC_POINTS = int(os.environ.get("MAC_POINTS", 40))
MAC_SCALES = [1, 2, 4]
FRAME_SEC = 0.08                       # 动画每帧时长

# Mac 系统指针标识 -> 图（标识取自 MaCursor 源码 MACCursorDefinitions.swift）
MAC_MAP = {
    "com.apple.coregraphics.Arrow": "01_正常", "com.apple.coregraphics.ArrowS": "01_正常",
    "com.apple.coregraphics.ArrowCtx": "01_正常",
    "com.apple.coregraphics.IBeam": "02_文字输入", "com.apple.coregraphics.IBeamS": "02_文字输入",
    "com.apple.cursor.26": "02_文字输入",
    "com.apple.cursor.13": "03_链接",
    "com.apple.cursor.3": "04_不可用",
    "com.apple.coregraphics.Wait": "05_忙碌", "com.apple.cursor.4": "05_忙碌",
    "com.apple.coregraphics.Move": "06_移动",
    **{f"com.apple.cursor.{i}": "07_左右调整" for i in (17, 18, 19, 27, 28, 38)},
    **{f"com.apple.cursor.{i}": "08_上下调整" for i in (21, 22, 23, 31, 32, 36)},
    **{f"com.apple.cursor.{i}": "09_斜向调整_左上右下" for i in (33, 34, 35)},
    **{f"com.apple.cursor.{i}": "10_斜向调整_右上左下" for i in (29, 30, 37)},
    "com.apple.cursor.40": "11_帮助",
    "com.apple.cursor.7": "12_十字瞄准", "com.apple.cursor.8": "12_十字瞄准",
    "com.apple.coregraphics.Copy": "13_复制", "com.apple.cursor.5": "13_复制",
    "com.apple.cursor.12": "14_张开爪", "com.apple.cursor.11": "15_抓紧爪",
    "com.apple.cursor.42": "16_放大",
}

# Windows 方案字段顺序（与开源包 capitaine-cursors 的 install.inf 一致），None = 用系统默认
WIN_ORDER = [
    ("pointer", "01_正常"), ("help", "11_帮助"), ("work", "05_忙碌"), ("busy", "05_忙碌"),
    ("cross", "12_十字瞄准"), ("text", "02_文字输入"), ("hand", None), ("unavailable", "04_不可用"),
    ("vert", "08_上下调整"), ("horz", "07_左右调整"), ("dgn1", "09_斜向调整_左上右下"),
    ("dgn2", "10_斜向调整_右上左下"), ("move", "06_移动"), ("alternate", "01_正常"), ("link", "03_链接"),
]
WIN_FILE = {"01_正常": "normal.cur", "02_文字输入": "text.cur", "03_链接": "link.cur", "04_不可用": "unavailable.cur",
            "05_忙碌": "busy.ani", "06_移动": "move.cur", "07_左右调整": "horz.cur", "08_上下调整": "vert.cur",
            "09_斜向调整_左上右下": "dgn1.cur", "10_斜向调整_右上左下": "dgn2.cur", "11_帮助": "help.cur",
            "12_十字瞄准": "cross.cur"}


def frames(stem):
    fs = sorted(glob.glob(os.path.join(SRC, f"{stem}_f[0-9][0-9].png")))
    ims = [Image.open(f).convert("RGBA") for f in (fs or [os.path.join(SRC, stem + ".png")])]
    for im in ims:
        assert im.size == (256, 256), (stem, im.size)
    return ims


def cur_bytes(im, hx, hy):
    """多尺寸 .cur：每个尺寸存 32 位 BGRA 位图 + 1 位 AND 掩码。"""
    blobs, entries = [], []
    for s in WIN_SIZES:
        px = im.resize((s, s), Image.LANCZOS).load()
        xor = bytearray()
        for y in range(s - 1, -1, -1):
            for x in range(s):
                r, g, b, a = px[x, y]
                xor += bytes((b, g, r, a))
        row = ((s + 31) // 32) * 4
        mask = bytearray()
        for y in range(s - 1, -1, -1):
            bits = bytearray(row)
            for x in range(s):
                if px[x, y][3] == 0:
                    bits[x // 8] |= 0x80 >> (x % 8)
            mask += bits
        hdr = struct.pack("<IiiHHIIiiII", 40, s, s * 2, 1, 32, 0, len(xor) + len(mask), 0, 0, 0, 0)
        blobs.append(hdr + xor + mask)
        entries.append((s, round(hx * s / 256), round(hy * s / 256)))
    out = struct.pack("<HHH", 0, 2, len(blobs))
    off = 6 + 16 * len(blobs)
    for (s, x, y), blob in zip(entries, blobs):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, x, y, len(blob), off)
        off += len(blob)
    return out + b"".join(blobs)


def chunk(cid, data):
    return cid + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def ani_bytes(ims, hx, hy):
    """RIFF/ACON 动画指针：anih 头 + 每帧一个完整 .cur。"""
    jiffies = max(1, round(FRAME_SEC * 60))
    anih = struct.pack("<IIIIIIIII", 36, len(ims), len(ims), 0, 0, 0, 0, jiffies, 1)
    fram = b"fram" + b"".join(chunk(b"icon", cur_bytes(im, hx, hy)) for im in ims)
    body = b"ACON" + chunk(b"anih", anih) + chunk(b"LIST", fram)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def build_windows():
    d = os.path.join(OUT, "Windows")
    os.makedirs(d, exist_ok=True)
    for stem, fn in WIN_FILE.items():
        ims = frames(stem)
        data = ani_bytes(ims, *HOTSPOTS[stem]) if fn.endswith(".ani") else cur_bytes(ims[0], *HOTSPOTS[stem])
        with open(os.path.join(d, fn), "wb") as f:
            f.write(data)
    paths = ",".join("" if stem is None else f"%10%\\%CUR_DIR%\\%{key}%" for key, stem in WIN_ORDER)
    strings = "\n".join(f'{key:<12}= "{WIN_FILE[stem]}"' for key, stem in WIN_ORDER if stem)
    inf = f"""; {NAME}
; 右键 install.inf 选「安装」，然后打开 设置 > 蓝牙和其他设备 > 鼠标 > 其他鼠标设置 > 指针，方案里选「{NAME}」点应用。

[Version]
signature="$CHICAGO$"

[DefaultInstall]
CopyFiles = Scheme.Cur
AddReg    = Scheme.Reg

[DestinationDirs]
Scheme.Cur = 10,"%CUR_DIR%"

[Scheme.Reg]
HKCU,"Control Panel\\Cursors\\Schemes","%SCHEME_NAME%",,"{paths}"

[Scheme.Cur]
{chr(10).join(WIN_FILE.values())}

[Strings]
CUR_DIR     = "Cursors\\OrangeCat"
SCHEME_NAME = "{NAME}"
{strings}
"""
    with open(os.path.join(d, "install.inf"), "w", encoding="utf-16") as f:   # UTF-16 中文方案名才不乱码
        f.write(inf.replace("\n", "\r\n"))


def sheet_png(ims, s):
    """Mac 多帧：同一尺寸的各帧竖着拼成一张。"""
    sheet = Image.new("RGBA", (s, s * len(ims)))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((s, s), Image.LANCZOS), (0, i * s))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def build_mac():
    cursors = {}
    for ident, stem in MAC_MAP.items():
        ims = frames(stem)
        hx, hy = HOTSPOTS[stem]
        cursors[ident] = {
            "FrameCount": len(ims), "FrameDuration": FRAME_SEC if len(ims) > 1 else 0.0,
            "HotSpotX": float(hx * MAC_POINTS / 256), "HotSpotY": float(hy * MAC_POINTS / 256),
            "PointsWide": float(MAC_POINTS), "PointsHigh": float(MAC_POINTS),
            "Representations": [sheet_png(ims, MAC_POINTS * k) for k in MAC_SCALES],
        }
    native = {   # 键名照 MaCursor 生成的 SystemDefault.cursor
        "ThemeName": NAME, "Creator": "Tink", "HiDPI": True, "ThemeVersion": 1.0,
        "UUID": str(uuid.uuid5(uuid.NAMESPACE_DNS, "orangecat." + NAME)).upper(),
        "Identifier": "local.petcursor", "Cursors": cursors,
    }
    os.makedirs(os.path.join(OUT, "Mac"), exist_ok=True)
    with open(os.path.join(OUT, "Mac", f"{NAME}.cursor"), "wb") as f:
        plistlib.dump(native, f, fmt=plistlib.FMT_XML)


def build_idle():
    """待机动画（只给 Mac App 用）：IDLE_DIR 里的 跳舞_fNN / 睡觉_fNN，点击点同「正常」。"""
    idle_dir = os.environ.get("IDLE_DIR")
    if not idle_dir:
        return
    hx, hy = HOTSPOTS["01_正常"]
    out = {}
    tf = os.path.join(idle_dir, "timing.json")          # make_idle.py 写的每帧时长
    timing = json.load(open(tf)) if os.path.exists(tf) else {"dance": 0.15, "sleep": 0.25}
    for key, prefix in [("dance", "跳舞"), ("sleep", "睡觉")]:
        dur = timing[key]
        ims = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(os.path.join(idle_dir, f"{prefix}_f[0-9][0-9].png")))]
        out[key] = {
            "FrameCount": len(ims), "FrameDuration": dur,
            "HotSpotX": float(hx * MAC_POINTS / 256), "HotSpotY": float(hy * MAC_POINTS / 256),
            "PointsWide": float(MAC_POINTS), "PointsHigh": float(MAC_POINTS),
            "Representations": [sheet_png(ims, MAC_POINTS * k) for k in MAC_SCALES],
        }
    with open(os.path.join(OUT, "Mac", "idle.plist"), "wb") as f:
        plistlib.dump(out, f, fmt=plistlib.FMT_BINARY)


build_windows()
build_mac()
build_idle()
print("done", OUT)
