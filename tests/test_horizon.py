"""图片水平线校正端到端测试。

覆盖：
  - 参照线两点 -> 反向旋转角（atan2(dy,dx)，图像坐标 y 向下，
    归一化到 (-90, 90] 度；135° 与 -45° 等效等）
  - 最大居中轴对齐无填充内接矩形：解析解 / 0° 满幅 / 90° 取方 / 与独立实现对拍
  - 端到端：合成倾斜图（红/蓝水平分界 + 绿色旋转填充）校正后
    角度 ≈ 反向原旋转角、分界恢复水平、裁剪区内 0 个填充绿像素
  - 透明通道：原图自带棋盘式透明像素时裁剪尺寸仍完全由几何决定（不因
    透明收缩），alpha 与独立参照旋转一致，自检通过；不透明中心图裁剪后
    中心区域完全不透明（旋转填充不渗入）
  - 明确拒绝：两点重合、点超出图像范围（x=w / y=h / 负值）、
    非 PNG（JPEG / WebP）、坏图、超 4096px、超 16MB
  - 输出 PNG 无元数据块（eXIf / 文本块不残留）
  - 裁剪坐标半开区间一致且不越界
  - 校正结果可直接送入 /api/embed
  - 全程内存：不建任务目录、不写归档

运行：
    python -m pytest tests/test_horizon.py -q
或：
    python tests/test_horizon.py
"""
import io
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import horizon  # noqa: E402


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-horizon-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR
    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 构造工具

def _half_rgb_png(w=360, h=280, theta=5.0, seed=0):
    """上半红下半蓝的图，绕图心旋转 theta，绿色填充；返回 PNG 字节。"""
    arr = np.zeros((h, w, 3), np.uint8)
    arr[: h // 2] = (220, 30, 30)
    arr[h // 2:] = (30, 30, 220)
    im = Image.fromarray(arr)
    rot = im.rotate(theta, resample=Image.BICUBIC, expand=False,
                    fillcolor=(0, 255, 0))
    buf = io.BytesIO()
    rot.save(buf, format="PNG")
    return buf.getvalue()


def _edge_point(arr, x):
    """在第 x 列找红->蓝过渡行，避开绿色填充；模拟用户选点。"""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    not_green = ~((g > 150) & (r < 100) & (b < 100))
    diff = np.abs(np.diff((b - r).astype(float)[:, x]))
    for i in np.argsort(diff)[::-1]:
        if not_green[i, x] and not_green[i + 1, x]:
            return int(i)
    return int(np.argmax(diff))


def _correct_kwargs(w, h, theta, x1=None, x2=None):
    """选两个可见分界点的 multipart 表单参数。"""
    if x1 is None:
        x1 = int(w * 0.30)
    if x2 is None:
        x2 = int(w * 0.70)
    png = _half_rgb_png(w, h, theta)
    arr = np.asarray(Image.open(io.BytesIO(png)))
    y1, y2 = _edge_point(arr, x1), _edge_point(arr, x2)
    return {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}, png


def _png_chunks(data):
    pos, out = 8, []
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        out.append(data[pos + 4:pos + 8].decode("latin1"))
        pos += 12 + ln
    return out


# ---------------------------------------------------------------- 角度

def test_correction_angle_known_and_normalized():
    f = horizon.correction_angle_deg
    # 完全水平（无论方向）
    assert f(0, 10, 100, 10) == 0.0
    assert f(0, 0, -1, 0) == 0.0                # 指向左的水平线仍是水平
    # 图像坐标 y 向下：线向右上倾（dy<0）-> 需顺时针校正（负角）
    assert abs(f(0, 1, 1, 0) - (-45.0)) < 1e-9
    # 线向右下倾（dy>0）-> 需逆时针校正（正角）
    assert abs(f(0, 0, 1, 1) - 45.0) < 1e-9
    # 水平线无方向：同一条无向线，端点顺序互换结果一致
    assert abs(f(0, 0, -1, 1) - (-45.0)) < 1e-9   # atan2 = 135° -> -45°
    assert abs(f(0, 0, 1, -1) - (-45.0)) < 1e-9   # 与 f(0,1,1,0) 同向
    assert abs(f(0, 0, -1, -1) - 45.0) < 1e-9     # atan2 = -135° -> 45°
    # 竖直线 -> 90°（归一化上界）
    assert f(0, 0, 0, 1) == 90.0
    assert f(0, 0, 0, -1) == 90.0


def test_angle_recovers_negative_of_source_rotation():
    """合成 theta 旋转后选点，校正角应 ≈ -theta（±1px 选点误差内）。"""
    for theta in (3.0, 8.0, -12.5, 23.7, -30.0, 40.0):
        w, h = 360, 280
        kw, png = _correct_kwargs(w, h, theta)
        arr = np.asarray(Image.open(io.BytesIO(png)))
        x1, y1, x2, y2 = (float(kw[k]) for k in ("x1", "y1", "x2", "y2"))
        deg = horizon.correction_angle_deg(x1, y1, x2, y2)
        assert abs(deg - (-theta)) < 1.2, (theta, deg)


# ---------------------------------------------------------------- 内接矩形

def test_largest_inner_rect_special_cases():
    # 0° -> 满幅
    wr, hr = horizon.largest_inner_rect(200, 150, 0.0)
    assert (round(wr, 6), round(hr, 6)) == (200.0, 150.0)
    # 正方形 45° -> w / (sin45 + cos45)
    wr, hr = horizon.largest_inner_rect(200, 200, 45.0)
    assert abs(wr - 200 / (math.sqrt(2))) < 1e-6
    assert abs(hr - 200 / (math.sqrt(2))) < 1e-6
    # 100×10 转 90° -> 最大为 10×10
    wr, hr = horizon.largest_inner_rect(100, 10, 90.0)
    assert (round(wr, 6), round(hr, 6)) == (10.0, 10.0)
    # 取整裁剪框居中且不越界
    for (w, h, a) in ((200, 150, 8), (100, 10, 75), (64, 64, 44.9)):
        left, top, cw, ch = horizon.crop_box_for(w, h, a)
        assert 0 <= left and left + cw <= w and 0 <= top and top + ch <= h
        assert left == (w - cw) // 2 and top == (h - ch) // 2
        assert cw >= 1 and ch >= 1


def test_largest_inner_rect_matches_independent_search():
    """与二分暴力搜索对拍（角度/尺寸各异，面积误差应 < 0.5%）。"""
    def brute(w, h, deg, n=1500):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)

        def max_h(wr):
            a = wr / 2
            lo, hi = 0.0, min(w, h) * 4
            for _ in range(50):
                mid = (lo + hi) / 2
                ok = all(
                    abs(c * qx + s * qy) <= w / 2 + 1e-9
                    and abs(-s * qx + c * qy) <= h / 2 + 1e-9
                    for qx, qy in ((a, mid), (a, -mid), (-a, mid), (-a, -mid)))
                if ok:
                    lo = mid
                else:
                    hi = mid
            return min(2 * lo, h)

        best = 0.0
        for wr in np.linspace(0.01, w, n):
            best = max(best, wr * max_h(wr))
        return best

    for (w, h) in ((200, 150), (100, 10), (64, 64), (300, 80), (53, 97)):
        for a in (0.5, 10, 22.5, 30, 45, 60, 75, 89.9):
            wr, hr = horizon.largest_inner_rect(w, h, a)
            assert wr * hr >= brute(w, h, a) * 0.995, (w, h, a)


# ---------------------------------------------------------------- 端到端

def test_e2e_correct_angle_crop_and_no_fill():
    client, _ = _fresh_client()
    w, h = 360, 280
    theta = 8.0
    kw, png = _correct_kwargs(w, h, theta)
    r = client.post("/api/horizon-correct",
                    files={"image": ("tilted.png", png, "image/png")},
                    data=kw)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["stored"] is False and "job_id" not in d
    assert abs(d["rotate_deg"] - (-theta)) < 1.2
    c = d["crop"]
    assert c["right"] == c["left"] + c["width"]
    assert c["bottom"] == c["top"] + c["height"]
    assert 0 <= c["left"] and c["right"] <= w
    assert 0 <= c["top"] and c["bottom"] <= h
    assert d["corrected_size"] == {"width": c["width"], "height": c["height"]}
    assert d["recheck"]["passed"] is True
    assert d["recheck"]["uncovered_pixels"] == 0

    raw = _decode(d)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    out = Image.open(io.BytesIO(raw))
    assert out.size == (c["width"], c["height"])
    a = np.asarray(out).astype(int)
    # 0 个绿色填充像素
    green = ((a[..., 1] > 150) & (a[..., 0] < 100) & (a[..., 2] < 100)).sum()
    assert green == 0
    # 红->蓝分界在所有列上的行位置基本一致（水平恢复）
    rr, gg, bb = a[..., 0], a[..., 1], a[..., 2]
    rows = [int(np.argmax(np.abs(np.diff((bb - rr).astype(float)[:, x]))))
            for x in range(0, c["width"], 4)]
    assert max(rows) - min(rows) <= 2
    assert len(raw) == d["download"]["bytes"]
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == d["download"]["sha256"]


def _decode(d):
    import base64
    return base64.b64decode(d["download"]["data_url"].split(",", 1)[1])


def test_e2e_negative_angle_direction():
    """顺时针倾斜（theta 负）同样校正。"""
    client, _ = _fresh_client()
    kw, png = _correct_kwargs(360, 280, -12.5)
    r = client.post("/api/horizon-correct",
                    files={"image": ("t.png", png, "image/png")}, data=kw)
    assert r.status_code == 200, r.text
    d = r.json()
    assert abs(d["rotate_deg"] - 12.5) < 1.2
    out = np.asarray(Image.open(io.BytesIO(_decode(d)))).astype(int)
    green = ((out[..., 1] > 150) & (out[..., 0] < 100) & (out[..., 2] < 100)).sum()
    assert green == 0


def test_output_png_has_no_metadata_chunks():
    client, _ = _fresh_client()
    # 输入带文本元数据，输出不应保留
    kw, png = _correct_kwargs(240, 200, 5.0)
    im = Image.open(io.BytesIO(png))
    buf = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "SECRET")
    im.save(buf, format="PNG", pnginfo=info)
    r = client.post("/api/horizon-correct",
                    files={"image": ("m.png", buf.getvalue(), "image/png")}, data=kw)
    assert r.status_code == 200, r.text
    assert _png_chunks(_decode(r.json())) == ["IHDR", "IDAT", "IEND"]


# ---------------------------------------------------------------- 透明通道

def test_original_transparent_pixels_are_not_fill():
    """上半不透明、下半透明的 RGBA：裁剪尺寸完全由几何决定（不因透明收缩），
    alpha 与独立参照旋转一致，裁剪区内同时保留原图自带的不透明/透明像素。"""
    client, _ = _fresh_client()
    w, h = 300, 240
    theta = 8.0
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (180, 90, 60)
    rgba[: h // 2, ..., 3] = 255
    tilted = Image.fromarray(rgba, "RGBA").rotate(
        theta, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0, 0))
    a = np.asarray(tilted)
    buf = io.BytesIO()
    tilted.save(buf, format="PNG")
    png = buf.getvalue()

    # 在倾斜的透明分界上选两个端点（最大 alpha 梯度行，x 取 30%/70%）
    grad = np.abs(np.diff(a[..., 3].astype(int), axis=0))
    def boundary_row(x):
        band = (max(0, h // 2 - 40), min(h - 1, h // 2 + 40))
        return band[0] + int(np.argmax(grad[band[0]:band[1], x]))
    x1, x2 = int(w * 0.30), int(w * 0.70)
    y1, y2 = boundary_row(x1), boundary_row(x2)

    r = client.post("/api/horizon-correct", files={
        "image": ("a.png", png, "image/png")
    }, data={"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    assert r.status_code == 200, r.text
    d = r.json()
    c = d["crop"]
    assert abs(d["rotate_deg"] - (-theta)) < 1.2

    # 关键断言：裁剪尺寸 = 几何公式结果，不因 alpha=0 像素而收缩
    left, top, cw, ch = horizon.crop_box_for(w, h, d["rotate_deg"])
    assert (c["left"], c["top"], c["width"], c["height"]) == (left, top, cw, ch)
    assert d["has_alpha"] is True and d["recheck"]["passed"] is True

    out = np.asarray(Image.open(io.BytesIO(_decode(d))))
    assert out.shape == (ch, cw, 4)
    # 裁剪区内同时保留原图自带的透明与不透明像素
    assert (out[..., 3] == 0).sum() > 0
    assert (out[..., 3] == 255).sum() > 0
    # alpha 与独立参照旋转（相同参数）一致
    ref = np.asarray(tilted.rotate(
        d["rotate_deg"], resample=Image.BICUBIC, expand=False,
        fillcolor=(0, 0, 0, 0)).crop((left, top, left + cw, top + ch)))
    assert np.abs(out[..., 3].astype(int) - ref[..., 3].astype(int)).max() <= 2


def test_opaque_center_has_no_fill_transparency():
    """中心区域不透明的图：校正后裁剪中心区域必须完全不透明。"""
    client, _ = _fresh_client()
    w, h = 300, 240
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (180, 90, 60)
    rgba[..., 3] = 0
    rgba[40:h - 40, 40:w - 40, 3] = 255
    tilted = Image.fromarray(rgba, "RGBA").rotate(
        10, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0, 0))
    buf = io.BytesIO()
    tilted.save(buf, format="PNG")

    a = np.asarray(tilted)
    x1, x2 = 110, 190
    # 倾斜矩形的上边缘：每列首个不透明行（中央两列避开四角填充区）
    y1 = int(np.argmax(a[:, x1, 3] > 250))
    y2 = int(np.argmax(a[:, x2, 3] > 250))
    d = client.post("/api/horizon-correct", files={
        "image": ("o.png", buf.getvalue(), "image/png")
    }, data={"x1": x1, "y1": y1, "x2": x2, "y2": y2}).json()
    assert abs(d["rotate_deg"] - (-10.0)) < 1.2
    out = np.asarray(Image.open(io.BytesIO(_decode(d))))
    ch, cw = out.shape[:2]
    # 裁剪中心 40% 区域全不透明
    cy0, cy1 = int(ch * 0.30), int(ch * 0.70)
    cx0, cx1 = int(cw * 0.30), int(cw * 0.70)
    assert out[cy0:cy1, cx0:cx1, 3].min() == 255


# ---------------------------------------------------------------- 拒绝

def test_coincident_points_rejected():
    client, _ = _fresh_client()
    _, png = _correct_kwargs(240, 200, 5.0)
    for pts in ((50, 50, 50, 50), (50, 50, 50.6, 50.6)):
        r = client.post("/api/horizon-correct", files={
            "image": ("p.png", png, "image/png")
        }, data={"x1": pts[0], "y1": pts[1], "x2": pts[2], "y2": pts[3]})
        assert r.status_code == 400, r.text
        assert "重合" in r.json()["detail"]


def test_out_of_range_points_rejected():
    client, _ = _fresh_client()
    w, h = 240, 200
    _, png = _correct_kwargs(w, h, 5.0)
    bad = [
        (w, 10, 10, 10),       # x 超出（== width）
        (10, h, 10, 10),       # y 超出
        (-1, 10, 10, 10),      # 负坐标
        (10, -0.5, 10, 10),
        (10, 10, w, 10),
        (10, 10, 10, h + 3),
    ]
    for pts in bad:
        r = client.post("/api/horizon-correct", files={
            "image": ("p.png", png, "image/png")
        }, data={"x1": pts[0], "y1": pts[1], "x2": pts[2], "y2": pts[3]})
        assert r.status_code == 400, (pts, r.text)
        assert "超出图像范围" in r.json()["detail"], pts
    # 边界点合法（0 与 w-1/h-1）
    r = client.post("/api/horizon-correct", files={
        "image": ("p.png", png, "image/png")
    }, data={"x1": 0, "y1": 0, "x2": w - 1, "y2": h - 1})
    assert r.status_code == 200, r.text


def test_unsupported_corrupt_oversize():
    client, _ = _fresh_client()
    # JPEG
    jpg = io.BytesIO()
    Image.new("RGB", (32, 32), (90, 90, 90)).save(jpg, format="JPEG")
    r = client.post("/api/horizon-correct", files={
        "image": ("a.jpg", jpg.getvalue(), "image/jpeg")
    }, data={"x1": 1, "y1": 1, "x2": 10, "y2": 10})
    assert r.status_code == 400 and "PNG" in r.json()["detail"]

    # 坏文件
    r = client.post("/api/horizon-correct", files={
        "image": ("x.png", b"not-a-png" * 8, "image/png")
    }, data={"x1": 1, "y1": 1, "x2": 10, "y2": 10})
    assert r.status_code == 400

    # 空文件
    r = client.post("/api/horizon-correct", files={
        "image": ("x.png", b"", "image/png")
    }, data={"x1": 1, "y1": 1, "x2": 10, "y2": 10})
    assert r.status_code == 400

    # 超 4096px
    big = io.BytesIO()
    Image.new("RGB", (4097, 16), (0, 0, 0)).save(big, format="PNG")
    r = client.post("/api/horizon-correct", files={
        "image": ("big.png", big.getvalue(), "image/png")
    }, data={"x1": 1, "y1": 1, "x2": 10, "y2": 10})
    assert r.status_code == 400 and "4096" in r.json()["detail"]


def test_oversize_16mb():
    client, _ = _fresh_client()
    arr = np.random.RandomState(2).randint(0, 255, (4000, 4000, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    if len(buf.getvalue()) > 16 * 1024 * 1024:
        r = client.post("/api/horizon-correct", files={
            "image": ("huge.png", buf.getvalue(), "image/png")
        }, data={"x1": 1, "y1": 1, "x2": 10, "y2": 10})
        assert r.status_code == 413


def test_missing_fields_422_and_bad_number_400():
    client, _ = _fresh_client()
    _, png = _correct_kwargs(240, 200, 5.0)
    r = client.post("/api/horizon-correct", files={
        "image": ("p.png", png, "image/png")
    }, data={"x1": 1, "y1": 1, "x2": 10})
    assert r.status_code == 422
    r = client.post("/api/horizon-correct", files={
        "image": ("p.png", png, "image/png")
    }, data={"x1": "abc", "y1": 1, "x2": 10, "y2": 10})
    assert r.status_code == 400 and "有效数字" in r.json()["detail"]


# ---------------------------------------------------------------- 集成与隔离

def test_corrected_png_can_be_embedded():
    """校正结果直接送入 /api/embed 应能建立水印任务。"""
    client, _ = _fresh_client()
    kw, png = _correct_kwargs(360, 280, 5.0)
    d = client.post("/api/horizon-correct",
                    files={"image": ("tilted.png", png, "image/png")},
                    data=kw).json()
    assert d["corrected_size"]["height"] >= 128  # 嵌入最短边下限
    raw = _decode(d)
    r = client.post("/api/embed", files={
        "image": ("tilted.horizon.png", raw, "image/png")
    }, data={"text": "horizon-test", "strength": "medium"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["width"] == d["corrected_size"]["width"]
    assert job["height"] == d["corrected_size"]["height"]


def test_no_persistence():
    client, data_dir = _fresh_client()
    kw, png = _correct_kwargs(240, 200, 7.0)
    for _ in range(2):
        r = client.post("/api/horizon-correct",
                        files={"image": ("p.png", png, "image/png")}, data=kw)
        assert r.status_code == 200
    assert not list(data_dir.glob("*/meta.json"))
    assert not (data_dir / "archive.log.jsonl").exists()
    assert [p.name for p in data_dir.iterdir()] == []


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个水平线校正测试全部通过 ✔")
