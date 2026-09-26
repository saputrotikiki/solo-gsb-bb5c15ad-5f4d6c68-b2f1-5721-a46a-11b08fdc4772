"""图片水平线校正端到端测试。

覆盖：
  - 角度计算：符号约定 / 归一化到 (-90, 90] / 点选顺序无关 / 水平与垂直线
  - 最大居中裁剪：全 1 掩膜（奇偶尺寸）/ 菱形掩膜暴力验证面积最大 /
    裁剪区全为有效像素且关于画布中心居中
  - 合成倾斜地平线：角度还原 / 校正后地平线水平 / 裁剪区无旋转填充
  - 原图自身透明像素不视为填充空白：同一图带透明区域与不透明版本
    裁剪范围完全一致，输出保留透明通道
  - 接口：200  happy path / 两点重合 400 / 越界 400 / 非数值 400 /
    非 PNG 400 / 空文件 400 / 超 16MB 413 / 缺字段 422
  - 校正结果可直接送入 /api/embed 水印嵌入表单
  - 全程内存：不落盘、不建任务

运行：
    python -m pytest tests/test_leveling.py -q
或：
    python tests/test_leveling.py
"""
import base64
import io
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import leveling  # noqa: E402
from app.leveling import (  # noqa: E402
    LevelingError,
    compute_angle_deg,
    largest_centered_crop,
    straighten_image,
)


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-leveling-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR
    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 构造工具

def _tilted_horizon_png(tilt_deg=7.0, size=(240, 320), seed=0, rgba_hole=False):
    """合成带倾斜地平线的 PNG：上方天空、下方地面，加噪点。

    返回 (png_bytes, horizon_y_fn)：horizon_y_fn(x) 给出地平线在该列的 y。
    """
    h, w = size
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.indices((h, w))
    tilt = math.radians(tilt_deg)

    def hy(x):
        return h / 2 + math.tan(tilt) * (x - w / 2)

    ground = yy > hy(xx)
    img[ground] = (40, 90, 160)
    img[~ground] = (160, 200, 235)
    img = img + rng.randint(0, 6, img.shape).astype(np.uint8)
    buf = io.BytesIO()
    if rgba_hole:
        rgba = np.dstack([img, np.full((h, w), 255, np.uint8)])
        hole = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(h, w) / 6) ** 2
        rgba[..., 3] = np.where(hole, 0, 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    else:
        Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue(), hy


def _decode_download(d):
    return base64.b64decode(d["download"]["data_url"].split(",", 1)[1])


def _nearest_coverage_mask(size, angle_deg):
    """最近邻重采样的覆盖掩膜：255 = 像素直接来自原图矩形（无填充）。"""
    w, h = size
    m = Image.new("L", (w, h), 255).rotate(
        angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    return np.asarray(m)


# ---------------------------------------------------------------- 角度计算

def test_angle_sign_and_normalization():
    # 屏幕坐标 y 向下：右端下沉 -> 正角（Pillow 正向 = 逆时针校正）
    a = compute_angle_deg((10, 10), (110, 20))
    assert abs(a - math.degrees(math.atan2(10, 100))) < 1e-9
    # 点选顺序不影响结果
    b = compute_angle_deg((110, 20), (10, 10))
    assert abs(a - b) < 1e-9
    # 水平线 -> 0
    assert compute_angle_deg((0, 50), (200, 50)) == 0.0
    # 垂直线 -> ±90（归一化到 90）
    assert abs(compute_angle_deg((50, 0), (50, 200)) - 90.0) < 1e-9
    assert abs(compute_angle_deg((50, 200), (50, 0)) - 90.0) < 1e-9
    # 右端升高 -> 负角（Pillow 负向 = 顺时针校正）
    assert abs(compute_angle_deg((0, 100), (100, 0)) + 45.0) < 1e-9
    # 归一化范围 (-90, 90]
    for dy in range(-10, 11):
        for dx in (-5, -1, 1, 5):
            ang = compute_angle_deg((0, 0), (dx, dy))
            assert -90.0 < ang <= 90.0


# ---------------------------------------------------------------- 最大居中裁剪

def test_crop_full_frame_when_no_fill():
    assert largest_centered_crop(np.ones((100, 200), np.uint8)) == (0, 0, 200, 100)
    assert largest_centered_crop(np.ones((101, 201), np.uint8)) == (0, 0, 201, 101)


def test_crop_diamond_mask_is_maximal_and_centered():
    yy, xx = np.indices((200, 200))
    mask = ((np.abs(xx - 99.5) + np.abs(yy - 99.5)) < 70).astype(np.uint8)
    left, top, right, bottom = largest_centered_crop(mask)
    # 裁剪区全为有效像素
    assert mask[top:bottom, left:right].all()
    # 居中（允许 1px 奇偶偏差）
    assert abs(left - (200 - right)) <= 1
    assert abs(top - (200 - bottom)) <= 1
    # 暴力枚举验证面积最大
    best = 0
    for b in range(1, 100):
        for a in range(1, 100):
            if mask[100 - b:100 + b, 100 - a:100 + a].all():
                best = max(best, (2 * a) * (2 * b))
    assert (right - left) * (bottom - top) == best


def test_crop_real_rotation_mask_matches_brute_force():
    """真实旋转掩膜上，模块结果与暴力枚举的最大居中全 1 矩形一致。"""
    size = (90, 120)  # h, w（奇偶混合）
    w, h = size[1], size[0]
    angle = 13.0
    mask_im = Image.new("L", (w, h), 255).rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
    mask = (np.asarray(mask_im) >= 254).astype(np.uint8)
    left, top, right, bottom = largest_centered_crop(mask)
    assert mask[top:bottom, left:right].all()
    mh, mw = mask.shape
    assert abs(left - (mw - right)) <= 1
    assert abs(top - (mh - bottom)) <= 1
    # 暴力：以画布中心为中心的所有轴对齐矩形
    cy2, cx2 = mh // 2, mw // 2
    hpad, wpad = mh % 2, mw % 2
    best = 0
    for b in range(0 if hpad else 1, cy2 + 1):
        for a in range(0 if wpad else 1, cx2 + 1):
            t, bb = cy2 - b, cy2 + b + hpad
            l, r = cx2 - a, cx2 + a + wpad
            if mask[t:bb, l:r].all():
                best = max(best, (r - l) * (bb - t))
    assert (right - left) * (bottom - top) == best


# ---------------------------------------------------------------- 完整校正流程

def test_straighten_recovers_angle_and_levels_horizon():
    tilt = 7.0
    png, hy = _tilted_horizon_png(tilt_deg=tilt)
    h, w = 240, 320
    p1 = (40.0, hy(40))
    p2 = (280.0, hy(280))
    res = straighten_image(png, p1, p2)
    # 角度还原（符号约定：右端下沉 -> 正角，逆时针校正）
    assert abs(res["angle_deg"] - tilt) < 0.05
    assert "逆时针" in res["angle_direction"]
    # 裁剪区居中于旋转画布
    c = res["crop"]
    rw, rh = res["rotated_size"]["width"], res["rotated_size"]["height"]
    assert abs(c["left"] - (rw - c["right"])) <= 1
    assert abs(c["top"] - (rh - c["bottom"])) <= 1
    # 输出 PNG 可解码、尺寸与裁剪范围一致
    out = Image.open(io.BytesIO(_decode_download(res)))
    out.load()
    assert out.format == "PNG"
    assert out.size == (c["width"], c["height"])
    arr = np.array(out.convert("RGB"))
    oh, ow = arr.shape[:2]

    # 校正后地平线水平：左右两侧天空/地面边界行差 ≤ 3px
    def boundary_row(x):
        col = arr[:, x, :].astype(int)
        sky = np.all(np.abs(col - np.array([160, 200, 235])) < 60, axis=1)
        return int(np.flatnonzero(~sky).min())

    assert abs(boundary_row(ow // 6) - boundary_row(5 * ow // 6)) <= 3


def test_crop_region_contains_no_rotation_fill():
    """裁剪区在最近邻覆盖掩膜（精确逐像素覆盖）下必须全为有效像素。"""
    tilt = 11.0
    png, hy = _tilted_horizon_png(tilt_deg=tilt, size=(180, 260))
    res = straighten_image(png, (30.0, hy(30)), (230.0, hy(230)))
    cover = _nearest_coverage_mask((260, 180), res["angle_deg"])
    c = res["crop"]
    region = cover[c["top"]:c["bottom"], c["left"]:c["right"]]
    assert region.size > 0
    assert (region == 255).all(), "裁剪区内不得含任何旋转填充像素"


def test_own_transparent_pixels_are_not_fill():
    """原图自身透明像素不视为填充空白：裁剪范围与不透明版本完全一致。"""
    tilt = 9.0
    png_opaque, hy = _tilted_horizon_png(tilt_deg=tilt, rgba_hole=False)
    png_hole, _ = _tilted_horizon_png(tilt_deg=tilt, rgba_hole=True)
    p1 = (40.0, hy(40))
    p2 = (280.0, hy(280))
    res_o = straighten_image(png_opaque, p1, p2)
    res_t = straighten_image(png_hole, p1, p2)
    assert res_t["has_alpha"] is True and res_t["mode"] == "RGBA"
    assert res_o["has_alpha"] is False and res_o["mode"] == "RGB"
    # 透明洞不影响裁剪范围（掩膜只描述原图矩形覆盖，与 alpha 无关）
    for key in ("left", "top", "right", "bottom", "width", "height"):
        assert res_t["crop"][key] == res_o["crop"][key]
    # 输出保留原图自身的透明像素
    out = np.array(Image.open(io.BytesIO(_decode_download(res_t))))
    assert out.shape[2] == 4
    assert int(out[..., 3].min()) == 0
    assert int(out[..., 3].max()) == 255


def test_zero_angle_keeps_full_frame():
    png, _ = _tilted_horizon_png(tilt_deg=0.0)
    res = straighten_image(png, (10.0, 120.0), (300.0, 120.0))
    assert res["angle_deg"] == 0.0
    c = res["crop"]
    assert (c["width"], c["height"]) == (320, 240)
    assert (c["left"], c["top"]) == (0, 0)


def test_point_order_does_not_change_result():
    png, hy = _tilted_horizon_png(tilt_deg=6.0)
    p1 = (40.0, hy(40))
    p2 = (280.0, hy(280))
    r1 = straighten_image(png, p1, p2)
    r2 = straighten_image(png, p2, p1)
    assert abs(r1["angle_deg"] - r2["angle_deg"]) < 1e-9
    assert r1["crop"] == r2["crop"]


# ---------------------------------------------------------------- 拒绝路径（单元）

def test_reject_coincident_points():
    png, hy = _tilted_horizon_png()
    with _expect_leveling_error("重合"):
        straighten_image(png, (40.0, hy(40)), (40.4, hy(40) + 0.3))


def test_reject_out_of_range_points():
    png, hy = _tilted_horizon_png()
    for p1, p2 in (
        ((-1.0, 10.0), (100.0, 20.0)),      # 左越界
        ((320.0, 10.0), (100.0, 20.0)),     # x == 宽度（合法最大为 319）
        ((10.0, 240.0), (100.0, 20.0)),     # y == 高度
        ((10.0, 10.0), (100.0, -0.5)),      # 上越界
    ):
        with _expect_leveling_error("超出图像范围"):
            straighten_image(png, p1, p2)


def test_reject_non_png_and_undecodable():
    img = np.random.RandomState(2).randint(0, 255, (80, 100, 3), np.uint8)
    jpg = io.BytesIO()
    Image.fromarray(img).save(jpg, format="JPEG")
    with _expect_leveling_error("PNG"):
        straighten_image(jpg.getvalue(), (5.0, 5.0), (90.0, 5.0))
    with _expect_leveling_error("无法解析"):
        straighten_image(b"not-an-image", (5.0, 5.0), (90.0, 5.0))


class _expect_leveling_error:
    def __init__(self, needle):
        self.needle = needle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is LevelingError, f"应抛 LevelingError，实际 {exc_type}"
        assert self.needle in str(exc), f"错误信息应包含 {self.needle!r}：{exc}"
        return True


# ---------------------------------------------------------------- 接口

def _post_leveling(client, png, p1=(30, 40), p2=(230, 60), name="a.png",
                   content_type="image/png"):
    return client.post(
        "/api/leveling",
        files={"image": (name, png, content_type)},
        data={"p1x": str(p1[0]), "p1y": str(p1[1]),
              "p2x": str(p2[0]), "p2y": str(p2[1])},
    )


def test_api_happy_path():
    client, data_dir = _fresh_client()
    png, hy = _tilted_horizon_png(tilt_deg=5.0)
    r = _post_leveling(client, png, p1=(40, hy(40)), p2=(280, hy(280)))
    assert r.status_code == 200, r.text
    d = r.json()
    assert abs(d["angle_deg"] - 5.0) < 0.05
    assert d["width"] == 320 and d["height"] == 240
    assert d["stored"] is False
    c = d["crop"]
    assert c["width"] == c["right"] - c["left"]
    assert c["height"] == c["bottom"] - c["top"]
    assert d["fits_watermark_input"] is True
    out = Image.open(io.BytesIO(_decode_download(d)))
    out.load()
    assert out.format == "PNG" and out.size == (c["width"], c["height"])
    # 全程内存：数据目录没有任何文件
    assert not any(data_dir.iterdir())


def test_api_rejections():
    client, _ = _fresh_client()
    png, hy = _tilted_horizon_png()

    # 两点重合
    r = _post_leveling(client, png, p1=(40, hy(40)), p2=(40.4, hy(40) + 0.3))
    assert r.status_code == 400 and "重合" in r.json()["detail"]

    # 越界
    r = _post_leveling(client, png, p1=(-1, 40), p2=(280, hy(280)))
    assert r.status_code == 400 and "超出图像范围" in r.json()["detail"]
    r = _post_leveling(client, png, p1=(40, 40), p2=(320, 60))
    assert r.status_code == 400 and "超出图像范围" in r.json()["detail"]

    # 非数值坐标
    r = client.post("/api/leveling",
                    files={"image": ("a.png", png, "image/png")},
                    data={"p1x": "abc", "p1y": "40", "p2x": "100", "p2y": "60"})
    assert r.status_code == 400 and "数值" in r.json()["detail"]

    # 非 PNG
    img = np.random.RandomState(4).randint(0, 255, (80, 100, 3), np.uint8)
    jpg = io.BytesIO()
    Image.fromarray(img).save(jpg, format="JPEG")
    r = _post_leveling(client, jpg.getvalue(), p1=(5, 5), p2=(90, 5),
                       name="a.jpg", content_type="image/jpeg")
    assert r.status_code == 400 and "PNG" in r.json()["detail"]

    # 空文件
    r = _post_leveling(client, b"", p1=(5, 5), p2=(90, 5))
    assert r.status_code == 400

    # 缺字段 -> 422
    r = client.post("/api/leveling",
                    files={"image": ("a.png", png, "image/png")},
                    data={"p1x": "10", "p1y": "40", "p2x": "100"})
    assert r.status_code == 422


def test_api_oversize_413():
    client, _ = _fresh_client()
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024 - 7)
    r = _post_leveling(client, oversized, p1=(5, 5), p2=(90, 5))
    assert r.status_code == 413


def test_leveled_output_feeds_embed_form():
    """校正结果（下载的 PNG）可直接送入现有 /api/embed 水印嵌入表单。"""
    client, _ = _fresh_client()
    png, hy = _tilted_horizon_png(tilt_deg=4.0, size=(360, 480))
    r = _post_leveling(client, png, p1=(60, hy(60)), p2=(420, hy(420)))
    assert r.status_code == 200, r.text
    leveled = _decode_download(r.json())
    r2 = client.post(
        "/api/embed",
        files={"image": ("leveled.png", leveled, "image/png")},
        data={"text": "水平线校正-2026", "strength": "medium"},
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["job_id"]
    assert d2["width"] == r.json()["crop"]["width"]
    assert d2["height"] == r.json()["crop"]["height"]


if __name__ == "__main__":
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
