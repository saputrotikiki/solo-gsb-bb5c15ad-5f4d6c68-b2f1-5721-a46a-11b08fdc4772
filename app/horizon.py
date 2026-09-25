"""图片水平线校正 —— 两点选线、反向旋转、最大无填充居中裁剪（全内存）。

流程（全程仅在内存中进行，不落盘、不创建任务）：

  1. 仅接受 PNG，严格解码（损坏 / 非 PNG 直接拒绝）；
  2. 校验用户在同一条**水平参照线**上点选的两个端点：
     任一点超出图像范围、或两点重合（间距不足 1px）都明确拒绝；
  3. 按**原图像素坐标**求反向旋转角：
     ``deg = atan2(y2 - y1, x2 - x1)``（图像坐标系 y 向下，角度为
     视觉逆时针、与 Pillow ``Image.rotate`` 同号），并归一化到
     (-90°, 90°]——水平线没有方向，旋转 θ 与 θ±180° 等效，取最小旋转量；
  4. 以双三次重采样绕图心旋转（画布与原图同尺寸），旋转填充区
     （RGB 为黑色、带透明通道时为全透明）只出现在原图矩形旋转后
     覆盖范围之外；
  5. 根据**原图矩形边界**精确求解旋转后画布内**最大居中、轴对齐、
     且完全不含旋转填充**的裁剪矩形（解析解，见
     :func:`largest_inner_rect`），取整后居中裁剪；
  6. 用「哨兵覆盖自检」复核裁剪区：把与原图同尺寸的全白哨兵图做
     完全相同参数的旋转，凡被旋转覆盖的位置均为白色、填充为黑色，
     裁剪区内必须逐像素被覆盖，否则抛 :class:`HorizonRecheckError`
     （调用方返回错误且**不提供下载**）——裁剪区完全由几何计算决定，
     **从不依据像素内容收缩，因此原图自身的透明像素不会被当成
     旋转填充空白**；
  7. 像素经 ``numpy`` 数组 + ``Image.fromarray`` 重建为全新图像对象
     后以 PNG 重编码（不带入任何元数据），随响应内联返回。

与水印任务、攻击记录、报告归档完全隔离；校正结果可直接作为
``/api/embed`` 的上传图片继续使用。
"""

from __future__ import annotations

import base64
import hashlib
import io
import math

import numpy as np
from PIL import Image

from .privacy_export import normalize_mode

# 源图最长边上限（与平台其它图片接口一致的 4096px 口径）
MAX_SOURCE_DIM = 4096

# 两点间距低于该值视为重合（参照线方向无法定义）
MIN_POINT_DISTANCE_PX = 1.0


class HorizonError(ValueError):
    """校正前置条件不满足（点越界 / 两点重合 / 非 PNG / 尺寸超限等）。"""


class HorizonRecheckError(RuntimeError):
    """裁剪区哨兵覆盖自检未通过：不提供下载。"""


# ---------------------------------------------------------------- 打开与校验

def open_png_image(data: bytes) -> Image.Image:
    """严格打开 PNG 图片；无法解码或非 PNG -> HorizonError。"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise HorizonError("无法解析上传的图片。") from e
    if im.format != "PNG":
        raise HorizonError(
            f"水平线校正仅支持 PNG 图片（当前为 {im.format or '未知格式'}）。")
    return im


def _as_finite_float(value, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise HorizonError(f"参照点坐标 {name} 不是有效数字。") from None
    if not math.isfinite(f):
        raise HorizonError(f"参照点坐标 {name} 不是有限数值。")
    return f


def validate_points(
    x1, y1, x2, y2, width: int, height: int,
) -> tuple[float, float, float, float]:
    """校验两个参照端点，返回四个 float 坐标。

    - 坐标必须有限；
    - 每个点都必须落在图像范围内（0 <= x <= width-1，0 <= y <= height-1）；
    - 两点间距不得小于 MIN_POINT_DISTANCE_PX（重合 / 过近无法定义方向）。
    """
    pts = (
        (_as_finite_float(x1, "x1"), _as_finite_float(y1, "y1")),
        (_as_finite_float(x2, "x2"), _as_finite_float(y2, "y2")),
    )
    for i, (px, py) in enumerate(pts, start=1):
        if not (0.0 <= px <= width - 1 and 0.0 <= py <= height - 1):
            raise HorizonError(
                f"参照点 {i} ({px:g}, {py:g}) 超出图像范围"
                f"（图像宽 {width}px、高 {height}px，"
                f"x 需在 0~{width - 1}、y 需在 0~{height - 1} 之间）。")
    (ax, ay), (bx, by) = pts
    dist = math.hypot(bx - ax, by - ay)
    if dist < MIN_POINT_DISTANCE_PX:
        raise HorizonError(
            f"两个参照点重合或间距不足 {MIN_POINT_DISTANCE_PX:g}px"
            f"（实测 {dist:.3f}px），无法确定水平参照线方向，"
            "请沿参照线拉开距离重新点选两端。")
    return ax, ay, bx, by


# ---------------------------------------------------------------- 角度与裁剪

def correction_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """由参照线两端点（原图像素坐标）求反向校正旋转角。

    返回使该线变为水平所需的旋转角度（度）：正值 = 视觉逆时针，
    与 Pillow ``Image.rotate`` 的角度约定一致；归一化到 (-90, 90]。
    """
    deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if deg > 90.0:
        deg -= 180.0
    elif deg <= -90.0:
        deg += 180.0
    return deg


def largest_inner_rect(width: int, height: int, angle_deg: float) -> tuple[float, float]:
    """原图矩形旋转 ``angle_deg`` 后，画布（与原图同尺寸）内**最大居中、
    轴对齐且完全不含旋转填充**的矩形尺寸 ``(宽, 高)``（浮点、精确解）。

    设半宽 a、半高 b（矩形与画布同中心），约束为：

    - 四角落在旋转后的原图矩形内（旋转矩形为凸集，四角在内即整体在内）：

      ``cos·a + sin·b <= w/2``，``sin·a + cos·b <= h/2``；

    - 不超出画布：``a <= w/2``，``b <= h/2``。

    目标 max a·b。b 关于 a 的上界是三条直线的逐点最小值（分段线性），
    乘积 a·b(a) 分段二次，最优点只可能出现在：可行域端点、相邻约束线
    交点、或各二次段顶点。解析枚举全部候选取最大，为精确解。
    """
    alpha = abs(math.radians(angle_deg)) % math.pi
    if alpha > math.pi / 2:
        alpha = math.pi - alpha
    s, c = math.sin(alpha), math.cos(alpha)
    hw, hh = width / 2.0, height / 2.0
    EPS = 1e-12
    INF = float("inf")

    # a 的可行域上界：除画布约束 a<=hw 外，c=0 / s=0 时另一条约束
    # 会退化为对 a 的直接约束（旋转 90° 附近不可缺失）。
    a_max = hw
    if s > EPS:
        a_max = min(a_max, hh / s)   # sin·a + cos·b <= hh 在 c=0 时
    if c > EPS:
        a_max = min(a_max, hw / c)   # cos·a + sin·b <= hw 在 s=0 时

    def b1(a: float) -> float:   # cos·a + sin·b <= hw 给出的 b 上界
        return (hw - c * a) / s if s > EPS else INF

    def b2(a: float) -> float:   # sin·a + cos·b <= hh 给出的 b 上界
        return (hh - s * a) / c if c > EPS else INF

    def b_max(a: float) -> float:
        return max(0.0, min(b1(a), b2(a), hh))

    candidates = {0.0, a_max}
    if s > EPS and c > EPS:
        den = s * s - c * c
        if abs(den) > EPS:
            candidates.add((hh * s - hw * c) / den)   # b1 与 b2 交点
        candidates.add((hw - s * hh) / c)             # b1 与 b=hh 交点
    if s > EPS:
        candidates.add(hh * (1.0 - c) / s)            # b2 与 b=hh 交点
    if c > EPS:
        candidates.add(hw / (2.0 * c))                # a·b1(a) 段顶点
    if s > EPS:
        candidates.add(hh / (2.0 * s))                # a·b2(a) 段顶点

    best_a = best_b = 0.0
    for a in candidates:
        if a < -1e-9 or a > a_max + 1e-9:
            continue
        a = min(max(a, 0.0), a_max)
        b = b_max(a)
        if a * b > best_a * best_b:
            best_a, best_b = a, b
    return 2.0 * best_a, 2.0 * best_b


def crop_box_for(width: int, height: int, angle_deg: float) -> tuple[int, int, int, int]:
    """最大内接矩形取整后的居中裁剪框 ``(left, top, crop_w, crop_h)``。

    宽高向下取整并夹在画布内，再居中放置；结果保证完全落在旋转后的
    原图矩形覆盖范围内（不含旋转填充）。
    """
    wr, hr = largest_inner_rect(width, height, angle_deg)
    cw = min(int(math.floor(wr)), width)
    ch = min(int(math.floor(hr)), height)
    left = (width - cw) // 2
    top = (height - ch) // 2
    return left, top, cw, ch


# ---------------------------------------------------------------- 旋转与编码

def _fillcolor_for(mode: str):
    """旋转填充色：带透明通道用全透明，其余用黑色（最终都会被裁掉）。"""
    if mode == "RGBA":
        return (0, 0, 0, 0)
    if mode == "LA":
        return (0, 0)
    return 0


def _rotate(im: Image.Image, angle_deg: float) -> Image.Image:
    """绕图心旋转（画布与原图同尺寸），双三次重采样。"""
    return im.rotate(
        angle_deg,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=_fillcolor_for(im.mode),
    )


def _sentinel_coverage(width: int, height: int, angle_deg: float) -> np.ndarray:
    """与原图同尺寸的全白哨兵图做完全相同旋转：被旋转覆盖处为 255，
    旋转填充处为 0（哨兵图不含任何透明 / 空洞，白色铺满整个矩形）。"""
    sentinel = Image.new("L", (width, height), 255)
    rotated = sentinel.rotate(
        angle_deg, resample=Image.NEAREST, expand=False, fillcolor=0)
    return np.asarray(rotated)


def recheck_crop_coverage(
    width: int, height: int, angle_deg: float, box: tuple[int, int, int, int],
) -> dict:
    """哨兵覆盖自检：裁剪区内每个像素都必须被旋转后的原图矩形覆盖。

    返回自检明细；未覆盖像素数 > 0 时抛 :class:`HorizonRecheckError`。
    """
    left, top, cw, ch = box
    coverage = _sentinel_coverage(width, height, angle_deg)
    region = coverage[top:top + ch, left:left + cw]
    uncovered = int((region < 255).sum())
    result = {
        "method": "sentinel_rotation_coverage",
        "crop": {"left": left, "top": top, "width": cw, "height": ch},
        "crop_pixels": int(cw * ch),
        "uncovered_pixels": uncovered,
        "passed": uncovered == 0,
    }
    if uncovered:
        raise HorizonRecheckError(
            f"裁剪区覆盖自检未通过：{uncovered} 个像素可能含旋转填充。")
    return result


def encode_png(im: Image.Image) -> bytes:
    """像素经数组重建为全新图像对象后以 PNG 重编码（不带入元数据）。"""
    clean = Image.fromarray(np.asarray(im), im.mode)
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- 入口

def correct_horizon(data: bytes, x1, y1, x2, y2) -> dict:
    """对上传的 PNG 字节执行水平线校正，返回可直接序列化的结果字典。

    - 无法解码 / 非 PNG / 点越界 / 两点重合 / 尺寸超限 / 裁剪区退化
      -> :class:`HorizonError`（调用方返 400）；
    - 裁剪区覆盖自检未通过 -> :class:`HorizonRecheckError`
      （调用方返 500，不提供下载）。
    """
    im = open_png_image(data)
    width, height = im.size
    if max(width, height) > MAX_SOURCE_DIM:
        raise HorizonError(
            f"图片尺寸 {width}×{height} 超过 {MAX_SOURCE_DIM}px 上限，请先缩小。")

    px1, py1, px2, py2 = validate_points(x1, y1, x2, y2, width, height)
    angle = correction_angle_deg(px1, py1, px2, py2)
    # 参照线原始倾斜角（未归一化，展示用）；校正角为其归一化到 (-90,90] 的值
    line_tilt = math.degrees(math.atan2(py2 - py1, px2 - px1))

    # 规整像素模式：P 调板 / 带 tRNS 的 L·1 等按透明语义升格为 RGB/RGBA/LA，
    # 其余模式原样保留（与隐私副本导出同一实现，复用现有图片处理依赖）
    im = normalize_mode(im)

    rotated = _rotate(im, angle)
    box = crop_box_for(width, height, angle)
    left, top, cw, ch = box
    if cw < 1 or ch < 1:
        raise HorizonError(
            "按该参照线校正后，不含旋转填充的最大居中裁剪区不足 1px，"
            "请检查参照线是否过陡或图片过小。")

    recheck = recheck_crop_coverage(width, height, angle, box)

    corrected = rotated.crop((left, top, left + cw, top + ch))
    png_data = encode_png(corrected)

    return {
        "source_size": {"width": width, "height": height},
        "points": {
            "p1": {"x": px1, "y": py1},
            "p2": {"x": px2, "y": py2},
        },
        "line_tilt_deg": round(line_tilt, 4),
        "rotate_deg": round(angle, 4),
        "angle_convention": "正值 = 视觉逆时针（与 Pillow Image.rotate 一致），"
                            "已归一化到 (-90, 90] 度",
        "crop": {
            "left": left,
            "top": top,
            "width": cw,
            "height": ch,
            "right": left + cw,
            "bottom": top + ch,
            "coordinate": "旋转后画布（与原图同尺寸）左上角原点，半开区间",
        },
        "corrected_size": {"width": cw, "height": ch},
        "mode": corrected.mode,
        "has_alpha": bool(corrected.has_transparency_data),
        "rotation_fill": "透明" if corrected.has_transparency_data else "黑色",
        "recheck": recheck,
        "download": {
            "media_type": "image/png",
            "bytes": len(png_data),
            "sha256": hashlib.sha256(png_data).hexdigest(),
            "data_url": "data:image/png;base64,"
                        + base64.b64encode(png_data).decode("ascii"),
        },
        "stored": False,
    }


__all__ = [
    "MAX_SOURCE_DIM",
    "MIN_POINT_DISTANCE_PX",
    "HorizonError",
    "HorizonRecheckError",
    "open_png_image",
    "validate_points",
    "correction_angle_deg",
    "largest_inner_rect",
    "crop_box_for",
    "recheck_crop_coverage",
    "encode_png",
    "correct_horizon",
]
