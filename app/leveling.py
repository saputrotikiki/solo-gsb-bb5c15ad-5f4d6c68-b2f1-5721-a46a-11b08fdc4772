"""图片水平线校正 —— 两点选线、反向旋转、最大居中无填充裁剪。

流程（全程仅在内存中进行，不落盘、不创建任务）：

  1. 仅接受 PNG（按 Pillow 解码格式判定），复用统一的大小 / 尺寸上限；
  2. 用户在页面预览上点选**同一条水平参照线的两端**（原图像素坐标，
     左上角为原点，x 向右、y 向下）。两点必须落在图像范围内且不得
     重合（间距 < 1px 视为重合），否则明确拒绝；
  3. 参照线相对水平线的倾斜角 τ = atan2(y2−y1, x2−x1)（屏幕坐标，
     顺时针为正）；把整图**反向旋转**同一角度即可让参照线水平。
     返回的 ``angle_deg`` 即 Pillow ``Image.rotate`` 语义下的校正角
     （逆时针为正，归一化到 (−90, 90]，数值上与 τ 相同），
     旋转画布扩展（expand）为完整外接矩形；
  4. 旋转产生的**填充空白**用一张与原图同步旋转的**覆盖掩膜**界定：
     掩膜只表示「该像素是否来自原图矩形」，因此**原图自身的透明
     像素不会被当成填充空白**；在掩膜上求**最大居中、轴对齐且完全
     不含填充空白**的裁剪矩形（面积最大优先，面积相同取高度更大者）；
  5. 裁剪结果以全新 PNG 重编码返回（RGB / RGBA，保留原图透明通道），
     可直接送入现有水印嵌入表单（最短边 ≥ 128px 时满足嵌入要求）。

与水印任务、攻击记录、报告归档完全隔离。
"""

from __future__ import annotations

import base64
import hashlib
import io
import math

import numpy as np
from PIL import Image

# 校正结果需满足水印嵌入的最短边要求时，前端给出提示（与 dctwm.TILE*BLK 一致）
WATERMARK_MIN_DIM = 128

# 两点间距小于该值视为「重合」（亚像素级点击无法确定稳定角度）
MIN_POINT_DISTANCE_PX = 1.0


class LevelingError(ValueError):
    """校正前置条件不满足（点重合 / 越界 / 非 PNG / 无法裁剪等）。"""


# ---------------------------------------------------------------- 输入校验

def open_png(data: bytes) -> Image.Image:
    """打开并严格校验 PNG 图片（只接受 PNG，区别于隐私导出的 PNG/JPEG）。"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise LevelingError("无法解析上传的图片。") from e
    if im.format != "PNG":
        raise LevelingError(
            f"水平线校正仅支持 PNG 图片（当前为 {im.format or '未知格式'}）。")
    return im


def validate_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """校验两个参照点：有限数值、落在图像范围内、不重合。

    坐标为原图像素坐标（左上角原点，x∈[0,w−1]，y∈[0,h−1]）。
    """
    pts: list[tuple[float, float]] = []
    for name, p in (("p1", p1), ("p2", p2)):
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError) as e:
            raise LevelingError(f"{name} 坐标必须是两个数值。") from e
        if not (math.isfinite(x) and math.isfinite(y)):
            raise LevelingError(f"{name} 坐标必须是有限数值。")
        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise LevelingError(
                f"{name} ({x:g}, {y:g}) 超出图像范围"
                f"（x∈[0, {width - 1}]，y∈[0, {height - 1}]）。")
        pts.append((x, y))
    (x1, y1), (x2, y2) = pts
    if math.hypot(x2 - x1, y2 - y1) < MIN_POINT_DISTANCE_PX:
        raise LevelingError(
            "两个参照点重合（间距不足 1px），无法确定参照线方向，"
            "请点选同一条水平参照线上相距较远的两端。")
    return pts[0], pts[1]


# ---------------------------------------------------------------- 角度计算

def compute_angle_deg(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """由两点求校正旋转角（度），归一化到 (−90, 90]。

    屏幕坐标（y 向下）下参照线的倾斜角 τ = atan2(dy, dx)（顺时针为正）；
    把图像按 Pillow ``Image.rotate`` 语义旋转同一数值（逆时针为正）即可
    让参照线水平，因此 ``angle_deg`` 在数值上等于 τ。归一化到 (−90, 90]
    后点选顺序（先点哪一端）不影响结果。
    """
    (x1, y1), (x2, y2) = p1, p2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    # 归一化到 (−90, 90]：atan2 直接给出 (−90, 90] 之外的角度仅当 |angle|>90
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    # 消除 -0.0
    return angle + 0.0


# ---------------------------------------------------------------- 旋转 + 掩膜

def _normalize_for_rotation(im: Image.Image) -> tuple[Image.Image, bool]:
    """规整为 RGB / RGBA；返回 (图像, 原图是否含透明语义)。"""
    try:
        has_alpha = bool(im.has_transparency_data)
    except Exception:
        has_alpha = "A" in im.getbands()
    if has_alpha:
        return im.convert("RGBA"), True
    return im.convert("RGB"), False


def _rotate_with_coverage_mask(
    im: Image.Image, angle_deg: float
) -> tuple[Image.Image, np.ndarray]:
    """按 expand 旋转图像，并同步旋转一张全 255 掩膜以界定填充空白。

    掩膜为 1 表示该像素的重采样窗口完全落在原图矩形内（**与原图透明
    通道无关**），为 0 表示旋转产生的填充空白（含边缘约 1–2px 的填充
    混合带）。图像与掩膜尺寸 / 角度 / expand 完全一致，逐像素几何对应。
    """
    w, h = im.size
    rotated = im.rotate(
        angle_deg, resample=Image.BICUBIC, expand=True,
        fillcolor=(0, 0, 0, 0) if im.mode == "RGBA" else (0, 0, 0),
    )
    mask_im = Image.new("L", (w, h), 255).rotate(
        angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=0,
    )
    # 常数 255 区域经 BICUBIC 仍恰好为 255；只有重采样窗口触及填充 0
    # 的边缘像素才 < 255。阈值 254 保证裁剪区内不含任何填充混合。
    mask = (np.asarray(mask_im) >= 254).astype(np.uint8)
    return rotated, mask


# ---------------------------------------------------------------- 最大居中裁剪

def largest_centered_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    """在 0/1 掩膜上求最大居中、轴对齐且全为 1 的裁剪矩形。

    返回半开区间 ``(left, top, right, bottom)``（Pillow crop 约定）。
    「居中」指以画布中心为中心（奇偶尺寸允许 1px 偏差）；面积最大优先，
    面积相同取高度更大（垂直内缩更小）者。

    逐行求该行允许的最大水平半内缩 ``a``：偶数宽时裁剪列为
    ``[cx2−a, cx2+a)``、奇数宽时为 ``[cx2−a, cx2+a+1)``。设该行最靠近
    中心的左侧 / 右侧填充列为 zL / zR，则
    ``a ≤ cx2−zL−1`` 且 ``a ≤ zR−right_start``。掩膜关于画布中心
    180° 对称，但同一行不必然左右对称，故两侧约束都要取。
    垂直方向逐 ``b`` 扫描行窗（``b`` 增大时窗只向外扩展，窗内最小 ``a``
    单调不增），取面积最大者，O(h) 完成。
    """
    h, w = mask.shape
    cy2, cx2 = h // 2, w // 2
    wpad = 1 if w % 2 else 0          # 奇数宽时裁剪宽度 = 2a+1
    hpad = 1 if h % 2 else 0          # 奇数高时裁剪高度 = 2b+1
    right_start = cx2 + wpad         # 右半起始列

    rows = np.full(h, cx2, dtype=np.int64)  # 每行允许的最大水平半内缩 a
    for y in range(h):
        zeros = np.flatnonzero(mask[y] == 0)
        if zeros.size == 0:
            continue
        left_zeros = zeros[zeros < right_start]
        right_zeros = zeros[zeros >= right_start]
        amax = cx2
        if left_zeros.size:
            amax = min(amax, cx2 - int(left_zeros.max()) - 1)
        if right_zeros.size:
            amax = min(amax, int(right_zeros.min()) - right_start)
        rows[y] = amax  # 可能为 -1：该行连中心列都是填充，无法纳入任何裁剪

    best_area, best_a, best_b = -1, -1, -1
    # 偶数高时 b=0 窗高为 0，从 1 开始
    for b in range(0 if hpad else 1, cy2 + 1):
        lo, hi = cy2 - b, cy2 + b + hpad
        a = int(rows[lo:hi].min())
        width = 2 * a + wpad
        area = width * (hi - lo)
        # b 递增遍历：面积相同时先出现者高度更大，自然满足取舍规则
        if area > best_area:
            best_area, best_a, best_b = area, a, b

    if best_area <= 0 or best_a < 0:
        raise LevelingError(
            "旋转角度过大，旋转后不存在不含填充空白的居中裁剪区，"
            "请重新选择更接近水平的参照线。")

    left = cx2 - best_a
    right = cx2 + best_a + wpad
    top = cy2 - best_b
    bottom = cy2 + best_b + hpad
    return left, top, right, bottom


# ---------------------------------------------------------------- 入口

def straighten_image(
    data: bytes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    max_dim: int = 4096,
) -> dict:
    """执行水平线校正，返回角度、裁剪范围与可下载 PNG（data URL）。

    - 非 PNG / 无法解码 / 点重合 / 点越界 / 无法裁剪 -> LevelingError；
    - 返回的 ``crop`` 为旋转画布坐标系下的半开区间 [left, top, right, bottom]；
    - 全程内存，不落盘、不创建任务。
    """
    im = open_png(data)
    width, height = im.size
    if max(width, height) > max_dim:
        raise LevelingError(
            f"图片尺寸 {width}x{height} 超过 {max_dim}px 上限，请先缩小。")
    p1, p2 = validate_points(p1, p2, width, height)
    angle = compute_angle_deg(p1, p2)

    normalized, has_alpha = _normalize_for_rotation(im)
    rotated, mask = _rotate_with_coverage_mask(normalized, angle)
    left, top, right, bottom = largest_centered_crop(mask)
    cropped = rotated.crop((left, top, right, bottom))
    crop_w, crop_h = cropped.size

    # 全新 PNG 重编码（fromarray 重建，不带入任何元数据）
    arr = np.asarray(cropped)
    clean = Image.fromarray(arr, cropped.mode)
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=True)
    png_data = buf.getvalue()

    rot_w, rot_h = rotated.size
    # angle > 0：参照线右端偏低（dy>0），Pillow 正向旋转即逆时针校正
    tilt_dir = "水平（无需旋转）" if angle == 0.0 else (
        "逆时针校正（参照线右端偏低）" if angle > 0
        else "顺时针校正（参照线右端偏高）")
    return {
        "width": width,
        "height": height,
        "mode": cropped.mode,
        "has_alpha": has_alpha,
        "p1": {"x": p1[0], "y": p1[1]},
        "p2": {"x": p2[0], "y": p2[1]},
        "angle_deg": round(angle, 4),
        "angle_direction": tilt_dir,
        "angle_convention": (
            "angle_deg 为 Pillow Image.rotate 语义下的校正角（逆时针为正，"
            "已归一化到 (−90, 90]）；数值上等于屏幕坐标（y 向下）中参照线"
            "相对水平线的倾斜角，按该角度反向旋转后参照线水平。"),
        "rotated_size": {"width": rot_w, "height": rot_h},
        "crop": {
            "left": left, "top": top, "right": right, "bottom": bottom,
            "width": crop_w, "height": crop_h,
            "coordinate": "旋转画布坐标（左上角原点，半开区间 [left, right) × [top, bottom)）",
        },
        "fits_watermark_input": min(crop_w, crop_h) >= WATERMARK_MIN_DIM,
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
    "LevelingError",
    "MIN_POINT_DISTANCE_PX",
    "WATERMARK_MIN_DIM",
    "open_png",
    "validate_points",
    "compute_angle_deg",
    "largest_centered_crop",
    "straighten_image",
]
