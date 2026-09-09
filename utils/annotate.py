#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""annotate.py —— 车道线标注 GUI（跨平台，Windows/macOS/Linux）

读取 extract_json 产出的检测 JSON + 原始图像，逐帧显示中心点链，
鼠标点击循环切换三档标注：
    绿(1) = 是车道线    灰(0) = 否(误检/噪声)    蓝(2) = 难例(是但困难)

用法:
    python annotate.py <图像目录> <检测.json> [输出.json]

按键:
    a / ←  上一张      d / →  下一张
    s       手动保存     q / Esc 保存并退出
    c       切换显示(链 / 链+M5打分参考)
导航时自动保存标注（增量，二次打开继承旧标注）。

依赖: pip install opencv-python
"""

import sys
import os
import json
import time
import math

import cv2
import numpy as np

WIN = "lane-annotate"

# 三档状态名与颜色（BGR）
STATE_NAMES = ["否/灰", "是/绿", "难例/蓝"]
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
BLUE = (255, 128, 0)
STATE_COL = [GRAY, GREEN, BLUE]


def dist_to_seg(px, py, ax, ay, bx, by):
    """点到线段最短距离（显示像素坐标）"""
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    t = 0.0
    if len2 > 1e-12:
        t = ((px - ax) * abx + (py - ay) * aby) / len2
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


class Ctx:
    def __init__(self):
        self.image_dir = ""
        self.images = []        # 每图 dict(file, clusters:list[dict])
        self.cur = 0
        self.orig = None        # 原图 BGR
        self.sx = 1.0
        self.sy = 1.0
        self.show_m5 = False


g = Ctx()


def base_name(path):
    return os.path.basename(path)


def render(orig, clusters, states, sx, sy, show_m5):
    out = orig.copy()
    h, w = out.shape[:2]
    for i, c in enumerate(clusters):
        st = c["suggest"]
        col = STATE_COL[st % 3]
        pts = np.array(
            [[int(p[0] * sx), int(p[1] * sy)] for p in c["centers"]],
            dtype=np.int32,
        )
        lw = 6 if st == 0 else 12
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, col, lw, cv2.LINE_AA)
        a = tuple(pts[0]) if len(pts) else (10, 10)
        txt = "R%d %.2f" % (i, c.get("m5", 0.0)) if show_m5 else "R%d" % i
        cv2.putText(out, txt, (a[0] + 2, max(12, a[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # 图例
    ly = 16
    for s in range(3):
        cv2.circle(out, (12, ly - 4), 5, STATE_COL[s], -1, cv2.LINE_AA)
        cv2.putText(out, STATE_NAMES[s], (22, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COL[s], 1, cv2.LINE_AA)
        ly += 20
    return out


def on_mouse(event, x, y, _, __):
    if event != cv2.EVENT_LBUTTONDOWN or g.cur < 0:
        return
    clusters = g.images[g.cur]["clusters"]
    if not clusters or g.orig is None:
        return
    best, bestd = -1, 1e18
    for i, c in enumerate(clusters):
        pts = c["centers"]
        if len(pts) < 2:
            continue
        dmin = 1e18
        for k in range(1, len(pts)):
            d = dist_to_seg(x, y,
                            pts[k - 1][0] * g.sx, pts[k - 1][1] * g.sy,
                            pts[k][0] * g.sx, pts[k][1] * g.sy)
            if d < dmin:
                dmin = d
        if dmin < bestd:
            bestd, best = dmin, i
    if best < 0:
        return
    st = clusters[best]["suggest"] % 3
    tol = (6 if st == 0 else 12) / 2.0 + 8.0
    if bestd > tol:
        return
    clusters[best]["suggest"] = (clusters[best]["suggest"] + 1) % 3
    print("点击簇 R%d -> %s" % (best, STATE_NAMES[clusters[best]["suggest"] % 3]))
    cv2.imshow(WIN, render(g.orig, clusters,
                           [c["suggest"] for c in clusters],
                           g.sx, g.sy, g.show_m5))


def load_images(dirpath, jpath, outpath):
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    cached = {}
    if outpath and os.path.exists(outpath):
        try:
            with open(outpath, "r", encoding="utf-8") as f:
                od = json.load(f)
            for im in od.get("images", []):
                cached[im["file"]] = im["clusters"]
        except Exception:
            pass
    # 合并旧标注（同图簇数一致则覆盖，否则保留新默认）
    for im in data["images"]:
        old = cached.get(im["file"])
        if old and len(old) == len(im["clusters"]):
            for i in range(len(im["clusters"])):
                im["clusters"][i]["suggest"] = old[i]["suggest"]
    g.image_dir = os.path.abspath(dirpath)
    return data


def show_frame(idx, data):
    im = data["images"][idx]
    path = os.path.join(g.image_dir, im["file"])
    g.orig = cv2.imread(path, cv2.IMREAD_COLOR)
    if g.orig is None:
        print("[skip] 无法读图:", path)
        return False
    dh = data["meta"].get("det_h", 240)
    dw = data["meta"].get("det_w", 320)
    g.sx = g.orig.shape[1] / float(dw)
    g.sy = g.orig.shape[0] / float(dh)
    clusters = im["clusters"]
    n_green = sum(1 for c in clusters if c["suggest"] == 1)
    title = "ann %s [%d/%d] 绿=%d (a/d翻图 s存 q退 点击循环)" % (
        im["file"], idx + 1, len(data["images"]), n_green)
    cv2.setWindowTitle(WIN, title)
    cv2.imshow(WIN, render(g.orig, clusters,
                           [c["suggest"] for c in clusters],
                           g.sx, g.sy, g.show_m5))
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    imgdir = sys.argv[1]
    jpath = sys.argv[2]
    outpath = sys.argv[3] if len(sys.argv) > 3 else jpath

    data = load_images(imgdir, jpath, outpath)
    if not data["images"]:
        print("无图像条目")
        return 1

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    ndata = dict(data)
    ndata["generator"] = "annotate.py"
    if not show_frame(0, ndata):
        return 1

    def save():
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(ndata, f, ensure_ascii=False, indent=1)
        green = sum(1 for c in ndata["images"][g.cur]["clusters"]
                    if c["suggest"] == 1)
        print("[save] 已保存到 %s (当前图绿=%d)" % (outpath, green))

    g.cur = 0
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            save()
            break
        if key == ord("c") or key == ord("C"):
            g.show_m5 = not g.show_m5
            cur = ndata["images"][g.cur]
            cv2.imshow(WIN, render(g.orig, cur["clusters"],
                                   [c["suggest"] for c in cur["clusters"]],
                                   g.sx, g.sy, g.show_m5))
        nxt = None
        # 方向键/ASCII 统一处理（部分后端方向键带高字节，这里兼容常见形式）
        if key in (ord("a"), 81):
            nxt = (g.cur - 1) % len(ndata["images"])
        elif key in (ord("d"), 83):
            nxt = (g.cur + 1) % len(ndata["images"])
        elif key == ord("s") or key == ord("S"):
            save()
        if nxt is not None:
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(ndata, f, ensure_ascii=False, indent=1)
            g.cur = nxt
            show_frame(g.cur, ndata)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())