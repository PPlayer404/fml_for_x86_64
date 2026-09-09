// 项目名: fml_for_x86_64
// 文件名: fld.cpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// 传统 Canny/FLT 管线前端（fml_detect，检测框架出自 [1]，NFA 判据沿用 [2]）：
//   Sobel 梯度（3x3 直算 / 5 点可分离高斯-导数，SIMD）→ Canny 边缘
//   （幅值 L1 + 方向 NMS + 滞后连通）→ 边缘链提取（8 连通游走 + 方向
//   一致性）→ 最小二乘拟合 → NFA 显著性过滤 → 先验平移
// 后处理（长度/贴边过滤、方向统一、共线合并）由 detector.cpp 统一实现。
//
// 参考文献：
//   [1] Lee et al., Outdoor place recognition ... using straight lines, ICRA 2014.
//   [2] von Gioi et al., LSD: A Line Segment Detector, IPL 2012.

#include "detector.hpp"

#include <cstring>

namespace fml
{
namespace
{
// 8 连通方向枚举序表（getPointChainCached 的枚举序与平局规则与其一致）
const int DIR_IDX[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}};
} // namespace（文件内部）

// 梯度计算入口：σ>0 走 sobelSeparable；否则 3x3 可分离 Sobel——
// 横向/纵向差分入 hbuf_/vbuf_（含 replicate 哨兵行），再分别做
// [1,2,1] 平滑合成 dx_/dy_
void FmlDetector::computeGradientAngle(const cv::Mat &gray)
{
    if (grad_sigma > 0)
    {
        sobelSeparable(gray);
        return;
    }
    const int rows = gray.rows, cols = gray.cols;
    dx_.create(rows, cols, CV_16S);
    dy_.create(rows, cols, CV_16S);
    hbuf_.create(rows + 2, cols, CV_16S);
    vbuf_.create(rows + 2, cols, CV_16S);

    short *hb = hbuf_.ptr<short>(0);
    short *vb = vbuf_.ptr<short>(0);
    const size_t step = gray.step;
    const unsigned char *I = gray.ptr(0);
#if FML_AVX2
    const __m128i zero = _mm_setzero_si128();
#endif

    for (int r = 0; r < rows; ++r)
    {
        const unsigned char *row = I + r * step;
        const unsigned char *up = I + (size_t)(r > 0 ? r - 1 : 0) * step;
        const unsigned char *dn = I + (size_t)(r < rows - 1 ? r + 1 : rows - 1) * step;
        short *h = hb + (size_t)(r + 1) * cols;
        short *v = vb + (size_t)(r + 1) * cols;

        // 横向 [-1,0,1] 与 纵向 [-1,0,1]（左右端点单独处理）
        h[0] = (short)(row[1] - row[0]);
        v[0] = (short)(dn[0] - up[0]);
        if (cols > 1)
        {
            h[cols - 1] = (short)(row[cols - 1] - row[cols - 2]);
            v[cols - 1] = (short)(dn[cols - 1] - up[cols - 1]);
        }
        int c = 1;
#if FML_AVX2
        for (; c + 15 < cols - 1; c += 16)
        {
            // u8 → i16 解包后相减
            __m128i hl = _mm_loadu_si128((const __m128i *)(row + c - 1));
            __m128i hr = _mm_loadu_si128((const __m128i *)(row + c + 1));
            _mm_storeu_si128((__m128i *)(h + c),
                             _mm_sub_epi16(_mm_unpacklo_epi8(hr, zero), _mm_unpacklo_epi8(hl, zero)));
            _mm_storeu_si128((__m128i *)(h + c + 8),
                             _mm_sub_epi16(_mm_unpackhi_epi8(hr, zero), _mm_unpackhi_epi8(hl, zero)));
            __m128i vu = _mm_loadu_si128((const __m128i *)(up + c));
            __m128i vd = _mm_loadu_si128((const __m128i *)(dn + c));
            _mm_storeu_si128((__m128i *)(v + c),
                             _mm_sub_epi16(_mm_unpacklo_epi8(vd, zero), _mm_unpacklo_epi8(vu, zero)));
            _mm_storeu_si128((__m128i *)(v + c + 8),
                             _mm_sub_epi16(_mm_unpackhi_epi8(vd, zero), _mm_unpackhi_epi8(vu, zero)));
        }
#endif
        for (; c < cols - 1; ++c)
        {
            h[c] = (short)(row[c + 1] - row[c - 1]);
            v[c] = (short)(dn[c] - up[c]);
        }
    }
    // 上下 replicate 哨兵行
    memcpy(hb, hb + cols, (size_t)cols * sizeof(short));
    memcpy(hb + (size_t)(rows + 1) * cols, hb + (size_t)rows * cols, (size_t)cols * sizeof(short));
    memcpy(vb, vb + cols, (size_t)cols * sizeof(short));
    memcpy(vb + (size_t)(rows + 1) * cols, vb + (size_t)rows * cols, (size_t)cols * sizeof(short));

    // dy = 纵向差分后做横向 [1,2,1] 平滑；dx = 横向差分后做纵向 [1,2,1]
    for (int r = 0; r < rows; ++r)
    {
        const short *p0 = hb + (size_t)r * cols;
        const short *p1 = p0 + cols;
        const short *p2 = p1 + cols;
        short *dxd = dx_.ptr<short>(r);
        const short *q = vb + (size_t)(r + 1) * cols;
        short *dyd = dy_.ptr<short>(r);
        // 标量尾 + 边界（横向 replicate）
        dxd[0] = (short)(p0[0] + 2 * p1[0] + p2[0]);
        dyd[0] = (short)((cols > 1 ? q[1] : 4 * q[0]) + 3 * q[0]);
        if (cols > 1)
            dyd[cols - 1] = (short)(q[cols - 2] + 3 * q[cols - 1]);
        int cc = 1;
#if FML_AVX2
        for (; cc + 17 <= cols; cc += 16)
        {
            __m128i a0 = _mm_loadu_si128((const __m128i *)(p0 + cc));
            __m128i a1 = _mm_loadu_si128((const __m128i *)(p1 + cc));
            __m128i a2 = _mm_loadu_si128((const __m128i *)(p2 + cc));
            __m128i s = _mm_add_epi16(_mm_add_epi16(a0, a2), _mm_add_epi16(a1, a1));
            _mm_storeu_si128((__m128i *)(dxd + cc), s);
            a0 = _mm_loadu_si128((const __m128i *)(p0 + cc + 8));
            a1 = _mm_loadu_si128((const __m128i *)(p1 + cc + 8));
            a2 = _mm_loadu_si128((const __m128i *)(p2 + cc + 8));
            s = _mm_add_epi16(_mm_add_epi16(a0, a2), _mm_add_epi16(a1, a1));
            _mm_storeu_si128((__m128i *)(dxd + cc + 8), s);

            __m128i b0 = _mm_loadu_si128((const __m128i *)(q + cc - 1));
            __m128i b1 = _mm_loadu_si128((const __m128i *)(q + cc));
            __m128i b2 = _mm_loadu_si128((const __m128i *)(q + cc + 1));
            __m128i t = _mm_add_epi16(_mm_add_epi16(b0, b2), _mm_add_epi16(b1, b1));
            _mm_storeu_si128((__m128i *)(dyd + cc), t);
            b0 = _mm_loadu_si128((const __m128i *)(q + cc + 7));
            b1 = _mm_loadu_si128((const __m128i *)(q + cc + 8));
            b2 = _mm_loadu_si128((const __m128i *)(q + cc + 9));
            t = _mm_add_epi16(_mm_add_epi16(b0, b2), _mm_add_epi16(b1, b1));
            _mm_storeu_si128((__m128i *)(dyd + cc + 8), t);
        }
#endif
        for (; cc < cols - 1; ++cc)
        {
            dxd[cc] = (short)(p0[cc] + 2 * p1[cc] + p2[cc]);
            dyd[cc] = (short)(q[cc - 1] + 2 * q[cc] + q[cc + 1]);
        }
        dxd[cols - 1] = (short)(p0[cols - 1] + 2 * p1[cols - 1] + p2[cols - 1]);
    }
}

// dx/dy 双链首级滤波：水平差分（x 链）+ 垂直差分（y 链）
// 一次遍历 8 位灰度平面，同行产出两个输出平面
static void diffXY5(cv::Mat &dxl, cv::Mat &dyl, const cv::Mat &src8,
                    const int kdq[5], int r0 = 0, int r1 = -1)
{
    const int rows = src8.rows, cols = src8.cols;
    if (r1 < 0)
        r1 = rows;
#if FML_AVX2
    const __m256i vrnd = _mm256_set1_epi32(128);
    const __m256i vkA = _mm256_set1_epi32((int)(((unsigned)(-kdq[0]) << 16) | ((unsigned)kdq[4] & 0xFFFFu)));
    const __m256i vkB = _mm256_set1_epi32((int)(((unsigned)(-kdq[1]) << 16) | ((unsigned)kdq[3] & 0xFFFFu)));
#endif
    for (int r = r0; r < r1; ++r)
    {
        const unsigned char *sp = src8.ptr<unsigned char>(r);
        const unsigned char *up2 = src8.ptr<unsigned char>(r >= 2 ? r - 2 : 0);
        const unsigned char *up1 = src8.ptr<unsigned char>(r >= 1 ? r - 1 : 0);
        const unsigned char *dn1 = src8.ptr<unsigned char>(r < rows - 1 ? r + 1 : rows - 1);
        const unsigned char *dn2 = src8.ptr<unsigned char>(r < rows - 2 ? r + 2 : rows - 1);
        short *dpx = dxl.ptr<short>(r);
        short *dpy = dyl.ptr<short>(r);
#if FML_AVX2
        int c = 2; // x 链：AVX2 块从 c=2 起步（需读 sp[c-2]）
        for (; c + 18 <= cols; c += 16)
        {
            __m256i p2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c + 2)));
            __m256i p1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c + 1)));
            __m256i m2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c - 2)));
            __m256i m1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c - 1)));
            __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(p2, m2), vkA);
            __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(p2, m2), vkA);
            lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(p1, m1), vkB));
            hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(p1, m1), vkB));
            lo = _mm256_srai_epi32(_mm256_add_epi32(lo, vrnd), 8);
            hi = _mm256_srai_epi32(_mm256_add_epi32(hi, vrnd), 8);
            _mm256_storeu_si256((__m256i *)(dpx + c), _mm256_packs_epi32(lo, hi));
        }
        for (; c < cols - 2; ++c)
        {
            int v = -kdq[0] * sp[c - 2] - kdq[1] * sp[c - 1] + kdq[3] * sp[c + 1] + kdq[4] * sp[c + 2];
            dpx[c] = (short)((v + 128) >> 8);
        }
        // 边界列 0、1、cols-2、cols-1（replicate 钳位）
        {
            int eb[4] = {0, 1, cols >= 2 ? cols - 2 : 0, cols >= 1 ? cols - 1 : 0};
            if (cols == 1)
                eb[1] = eb[2] = eb[3] = 0;
            else if (cols == 2)
                eb[2] = eb[3] = 1;
            for (int ei = 0; ei < 4; ++ei)
            {
                int cc = eb[ei];
                if (cc >= cols)
                    continue;
                if (ei > 0 && cc <= eb[ei - 1])
                    continue;
                int cm2 = cc >= 2 ? cc - 2 : 0;
                int cm1 = cc >= 1 ? cc - 1 : 0;
                int cp1 = cc < cols - 1 ? cc + 1 : cols - 1;
                int cp2 = cc < cols - 2 ? cc + 2 : cols - 1;
                int v = -kdq[0] * sp[cm2] - kdq[1] * sp[cm1] + kdq[3] * sp[cp1] + kdq[4] * sp[cp2];
                dpx[cc] = (short)((v + 128) >> 8);
            }
        }
        // y 链：全列（读四行同列）
        {
            int cc = 0;
            for (; cc + 16 <= cols; cc += 16)
            {
                __m256i d2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(dn2 + cc)));
                __m256i d1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(dn1 + cc)));
                __m256i u2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(up2 + cc)));
                __m256i u1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(up1 + cc)));
                __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(d2, u2), vkA);
                __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(d2, u2), vkA);
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(d1, u1), vkB));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(d1, u1), vkB));
                lo = _mm256_srai_epi32(_mm256_add_epi32(lo, vrnd), 8);
                hi = _mm256_srai_epi32(_mm256_add_epi32(hi, vrnd), 8);
                _mm256_storeu_si256((__m256i *)(dpy + cc), _mm256_packs_epi32(lo, hi));
            }
            for (; cc < cols; ++cc)
            {
                int v = -kdq[0] * up2[cc] - kdq[1] * up1[cc] + kdq[3] * dn1[cc] + kdq[4] * dn2[cc];
                dpy[cc] = (short)((v + 128) >> 8);
            }
        }
#else
        // 标量路径（非 AVX2 构建）
        for (int c = 0; c < cols; ++c)
        {
            int cm2 = c >= 2 ? c - 2 : 0;
            int cm1 = c >= 1 ? c - 1 : 0;
            int cp1 = c < cols - 1 ? c + 1 : cols - 1;
            int cp2 = c < cols - 2 ? c + 2 : cols - 1;
            dpx[c] = (short)((-kdq[0] * sp[cm2] - kdq[1] * sp[cm1] + kdq[3] * sp[cp1] + kdq[4] * sp[cp2] + 128) >> 8);
            dpy[c] = (short)((-kdq[0] * up2[c] - kdq[1] * up1[c] + kdq[3] * dn1[c] + kdq[4] * dn2[c] + 128) >> 8);
        }
#endif
    }
}

// 垂直平滑(x 链次级) + 水平平滑(y 链次级) 一次遍历两个 i16 平面
static void smoothXY5(cv::Mat &dx, cv::Mat &dy, const cv::Mat &sx, const cv::Mat &sy,
                      const int ksq[5], int r0 = 0, int r1 = -1)
{
    const int rows = sx.rows, cols = sx.cols;
    if (r1 < 0)
        r1 = rows;
#if FML_AVX2
    const __m256i vrnd = _mm256_set1_epi32(128);
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vkA = _mm256_set1_epi32((int)(((unsigned)ksq[0] << 16) | ((unsigned)ksq[4] & 0xFFFFu)));
    const __m256i vkB = _mm256_set1_epi32((int)(((unsigned)ksq[1] << 16) | ((unsigned)ksq[3] & 0xFFFFu)));
    const __m256i vkC = _mm256_set1_epi32((int)((unsigned)ksq[2] & 0xFFFFu));
#endif
    for (int r = r0; r < r1; ++r)
    {
        const short *mx2 = sx.ptr<short>(r >= 2 ? r - 2 : 0);
        const short *mx1 = sx.ptr<short>(r >= 1 ? r - 1 : 0);
        const short *mx0 = sx.ptr<short>(r);
        const short *mx_1 = sx.ptr<short>(r < rows - 1 ? r + 1 : rows - 1);
        const short *mx_2 = sx.ptr<short>(r < rows - 2 ? r + 2 : rows - 1);
        const short *my = sy.ptr<short>(r);
        short *ox = dx.ptr<short>(r);
        short *oy = dy.ptr<short>(r);
#if FML_AVX2
        // x 链次级（垂直平滑）：全列 AVX2
        {
            int c = 0;
            for (; c + 16 <= cols; c += 16)
            {
                __m256i d2 = _mm256_loadu_si256((const __m256i *)(mx_2 + c));
                __m256i d1 = _mm256_loadu_si256((const __m256i *)(mx_1 + c));
                __m256i u2 = _mm256_loadu_si256((const __m256i *)(mx2 + c));
                __m256i u1 = _mm256_loadu_si256((const __m256i *)(mx1 + c));
                __m256i midv = _mm256_loadu_si256((const __m256i *)(mx0 + c));
                __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(d2, u2), vkA);
                __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(d2, u2), vkA);
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(d1, u1), vkB));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(d1, u1), vkB));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(midv, vzero), vkC));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(midv, vzero), vkC));
                lo = _mm256_srai_epi32(_mm256_add_epi32(lo, vrnd), 8);
                hi = _mm256_srai_epi32(_mm256_add_epi32(hi, vrnd), 8);
                _mm256_storeu_si256((__m256i *)(ox + c), _mm256_packs_epi32(lo, hi));
            }
            for (; c < cols; ++c)
                ox[c] = (short)((ksq[0] * mx2[c] + ksq[1] * mx1[c] + ksq[2] * mx0[c] + ksq[3] * mx_1[c] + ksq[4] * mx_2[c] + 128) >> 8);
        }
        // y 链次级（水平平滑）：AVX2 主块自 c=2 起步，边界列单独钳位计算
        {
            int c = 2;
            for (; c + 18 <= cols; c += 16)
            {
                __m256i p2 = _mm256_loadu_si256((const __m256i *)(my + c + 2));
                __m256i p1 = _mm256_loadu_si256((const __m256i *)(my + c + 1));
                __m256i m2 = _mm256_loadu_si256((const __m256i *)(my + c - 2));
                __m256i m1 = _mm256_loadu_si256((const __m256i *)(my + c - 1));
                __m256i mid = _mm256_loadu_si256((const __m256i *)(my + c));
                __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(p2, m2), vkA);
                __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(p2, m2), vkA);
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(p1, m1), vkB));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(p1, m1), vkB));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(mid, vzero), vkC));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(mid, vzero), vkC));
                lo = _mm256_srai_epi32(_mm256_add_epi32(lo, vrnd), 8);
                hi = _mm256_srai_epi32(_mm256_add_epi32(hi, vrnd), 8);
                _mm256_storeu_si256((__m256i *)(oy + c), _mm256_packs_epi32(lo, hi));
            }
            for (; c < cols - 2; ++c)
            {
                int v = ksq[0] * my[c - 2] + ksq[1] * my[c - 1] + ksq[2] * my[c] + ksq[3] * my[c + 1] + ksq[4] * my[c + 2];
                oy[c] = (short)((v + 128) >> 8);
            }
            {
                int eb[4] = {0, 1, cols >= 2 ? cols - 2 : 0, cols >= 1 ? cols - 1 : 0};
                if (cols == 1)
                    eb[1] = eb[2] = eb[3] = 0;
                else if (cols == 2)
                    eb[2] = eb[3] = 1;
                for (int ei = 0; ei < 4; ++ei)
                {
                    int cc = eb[ei];
                    if (cc >= cols)
                        continue;
                    if (ei > 0 && cc <= eb[ei - 1])
                        continue;
                    int cm2 = cc >= 2 ? cc - 2 : 0;
                    int cm1 = cc >= 1 ? cc - 1 : 0;
                    int cp1 = cc < cols - 1 ? cc + 1 : cols - 1;
                    int cp2 = cc < cols - 2 ? cc + 2 : cols - 1;
                    int v = ksq[0] * my[cm2] + ksq[1] * my[cm1] + ksq[2] * my[cc] + ksq[3] * my[cp1] + ksq[4] * my[cp2];
                    oy[cc] = (short)((v + 128) >> 8);
                }
            }
        }
#else
        for (int c = 0; c < cols; ++c)
        {
            int cm2 = c >= 2 ? c - 2 : 0;
            int cm1 = c >= 1 ? c - 1 : 0;
            int cp1 = c < cols - 1 ? c + 1 : cols - 1;
            int cp2 = c < cols - 2 ? c + 2 : cols - 1;
            ox[c] = (short)((ksq[0] * mx2[c] + ksq[1] * mx1[c] + ksq[2] * mx0[c] + ksq[3] * mx_1[c] + ksq[4] * mx_2[c] + 128) >> 8);
            oy[c] = (short)((ksq[0] * my[cm2] + ksq[1] * my[cm1] + ksq[2] * my[c] + ksq[3] * my[cp1] + ksq[4] * my[cp2] + 128) >> 8);
        }
#endif
    }
}

// 5 点可分离高斯-导数梯度（computeGradientAngle 的 σ>0 路径）：
// 平滑/差分核均由 σ 高斯采样生成，
//   S(u) = exp(-u²/2σ²)，归一化 ΣS=1
//   D(u) ∝ -u·exp(-u²/2σ²)，归一化 Σu·D(u)=1（单位斜坡响应）
// 核仅含 5 点，实用范围约 σ∈(0, 2]；更大平滑由调用方预模糊。
void FmlDetector::sobelSeparable(const cv::Mat &gray)
{
    const int rows = gray.rows, cols = gray.cols;
    dx_.create(rows, cols, CV_16S);
    dy_.create(rows, cols, CV_16S);
    pa_.create(rows, cols, CV_16S);
    pb_.create(rows, cols, CV_16S);

    // 核生成：q = exp(-1/2σ²)，则 G(±1)=q、G(±2)=q⁴，全程一次 exp。
    // 定标与二项式 5 点核同量级（ΣS=16、Σ|D|=6），保证阈值量级一致：
    const double ss = (double)grad_sigma;
    const double q = std::exp(-0.5 / (ss * ss));
    const double g1 = q, g2 = q * q * q * q;

    const double sraw[5] = {g2, 4 * g1, 6.0, 4 * g1, g2};
    const double ssum = sraw[0] + sraw[1] + sraw[2] + sraw[3] + sraw[4];
    const double sscale = 16.0 / ssum;
    for (int i = 0; i < 5; ++i)
        ks_[i] = (float)(sraw[i] * sscale);

    const double draw_[5] = {2 * g2, g1, 0.0, g1, 2 * g2}; // |−u·G(u)|，左右对称
    const double dnorm = draw_[0] + draw_[1] + draw_[3] + draw_[4];
    if (dnorm > 1e-12)
    {
        const float dscale = (float)(6.0 / dnorm);
        kd_[0] = (float)(draw_[0] * dscale);
        kd_[1] = (float)(draw_[1] * dscale);
        kd_[2] = 0.0f;
        kd_[3] = (float)(draw_[3] * dscale);
        kd_[4] = (float)(draw_[4] * dscale);
    }
    else
    {
        kd_[0] = -1.0f;
        kd_[1] = 0.0f;
        kd_[2] = 0.0f;
        kd_[3] = 0.0f;
        kd_[4] = 1.0f;
    }

    // Q8.8 定点核（AVX2 路径）
    for (int i = 0; i < 5; ++i)
    {
        ks_q_[i] = (int)std::lround(ks_[i] * 256.0);
        kd_q_[i] = (int)std::lround(kd_[i] * 256.0); // 存 |值|；差分符号在滤波内取
    }

    // x 链（dx）：水平差分 → 垂直平滑；y 链（dy）：垂直差分 → 水平平滑
    // 两链的同行 pass 融合为一次遍历
    // threads=2：两 pass 各按行二分并行（输出行不相交、源行只读）
    if (nthr_ == 2)
    {
        poolRows(rows, [&](int b, int e)
                 {
                     diffXY5(pa_, pb_, gray, kd_q_, b, e);
                 });
        poolRows(rows, [&](int b, int e)
                 {
                     smoothXY5(dx_, dy_, pa_, pb_, ks_q_, b, e);
                 });
    }
    else
    {
        diffXY5(pa_, pb_, gray, kd_q_);
        smoothXY5(dx_, dy_, pa_, pb_, ks_q_);
    }
}

// Canny 边缘提取：幅值(L1) + 非极大值抑制(NMS) + 滞后连通。
//   NMS 以 TG22=13573 定点方向分区（tan22.5°<<15）比较相邻幅值；
//   滞后连通从强边缘洪泛弱候选。梯度取自外部 dx_/dy_。
// nms_map_ 编码：255=边缘，128=弱候选，0=非边缘；强候选经 nms_stack_
// 记录，滞后连通后未与强边缘连通的弱候选保持 128
void FmlDetector::cannyFromGradient(int low, int high)
{
    const int rows = dx_.rows, cols = dx_.cols;
    const int mapstep = cols + 2;

    const int TG22 = 13573;
    nms_map_.create(rows + 2, mapstep, CV_8U);
    memset(nms_map_.data, 0, nms_map_.total());
    nms_stack_.clear();
    nms_stack_.reserve((size_t)rows * cols / 8 + 64);
    const size_t mstep = nms_map_.step;

    // NMS 单点判定
    auto nmsAt = [&](int k, unsigned char *pmap, const short *pdx, const short *pdy,
                     const short *mag_a, const short *mag_p, const short *mag_n,
                     std::vector<unsigned char *> &stk)
    {
        int m = mag_a[k];
        if (m <= low)
            return;
        int xs = pdx[k], ys = pdy[k];
        int x = std::abs(xs);
        int y = std::abs(ys) << 15;
        int tg22x = x * TG22;
        if (y < tg22x)
        {
            // 梯度近水平 → 比较左右邻
            if (m > mag_a[k - 1] && m >= mag_a[k + 1])
            {
                if (m > high)
                {
                    pmap[k] = 255;
                    stk.push_back(pmap + k);
                }
                else
                    pmap[k] = 128;
            }
        }
        else
        {
            int tg67x = tg22x + (x << 16);
            if (y > tg67x)
            {
                // 梯度近垂直 → 比较上下邻
                if (m > mag_p[k] && m >= mag_n[k])
                {
                    if (m > high)
                    {
                        pmap[k] = 255;
                        stk.push_back(pmap + k);
                    }
                    else
                        pmap[k] = 128;
                }
            }
            else
            {
                // 对角方向
                int s = (xs ^ ys) < 0 ? -1 : 1;
                if (m > mag_p[k - s] && m > mag_n[k + s])
                {
                    if (m > high)
                    {
                        pmap[k] = 255;
                        stk.push_back(pmap + k);
                    }
                    else
                        pmap[k] = 128;
                }
            }
        }
    };

    // 幅值行：L1 范数 |dx|+|dy|
    auto magRow = [&](int i, short *dst)
    {
        const short *pdx = dx_.ptr<short>(i);
        const short *pdy = dy_.ptr<short>(i);
        int j = 0;
#if FML_AVX2
        for (; j + 16 <= cols; j += 16)
        {
            __m128i gx0 = _mm_loadu_si128((const __m128i *)(pdx + j));
            __m128i gy0 = _mm_loadu_si128((const __m128i *)(pdy + j));
            __m128i gx1 = _mm_loadu_si128((const __m128i *)(pdx + j + 8));
            __m128i gy1 = _mm_loadu_si128((const __m128i *)(pdy + j + 8));
            _mm_storeu_si128((__m128i *)(dst + j),
                             _mm_add_epi16(_mm_abs_epi16(gx0), _mm_abs_epi16(gy0)));
            _mm_storeu_si128((__m128i *)(dst + j + 8),
                             _mm_add_epi16(_mm_abs_epi16(gx1), _mm_abs_epi16(gy1)));
        }
        if (j < cols)
        {
            // 标量尾：避免向量化读越过行末
            for (; j < cols; ++j)
                dst[j] = (short)(std::abs((int)pdx[j]) + std::abs((int)pdy[j]));
        }
#else
        for (; j < cols; ++j)
            dst[j] = (short)(std::abs((int)pdx[j]) + std::abs((int)pdy[j]));
#endif
        dst[-1] = 0;
        dst[cols] = 0;
    };

    // 单行 NMS：8 像素一组，先筛掉幅值全部低于 low 的块，
    // 再对幸存像素做方向分区与邻居比较
    auto nmsRow = [&](int r, const short *mag_p, const short *mag_a, const short *mag_n,
                      std::vector<unsigned char *> &stk)
    {
        const short *pdx = dx_.ptr<short>(r);
        const short *pdy = dy_.ptr<short>(r);
        unsigned char *pmap = nms_map_.data + (size_t)(r + 1) * mstep + 1;
        int j = 0;
#if FML_AVX2
        const __m256i vlow = _mm256_set1_epi32(low);
        const __m256i vhigh = _mm256_set1_epi32(high);
        const __m256i zero = _mm256_setzero_si256();
        // 16 像素 i16 预筛：整块幅值均 ≤low 则直接跳过
        const __m256i vlow16 = _mm256_set1_epi16(low);
        for (; j + 16 <= cols; j += 16)
        {
            __m256i m16 = _mm256_loadu_si256((const __m256i *)(mag_a + j));
            if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(m16, vlow16)))
                break;
        }
        // 16 像素主循环（两个 8 lane 半区共享 32B 邻域装载与预筛）
        for (; j + 16 <= cols; j += 16)
        {
            __m256i mlo = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_a + j)));
            __m256i mhi = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_a + j + 8)));
            __m256i cand0 = _mm256_cmpgt_epi32(mlo, vlow);
            __m256i cand1 = _mm256_cmpgt_epi32(mhi, vlow);
            if (!_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_or_si256(cand0, cand1))))
                continue; // 整 16 像素块低于 low，无候选
            __m256i xlo16 = _mm256_loadu_si256((const __m256i *)(pdx + j));
            __m256i ylo16 = _mm256_loadu_si256((const __m256i *)(pdy + j));
            __m256i dxv0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(xlo16));
            __m256i dxv1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xlo16, 1));
            __m256i dyv0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ylo16));
            __m256i dyv1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ylo16, 1));
            __m256i x0 = _mm256_abs_epi32(dxv0);
            __m256i x1 = _mm256_abs_epi32(dxv1);
            __m256i y0 = _mm256_slli_epi32(_mm256_abs_epi32(dyv0), 15);
            __m256i y1 = _mm256_slli_epi32(_mm256_abs_epi32(dyv1), 15);
            const __m256i kTG22 = _mm256_set1_epi32(13573);
            __m256i tg22x0 = _mm256_mullo_epi32(x0, kTG22);
            __m256i tg22x1 = _mm256_mullo_epi32(x1, kTG22);
            __m256i tg67x0 = _mm256_add_epi32(tg22x0, _mm256_slli_epi32(x0, 16));
            __m256i tg67x1 = _mm256_add_epi32(tg22x1, _mm256_slli_epi32(x1, 16));

            __m256i mleft = _mm256_loadu_si256((const __m256i *)(mag_a + j - 1));
            __m256i mright = _mm256_loadu_si256((const __m256i *)(mag_a + j + 1));
            __m256i mup = _mm256_loadu_si256((const __m256i *)(mag_p + j));
            __m256i mdown = _mm256_loadu_si256((const __m256i *)(mag_n + j));
            __m256i mupl = _mm256_loadu_si256((const __m256i *)(mag_p + j - 1));
            __m256i mupr = _mm256_loadu_si256((const __m256i *)(mag_p + j + 1));
            __m256i mdnl = _mm256_loadu_si256((const __m256i *)(mag_n + j - 1));
            __m256i mdnr = _mm256_loadu_si256((const __m256i *)(mag_n + j + 1));
            __m256i left0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mleft));
            __m256i left1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mleft, 1));
            __m256i right0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mright));
            __m256i right1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mright, 1));
            __m256i up0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mup));
            __m256i up1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mup, 1));
            __m256i down0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdown));
            __m256i down1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mdown, 1));
            __m256i upl0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mupl));
            __m256i upl1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mupl, 1));
            __m256i upr0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mupr));
            __m256i upr1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mupr, 1));
            __m256i dnl0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdnl));
            __m256i dnl1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mdnl, 1));
            __m256i dnr0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(mdnr));
            __m256i dnr1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(mdnr, 1));

            __m256i isH0 = _mm256_cmpgt_epi32(tg22x0, y0);
            __m256i isH1 = _mm256_cmpgt_epi32(tg22x1, y1);
            __m256i isV0 = _mm256_cmpgt_epi32(y0, tg67x0);
            __m256i isV1 = _mm256_cmpgt_epi32(y1, tg67x1);
            __m256i neg0 = _mm256_cmpgt_epi32(zero, _mm256_xor_si256(dxv0, dyv0));
            __m256i neg1 = _mm256_cmpgt_epi32(zero, _mm256_xor_si256(dxv1, dyv1));
            __m256i nbp0 = _mm256_blendv_epi8(upl0, upr0, neg0);
            __m256i nbp1 = _mm256_blendv_epi8(upl1, upr1, neg1);
            __m256i nbn0 = _mm256_blendv_epi8(dnr0, dnl0, neg0);
            __m256i nbn1 = _mm256_blendv_epi8(dnr1, dnl1, neg1);

            __m256i keepH0 = _mm256_and_si256(isH0, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mlo, left0),
                                                        _mm256_andnot_si256(_mm256_cmpgt_epi32(right0, mlo),
                                                                            _mm256_set1_epi32(-1))));
            __m256i keepH1 = _mm256_and_si256(isH1, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mhi, left1),
                                                        _mm256_andnot_si256(_mm256_cmpgt_epi32(right1, mhi),
                                                                            _mm256_set1_epi32(-1))));
            __m256i keepV0 = _mm256_and_si256(isV0, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mlo, up0),
                                                        _mm256_andnot_si256(_mm256_cmpgt_epi32(down0, mlo),
                                                                            _mm256_set1_epi32(-1))));
            __m256i keepV1 = _mm256_and_si256(isV1, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mhi, up1),
                                                        _mm256_andnot_si256(_mm256_cmpgt_epi32(down1, mhi),
                                                                            _mm256_set1_epi32(-1))));
            __m256i isD0 = _mm256_andnot_si256(_mm256_or_si256(isH0, isV0),
                                               _mm256_set1_epi32(-1));
            __m256i isD1 = _mm256_andnot_si256(_mm256_or_si256(isH1, isV1),
                                               _mm256_set1_epi32(-1));
            __m256i keepD0 = _mm256_and_si256(isD0, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mlo, nbp0), _mm256_cmpgt_epi32(mlo, nbn0)));
            __m256i keepD1 = _mm256_and_si256(isD1, _mm256_and_si256(
                                                        _mm256_cmpgt_epi32(mhi, nbp1), _mm256_cmpgt_epi32(mhi, nbn1)));

            __m256i keep0 = _mm256_and_si256(cand0,
                                             _mm256_or_si256(keepH0, _mm256_or_si256(keepV0, keepD0)));
            __m256i keep1 = _mm256_and_si256(cand1,
                                             _mm256_or_si256(keepH1, _mm256_or_si256(keepV1, keepD1)));
            __m256i strong0 = _mm256_and_si256(keep0, _mm256_cmpgt_epi32(mlo, vhigh));
            __m256i strong1 = _mm256_and_si256(keep1, _mm256_cmpgt_epi32(mhi, vhigh));

            __m256i sel0 = _mm256_or_si256(
                _mm256_and_si256(strong0, _mm256_set1_epi32(255)),
                _mm256_and_si256(keep0, _mm256_set1_epi32(128)));
            __m256i sel1 = _mm256_or_si256(
                _mm256_and_si256(strong1, _mm256_set1_epi32(255)),
                _mm256_and_si256(keep1, _mm256_set1_epi32(128)));
            __m128i packed0 = _mm_packus_epi16(
                _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                    _mm256_packs_epi32(sel0, sel0), 0xD8)),
                _mm_setzero_si128());
            __m128i packed1 = _mm_packus_epi16(
                _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                    _mm256_packs_epi32(sel1, sel1), 0xD8)),
                _mm_setzero_si128());
            _mm_storel_epi64((__m128i *)(pmap + j), packed0);
            _mm_storel_epi64((__m128i *)(pmap + j + 8), packed1);

            unsigned smask0 = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(strong0));
            while (smask0)
            {
                int l = __builtin_ctz(smask0);
                smask0 &= smask0 - 1;
                stk.push_back(pmap + j + l);
            }
            unsigned smask1 = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(strong1));
            while (smask1)
            {
                int l = __builtin_ctz(smask1);
                smask1 &= smask1 - 1;
                stk.push_back(pmap + j + 8 + l);
            }
        }
        for (; j + 8 <= cols; j += 8)
        {
            __m256i m = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_a + j)));
            __m256i cand0 = _mm256_cmpgt_epi32(m, vlow);
            if (!_mm256_movemask_ps(_mm256_castsi256_ps(cand0)))
                continue; // 整块低于 low，无候选，跳过全部方向判定
            __m256i dxv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(pdx + j)));
            __m256i dyv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(pdy + j)));
            __m256i x = _mm256_abs_epi32(dxv);
            __m256i y = _mm256_slli_epi32(_mm256_abs_epi32(dyv), 15);
            // tg22x = x*13573，移位和实现乘法避免溢出
            __m256i tg22x = _mm256_add_epi32(_mm256_slli_epi32(x, 13),
                                             _mm256_add_epi32(_mm256_slli_epi32(x, 12),
                                                              _mm256_add_epi32(_mm256_slli_epi32(x, 10),
                                                                               _mm256_add_epi32(_mm256_slli_epi32(x, 8),
                                                                                                _mm256_add_epi32(_mm256_slli_epi32(x, 2), x)))));
            __m256i tg67x = _mm256_add_epi32(tg22x, _mm256_slli_epi32(x, 16));

            __m256i left = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_a + j - 1)));
            __m256i right = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_a + j + 1)));
            __m256i up = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_p + j)));
            __m256i down = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_n + j)));
            __m256i upl = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_p + j - 1)));
            __m256i upr = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_p + j + 1)));
            __m256i dnl = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_n + j - 1)));
            __m256i dnr = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(mag_n + j + 1)));

            // 扇区：isH = y<tg22x；isV = y>tg67x；其余为对角
            __m256i isH = _mm256_cmpgt_epi32(tg22x, y);
            __m256i isV = _mm256_cmpgt_epi32(y, tg67x);
            // 对角邻居按 (dx^dy) 符号选择：neg → s=-1（取 upr/dnl），否则 s=+1
            __m256i neg = _mm256_cmpgt_epi32(zero, _mm256_xor_si256(dxv, dyv));
            __m256i nbp = _mm256_blendv_epi8(upl, upr, neg);
            __m256i nbn = _mm256_blendv_epi8(dnr, dnl, neg);

            // m >= right ⇔ !(right > m)
            __m256i keepH = _mm256_and_si256(isH, _mm256_and_si256(
                                                      _mm256_cmpgt_epi32(m, left),
                                                      _mm256_andnot_si256(_mm256_cmpgt_epi32(right, m),
                                                                          _mm256_set1_epi32(-1))));
            __m256i keepV = _mm256_and_si256(isV, _mm256_and_si256(
                                                      _mm256_cmpgt_epi32(m, up),
                                                      _mm256_andnot_si256(_mm256_cmpgt_epi32(down, m),
                                                                          _mm256_set1_epi32(-1))));
            __m256i isD = _mm256_andnot_si256(_mm256_or_si256(isH, isV),
                                              _mm256_set1_epi32(-1));
            __m256i keepD = _mm256_and_si256(isD, _mm256_and_si256(
                                                      _mm256_cmpgt_epi32(m, nbp),
                                                      _mm256_cmpgt_epi32(m, nbn)));

            __m256i keep = _mm256_and_si256(cand0,
                                            _mm256_or_si256(keepH, _mm256_or_si256(keepV, keepD)));
            __m256i strong = _mm256_and_si256(keep, _mm256_cmpgt_epi32(m, vhigh));

            // 候选值写 map：sel = strong ? 255 : (keep ? 128 : 0)
            __m256i sel = _mm256_or_si256(
                _mm256_and_si256(strong, _mm256_set1_epi32(255)),
                _mm256_and_si256(keep, _mm256_set1_epi32(128)));
            // pack i32 → u8（permute 修正 lane 交叉），一次写 8 字节
            __m128i packed8 = _mm_packus_epi16(
                _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                    _mm256_packs_epi32(sel, sel), 0xD8)),
                _mm_setzero_si128());
            _mm_storel_epi64((__m128i *)(pmap + j), packed8);

            unsigned smask = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(strong));
            while (smask)
            {
                int l = __builtin_ctz(smask);
                smask &= smask - 1;
                stk.push_back(pmap + j + l);
            }
        }
#endif
        for (; j < cols; ++j)
            nmsAt(j, pmap, pdx, pdy, mag_a, mag_p, mag_n, stk);
    };

    // 3 行幅值滚动缓冲，串行推进：第 i 行幅值就位后立即 NMS 第 i-1 行
    if (nthr_ == 2)
    {
        // 双线程路径：全图幅值缓冲 + mag/NMS 两遍行拆分
        //（输出行不相交、源平面只读）。强点栈拼接后与串行序不同，
        // 但滞后连通终图 = 强像素 8 连通闭包，与遍历序无关
        if ((int)mag_full_.size() < (rows + 2) * mapstep + 2)
            mag_full_.resize((size_t)(rows + 2) * mapstep + 2);
        short *mbase = mag_full_.data();
        auto rowPtr = [&](int r) -> short *
        {
            return mbase + (size_t)(r + 1) * mapstep + 2;
        };
        // 上下哨兵行清零（幅值行自身的哨兵列由 magRow 逐行写）。
        // 段范围 [P-2, P+cols]，末段尾索引 (rows+2)*mapstep 由 +2 余量覆盖
        memset(mbase, 0, (size_t)(mapstep + 1) * sizeof(short));
        memset(mbase + (size_t)(rows + 1) * mapstep, 0,
               (size_t)(mapstep + 1) * sizeof(short));
        const int mid = rows / 2;
        poolRows(rows, [&](int b, int e)
                 {
                     for (int r = b; r < e; ++r)
                         magRow(r, rowPtr(r));
                 });
        nms_stack_wk_.clear();
        poolRows(rows, [&](int b, int e)
                 {
                     for (int r = b; r < e; ++r)
                         nmsRow(r, rowPtr(r - 1), rowPtr(r), rowPtr(r + 1),
                                r < mid ? nms_stack_ : nms_stack_wk_);
                 });

        // 滞后连通（双栈并发）：弱候选认领 = relaxed 原子字节 128→255
        // 单向标记；双写同值/重复入栈只产生冗余遍历，闭包不变
        if (low < high)
        {
            auto floodStack = [&](std::vector<unsigned char *> &stk)
            {
                while (!stk.empty())
                {
                    unsigned char *m = stk.back();
                    stk.pop_back();
                    unsigned char *a = m - mstep;
                    unsigned char *b2 = m + mstep;
                    auto claim = [&](unsigned char *p)
                    {
                        // 弱候选认领为 relaxed 原子字节访问（__atomic 内建，
                        // C++17 兼容；语义与 std::atomic_ref 一致）
                        if (__atomic_load_n(p, __ATOMIC_RELAXED) == 128)
                        {
                            __atomic_store_n(p, (unsigned char)255, __ATOMIC_RELAXED);
                            stk.push_back(p);
                        }
                    };
                    claim(a - 1);
                    claim(a);
                    claim(a + 1);
                    claim(m - 1);
                    claim(m + 1);
                    claim(b2 - 1);
                    claim(b2);
                    claim(b2 + 1);
                }
            };
            FmlPool *fp = FmlPool::instance();
            fp->post(0, 0, [&](int, int)
                     {
                         floodStack(nms_stack_wk_);
                     });
            floodStack(nms_stack_);
            fp->waitDone();
        }
    }
    else
    {
        mag_buf_.assign(3 * (mapstep + 2), 0);
        short *mag_p = mag_buf_.data() + 2;
        short *mag_a = mag_p + mapstep;
        short *mag_n = mag_a + mapstep;
        memset(mag_a - 1, 0, (size_t)mapstep * sizeof(short)); // 顶部哨兵
        for (int i = 0; i < rows; ++i)
        {
            std::swap(mag_n, mag_a);
            std::swap(mag_n, mag_p);
            magRow(i, mag_n);
            if (i > 0)
                nmsRow(i - 1, mag_p, mag_a, mag_n, nms_stack_);
        }
        // 末行：幅值行清零后 NMS 最后一行（magRow 已写 dst[-1]/dst[cols] 哨兵）
        std::swap(mag_n, mag_a);
        std::swap(mag_n, mag_p);
        short *last = mag_n;
        for (int c = -1; c <= cols; ++c)
            last[c] = 0;
        nmsRow(rows - 1, mag_p, mag_a, mag_n, nms_stack_);

        // 滞后连通：强边缘（255）出发洪泛弱候选（128→255）。
        // low >= high 时弱候选集合必为空，可整体跳过
        if (low < high)
        {
            std::vector<unsigned char *> &stk = nms_stack_;
            while (!stk.empty())
            {
                unsigned char *m = stk.back();
                stk.pop_back();
                // 8 邻域逐项判定（哨兵圈保证地址合法）；弱候选置 255 并入栈
                unsigned char *a = m - mstep;
                unsigned char *b = m + mstep;
                if (a[-1] == 128)
                {
                    a[-1] = 255;
                    stk.push_back(a - 1);
                }
                if (a[0] == 128)
                {
                    a[0] = 255;
                    stk.push_back(a);
                }
                if (a[1] == 128)
                {
                    a[1] = 255;
                    stk.push_back(a + 1);
                }
                if (m[-1] == 128)
                {
                    m[-1] = 255;
                    stk.push_back(m - 1);
                }
                if (m[1] == 128)
                {
                    m[1] = 255;
                    stk.push_back(m + 1);
                }
                if (b[-1] == 128)
                {
                    b[-1] = 255;
                    stk.push_back(b - 1);
                }
                if (b[0] == 128)
                {
                    b[0] = 255;
                    stk.push_back(b);
                }
                if (b[1] == 128)
                {
                    b[1] = 255;
                    stk.push_back(b + 1);
                }
            }
            // 残余弱候选保持 128，由下游链种子扫描清零
        }
    }
}

void FmlDetector::extractSegments(const std::vector<cv::Point2i> &points,
                                  std::vector<Segment> &segments)
{
    int total = (int)points.size();

    // 最小二乘所需原始和：8 点一批累加，批内 i32 部分和，flush 时并入
    // int64 总和；flush 时机 = fitNow 调用前
    long long sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    long long cnt = 0;
    int pbx[8], pby[8];
    int pbn = 0;
    // AVX2 路径：8 点块一次水平归约直入 i64 总和
    auto flushPt = [&]()
    {
        if (!pbn)
            return;
        int ax = 0, ay = 0, axx = 0, ayy = 0, axy = 0;
        for (int i = 0; i < pbn; ++i)
        {
            ax += pbx[i];
            ay += pby[i];
            axx += pbx[i] * pbx[i];
            ayy += pby[i] * pby[i];
            axy += pbx[i] * pby[i];
        }
        sx += ax;
        sy += ay;
        sxx += axx;
        syy += ayy;
        sxy += axy;
        cnt += pbn;
        pbn = 0;
    };
    auto addPt = [&](const cv::Point2i &p)
    {
        pbx[pbn] = p.x;
        pby[pbn] = p.y;
        if (++pbn == 8)
        {
#if FML_AVX2
            __m256i vx8 = _mm256_loadu_si256((const __m256i *)pbx);
            __m256i vy8 = _mm256_loadu_si256((const __m256i *)pby);
            sx += hsum256_epi32(vx8);
            sy += hsum256_epi32(vy8);
            sxx += hsum256_epi32(_mm256_mullo_epi32(vx8, vx8));
            syy += hsum256_epi32(_mm256_mullo_epi32(vy8, vy8));
            sxy += hsum256_epi32(_mm256_mullo_epi32(vx8, vy8));
            cnt += 8;
            pbn = 0;
            return;
#endif
            // 标量路径：8 点块累加进 i64 总和
            flushPt();
        }
    };
    // 由原始和直接求拟合线方向：协方差 2x2 半角闭式特征向量 ——
    //   cos t = √((w+u)/2w)、sin t = sign(v)·√((w−u)/2w)，u=dx2−dy2, v=2dxy
    auto fitNow = [&](float &vx, float &vy, float &x0, float &y0)
    {
        double w = (double)cnt;
        double x = (double)sx / w, y = (double)sy / w;
        double dx2 = (double)sxx / w - x * x;
        double dy2 = (double)syy / w - y * y;
        double dxy = (double)sxy / w - x * y;
        const double u = dx2 - dy2;
        const double v = 2 * dxy;
        const double ww = std::sqrt(u * u + v * v);
        if (ww == 0.0)
        { // 各向同性退化：方向取 (1,0)
            vx = 1.0f;
            vy = 0.0f;
        }
        else
        {
            vx = (float)std::sqrt((ww + u) * 0.5 / ww);
            vy = (float)((v >= 0.0 ? 1.0 : -1.0) *
                         std::sqrt((ww - u) * 0.5 / ww));
        }
        x0 = (float)x;
        y0 = (float)y;
    };

    const float fthr = threshold_dist;
    for (int i = 0; i + threshold_length < total; ++i)
    {
        const cv::Point2i ps = points[i];
        cv::Point2i pe = points[i + threshold_length];

        // 首窗距离检查用未归一化叉积直接比较：
        // (l·p)² > fthr²·(l0²+l1²)
        double l[3];
        float ln0 = 0.f, ln1 = 0.f, ln2 = 0.f;
        const double l0i = (double)ps.y - pe.y;
        const double l1i = (double)pe.x - ps.x;
        const double l2i = (double)ps.x * pe.y - (double)pe.x * ps.y;
        const double nrm = (double)(fthr * fthr) * (l0i * l0i + l1i * l1i);

        bool is_line = true;
        l_points_buf.clear();
        l_points_buf.push_back(ps);
        sx = sy = sxx = syy = sxy = 0;
        cnt = 0;
        pbn = 0;
        addPt(ps);

        for (int j = 1; j < threshold_length; ++j)
        {
            const cv::Point2i &pt = points[i + j];
            const double dxp = l0i * pt.x + l1i * pt.y + l2i;
            if (dxp * dxp > nrm)
            {
                is_line = false;
                break;
            }
            l_points_buf.push_back(pt);
            addPt(pt);
        }

        if (!is_line)
            continue;

        l_points_buf.push_back(pe);
        addPt(pe);

        float vx, vy, x0, y0;
        flushPt();
        fitNow(vx, vy, x0, y0);
        lineFromDir(x0, y0, vx, vy, l);
        ln0 = (float)l[0];
        ln1 = (float)l[1];
        ln2 = (float)l[2];
        cv::Point2f e1, e2;

        int j;
        for (j = threshold_length + 1; i + j < total; ++j)
        {
            const cv::Point2i &pt = points[i + j];
            float dist = std::fabs(ln0 * (float)pt.x + ln1 * (float)pt.y + ln2);
            if (dist > fthr)
            {
                flushPt();
                fitNow(vx, vy, x0, y0);
                lineFromDir(x0, y0, vx, vy, l);
                ln0 = (float)l[0];
                ln1 = (float)l[1];
                ln2 = (float)l[2];
                dist = std::fabs(ln0 * (float)pt.x + ln1 * (float)pt.y + ln2);
                if (dist > fthr)
                {
                    --j;
                    break;
                }
            }
            pe = pt;
            l_points_buf.push_back(pt);
            addPt(pt);
        }

        flushPt();
        fitNow(vx, vy, x0, y0);
        lineFromDir(x0, y0, vx, vy, l);

        e1 = cv::Point2f((float)ps.x, (float)ps.y);
        e2 = cv::Point2f((float)pe.x, (float)pe.y);
        incidentPoint(l, e1);
        incidentPoint(l, e2);

        // NFA 显著性 + 主导方向统计融合：单次遍历支撑点，同时统计
        //   m      —— 梯度方向与线垂直的对齐数（NFA 判定用）
        //   gdx    —— 梯度矢量和的 x 分量 Σgx（主导方向：符号即图像左/右）
        // 链内点 y 变化少，行指针缓存避免逐点 ptr() 寻址
        {
            // 期望梯度方向单位向量 e（线段法向），模 π 下 e 与 -e 同判
            const float ang_e = std::atan2(vy, vx) + (float)CV_PI / 2.0f;
            const float ex = std::cos(ang_e), ey = std::sin(ang_e);
            int n = (int)l_points_buf.size();
            int m = 0;
            int gdx = 0;
            int lastY = -1;
            const short *dxr = nullptr;
            const short *dyr = nullptr;
            for (const cv::Point2i &q : l_points_buf)
            {
                if (q.y != lastY)
                {
                    dxr = dx_.ptr<short>(q.y);
                    dyr = dy_.ptr<short>(q.y);
                    lastY = q.y;
                }
                const int gx = dxr[q.x];
                const int gy = dyr[q.x];
                gdx += gx;
                if (isAligned((float)gx, (float)gy, ex, ey))
                    ++m;
            }
            const double l_nfa = logNfa(n, m);
            if (l_nfa > log10_nfa_eps_)
            {
                i = i + j;
                continue; // NFA 不显著，拒绝该线
            }

            // 边缘对几何先验：车道带两边缘反平行，按主导梯度方向对向平移——
            // 向左（Σgx<0）右移 n、向右（Σgx>0）左移 n、Σgx==0 不动；
            // n 可负即方向反转
            const int off = gdx < 0 ? shift_px_ : gdx > 0 ? -shift_px_
                                                          : 0;

            Segment seg;
            seg.x1 = e1.x + (float)off;
            seg.y1 = e1.y;
            seg.x2 = e2.x + (float)off;
            seg.y2 = e2.y;
            seg.conf = (float)(-l_nfa); // 置信度 -log10(NFA)，恒正
            seg.sup_n = n;
            seg.sup_m = m;
            segments.push_back(seg);
        }
        i = i + j;
    }
}

// rows[0..2] 为 y-1/y/y+1 行地址，由调用方随游走更新；
// nms_map_ 哨兵边界保证访问不越界

bool FmlDetector::getPointChainCached(const unsigned char *const rows[3], int x,
                                      int y [[maybe_unused]], cv::Point &chained_pt,
                                      float &direction, int step) const
{
    // y 未使用：行地址由 rows[0..2] 提供
    // 行窗零预读：一次 32 位读覆盖 [x-1, x+1] 三邻位（第 4 字节为
    // 哨兵列，掩掉；游走像素 x∈[1,cols-2] 保证读界安全）。
    // 行窗全零 ⇒ 该行 3 邻位全背景，直接跳过。
    // 枚举序与平局规则与 DIR_IDX 表一致。
    static const signed char DR[8] = {1, 1, 1, 0, -1, -1, -1, 0};
    static const signed char DC[8] = {1, 0, -1, -1, -1, 0, 1, 1};
    const unsigned char *rup = rows[0];
    const unsigned char *rmid = rows[1];
    const unsigned char *rdn = rows[2];
    unsigned wm = 0xFFFFFFFFu, wd = 0xFFFFFFFFu, wu = 0xFFFFFFFFu;
    if (step > 0)
    {
        std::memcpy(&wd, rdn + x - 1, 4);
        wd &= 0x00FFFFFFu; // byte0=x-1, byte1=x, byte2=x+1
        std::memcpy(&wm, rmid + x - 1, 4);
        wm &= 0x00FFFFFFu;
        std::memcpy(&wu, rup + x - 1, 4);
        wu &= 0x00FFFFFFu;
    }

    float min_dir_diff = 7.0f;
    cv::Point consistent_pt;
    int consistent_direction = 0;

    for (int i = 0; i < 8; ++i)
    {
        int ci;
        const unsigned char *rowp;
        if (step > 0)
        {
            if (i <= 2)
            {
                if (wd == 0)
                    continue;
                const unsigned b = (wd >> ((2 - i) * 8)) & 0xFFu; // i0→b2, i1→b1, i2→b0
                if (b == 0)
                    continue;
                rowp = rdn;
                ci = x + 1 - i;
            }
            else if (i == 3)
            {
                if ((wm & 0x000000FFu) == 0)
                    continue;
                rowp = rmid;
                ci = x - 1;
            }
            else if (i <= 6)
            {
                if (wu == 0)
                    continue;
                const unsigned b = (wu >> ((i - 4) * 8)) & 0xFFu; // i4→b0, i5→b1, i6→b2
                if (b == 0)
                    continue;
                rowp = rup;
                ci = x - 1 + (i - 4);
            }
            else
            {
                if ((wm & 0x00FF0000u) == 0)
                    continue;
                rowp = rmid;
                ci = x + 1;
            }
        }
        else
        {
            rowp = DR[i] < 0 ? rup : (DR[i] > 0 ? rdn : rmid);
            ci = x + DC[i];
            if (rowp[ci] == 0)
                continue;
        }

        if (step == 0)
        {
            chained_pt.x = ci;
            chained_pt.y = y + DR[i];
            direction = i > 4 ? (float)(i - 8) : (float)i;
            return true;
        }
        float curr_dir = i > 4 ? (float)(i - 8) : (float)i;
        float dir_diff = std::fabs(curr_dir - direction);
        dir_diff = dir_diff > 4.0f ? 8.0f - dir_diff : dir_diff;
        if (dir_diff <= min_dir_diff)
        {
            min_dir_diff = dir_diff;
            consistent_pt.x = ci;
            consistent_pt.y = y + DR[i];
            consistent_direction = i > 4 ? i - 8 : i;
        }
    }
    if (min_dir_diff < 2.0f)
    {
        chained_pt.x = consistent_pt.x;
        chained_pt.y = consistent_pt.y;
        direction = (direction * (float)step + (float)consistent_direction) / (float)(step + 1);
        return true;
    }
    return false;
}
// 线段提取主流程（[1] 的检测框架，NFA 过滤见 [2]）：
//   逐行扫描 canny 图的 255 种子 → 沿 8 连通游走收集边缘链（方向一致性
//   约束决定走向，边走边清零）→ 每条链提取线段并 NFA 过滤 → 可选两两合并
void FmlDetector::lineDetection(const cv::Mat &src, std::vector<Segment> &segments_all,
                                cv::Mat canny)
{
    imageheight = src.rows;
    imagewidth = src.cols;
    // 跨帧复用缓冲（免每帧分配）
    std::vector<cv::Point2i> &points = ld_points_;
    std::vector<Segment> &segments = ld_segments_;
    std::vector<Segment> &segments_tmp = ld_segments_tmp_;
    points.clear();
    segments.clear();
    segments_tmp.clear(); // 复用缓冲，每帧显式清零
    points.reserve(4096);
    segments.reserve(256);
    segments_tmp.reserve(256);

    for (int r = 0; r < imageheight; ++r)
    {
        unsigned char *crow = canny.ptr(r);
        int c = 0;
        while (c < imagewidth)
        {
            if (crow[c] != 255)
            {
#if FML_AVX2
                // 32 字节块零值跳过；命中块内首个非背景字节（0/128 均视为背景）
                if (c + 32 <= imagewidth)
                {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(crow + c));
                    if (_mm256_testz_si256(v, v))
                    {
                        c += 32;
                        continue;
                    }
                    // 0 与 128（残余弱候选）均视为背景
                    __m256i z = _mm256_setzero_si256();
                    __m256i isz = _mm256_or_si256(_mm256_cmpeq_epi8(v, z),
                                                  _mm256_cmpeq_epi8(v, _mm256_set1_epi8(128)));
                    unsigned nz = ~(unsigned)_mm256_movemask_epi8(isz);
                    if (nz)
                    {
                        c += __builtin_ctz(nz);
                        continue;
                    }
                    c += 32;
                    continue;
                }
#endif
                crow[c] = 0; // 残余弱候选或杂值清零
                ++c;
                continue;
            }

            cv::Point2i pt(c, r);
            points.push_back(pt);
            crow[c] = 0;

            float direction = 0.0f;
            int step = 0;
            {
                // 哨兵行保证 r-1 / r+1 行总可读（nms_map_ 外圈恒 0）
                const size_t sb = canny.step;
                unsigned char *base = canny.ptr(0);
                unsigned char *rows[3] = {
                    base + (size_t)(r - 1) * sb,
                    base + (size_t)r * sb,
                    base + (size_t)(r + 1) * sb};
                int cy = r;
                while (getPointChainCached(rows, pt.x, cy, pt, direction, step))
                {
                    points.push_back(pt);
                    ++step;
                    if (pt.y != cy)
                    {
                        if (pt.y > cy)
                        {
                            rows[0] = rows[1];
                            rows[1] = rows[2];
                            rows[2] += sb;
                        }
                        else
                        {
                            rows[2] = rows[1];
                            rows[1] = rows[0];
                            rows[0] -= sb;
                        }
                        cy = pt.y;
                    }
                    rows[1][pt.x] = 0;
                }
            }

            if (points.size() < (unsigned int)threshold_length + 1)
            {
                points.clear();
                continue;
            }

            extractSegments(points, segments);
            if (segments.empty())
            {
                points.clear();
                continue;
            }

            for (size_t i = 0; i < segments.size(); ++i)
            {
                Segment seg = segments[i];
                float length = std::sqrt((seg.x1 - seg.x2) * (seg.x1 - seg.x2) +
                                         (seg.y1 - seg.y2) * (seg.y1 - seg.y2));
                if (length < threshold_length)
                    continue;
                if ((seg.x1 <= 5.0f && seg.x2 <= 5.0f) ||
                    (seg.y1 <= 5.0f && seg.y2 <= 5.0f) ||
                    (seg.x1 >= imagewidth - 5.0f && seg.x2 >= imagewidth - 5.0f) ||
                    (seg.y1 >= imageheight - 5.0f && seg.y2 >= imageheight - 5.0f))
                    continue;
                additionalOperationsOnSegment(src, seg);
                if (!do_merge)
                    segments_all.push_back(seg);
                segments_tmp.push_back(seg);
            }
            points.clear();
            segments.clear();
            ++c; // 种子处理完，继续扫描本行
        }
    }

    if (!do_merge)
        return; // 非 merge 模式已在上方填充 segments_all

    mergeAndCollect(src, segments_tmp, segments_all);
}
} // namespace fml
