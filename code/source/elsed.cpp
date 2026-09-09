// 项目名: fml_for_x86_64
// 文件名: elsed.cpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// ELSED 管线前端（fml_detect_elsed，算法出自 [3]，边缘绘制机制沿袭 [4]）：
//   Gauss5x5(σ) + Sobel3 → L1 幅值阈值化 gImg + 2 类方向图 →
//   锚点扫描（间隔采样、阈值逐级减半）→ 边缘绘制双向游走
//   （梯度幅值图 3 邻域择优路由 + 增量最小二乘拟合 + 交叉点跳越分支栈）→
//   段角度验证（重投影双线性梯度角）
// 验证后接入共享后段（长度/贴边过滤、先验平移、方向统一、共线合并，
// 实现于 detector.cpp）。
//
// 参考文献：
//   [3] Suárez et al., ELSED: Enhanced Line SEgment Drawing, Pattern Recognition 2022.
//   [4] Akinlar & Topal, EDLines, Pattern Recognition Letters 2011.

#include "detector.hpp"

#include <opencv2/imgproc.hpp>

namespace fml
{
namespace
{
// 方向类（2 类量化：dirImg = |dx|>=|dy| ? VERTICAL : HORIZONTAL）
constexpr unsigned char ESED_EDGE_VERTICAL = 0;
constexpr unsigned char ESED_EDGE_HORIZONTAL = 255;

// 边缘状态图取值
constexpr unsigned char ESED_ED_NO_EDGE_PX = 0;
constexpr unsigned char ESED_ED_EDGE_PX = 255;
constexpr unsigned char ESED_ED_ANCHOR_PX = 204;
constexpr unsigned char ESED_ED_INLIER_PX = 153;
constexpr unsigned char ESED_ED_OUTLIER_PX = 102;
constexpr unsigned char ESED_ED_JUNCTION_PX = 152;

// 行进方向
constexpr unsigned char ESED_LEFT = 1, ESED_RIGHT = 2, ESED_UP = 3, ESED_DOWN = 4;

// EdgeDrawer 常量
constexpr int ESED_SKIP_EDGE_PT = 2;    // 窗口前移步长
constexpr int ESED_MAX_OUTLIERS_TH = 3; // 外点数上限

// 交叉点跳越内部判据（固定值）
constexpr double kElsedJunctionEigenvalsTh = 10; // junctionEigenvalsTh
constexpr double kElsedJunctionAngleTh = 10.0 * (CV_PI / 180.0);
const int kElsedJunctionSizes[] = {5, 7, 9}; // listJunctionSizes
} // namespace（文件内部常量）

void ElsedSegment::init(const std::vector<cv::Point> &pts, int startIdx)
{
    dx = pts.back().x - pts[startIdx].x;
    dy = pts.back().y - pts[startIdx].y;
    isHorizontal = std::abs(dx) >= std::abs(dy);

    leastSquareLineFit(pts, startIdx);

    firstPx = pts[startIdx];
    lastPx = pts.back();
    prevFirstPx = pts[startIdx + 1];
    prevLastPx = pts[pts.size() - 2];
    firstPxIndex = startIdx;
    lastPxIndex = (int)pts.size() - 1;

    arePixelsSorted = true;
    firstEndpointExtended = false;
    secondEndpointExtended = false;
}

void ElsedSegment::skipPositions()
{
    // 窗口整体后移 SKIP_EDGE_PT(=2)：老端点移出模型、新端点移入
    subtractPointFromModel(firstPx);
    subtractPointFromModel(prevFirstPx);

    firstPxIndex += ESED_SKIP_EDGE_PT;
    lastPxIndex += ESED_SKIP_EDGE_PT;
    firstPx = (*pixels)[firstPxIndex];
    lastPx = pixels->back();
    prevFirstPx = (*pixels)[firstPxIndex + 1];
    prevLastPx = (*pixels)[lastPxIndex - 1];

    dx = lastPx.x - firstPx.x;
    dy = lastPx.y - firstPx.y;
    const bool tmp = std::abs(dx) >= std::abs(dy);
    if (tmp != isHorizontal)
    {
        isHorizontal = tmp;
        leastSquareLineFit(*pixels, firstPxIndex); // 拟合方向翻转，整体重算
    }
    else
    {
        leastSquaresLineFitNewPoint(prevLastPx.x, prevLastPx.y);
        leastSquaresLineFitNewPoint(lastPx.x, lastPx.y);
    }
}

void ElsedSegment::addPixel(int x, int y, int pixelIndexInEdge, bool isPixelAtTheEnd)
{
    leastSquaresLineFitNewPoint(x, y);

    if (isPixelAtTheEnd)
    {
        prevLastPx = lastPx;
        lastPx.x = x;
        lastPx.y = y;
    }
    else
    {
        prevFirstPx = firstPx;
        firstPx.x = x;
        firstPx.y = y;
    }

    // 像素链虽然两端都可能生长，但内存上总是连续追加，最后加入的即链尾
    lastPxIndex = pixelIndexInEdge;
    arePixelsSorted = arePixelsSorted && isPixelAtTheEnd;
}

void ElsedSegment::finish()
{
    ensureSigned();
    calcSegmentEndpoints();
}

void ElsedSegment::removeLastPx(bool removeFromTheEnd)
{
    subtractPointFromModel(removeFromTheEnd ? lastPx : firstPx);
    if (removeFromTheEnd)
    {
        lastPx = prevLastPx;
        prevLastPx.x = -1;
        prevLastPx.y = -1;
    }
    else
    {
        firstPx = prevFirstPx;
        prevFirstPx.x = -1;
        prevFirstPx.y = -1;
    }
    --lastPxIndex;
}

void ElsedSegment::leastSquareLineFit(const std::vector<cv::Point> &pts, int startIdx)
{
    sum_x_i = 0;
    sum_y_i = 0;
    sum_x_i_y_i = 0;
    sum_x_i_2 = 0;
    N = (uint32_t)(pts.size() - startIdx);
    if (isHorizontal)
    {
        for (int i = startIdx; i < (int)pts.size(); ++i)
        {
            const int indpCoord = pts[i].x, depCoord = pts[i].y;
            sum_x_i += indpCoord;
            sum_y_i += depCoord;
            sum_x_i_2 += indpCoord * indpCoord;
            sum_x_i_y_i += indpCoord * depCoord;
        }
    }
    else
    {
        for (int i = startIdx; i < (int)pts.size(); ++i)
        {
            const int indpCoord = pts[i].y, depCoord = pts[i].x;
            sum_x_i += indpCoord;
            sum_y_i += depCoord;
            sum_x_i_2 += indpCoord * indpCoord;
            sum_x_i_y_i += indpCoord * depCoord;
        }
    }
    calculateLineEq();
}

void ElsedSegment::leastSquaresLineFitNewPoint(int x, int y)
{
    const int indpCoord = isHorizontal ? x : y;
    const int depCoord = isHorizontal ? y : x;
    ++N;
    sum_x_i += indpCoord;
    sum_y_i += depCoord;
    sum_x_i_2 += indpCoord * indpCoord;
    sum_x_i_y_i += indpCoord * depCoord;
    calculateLineEq();
}

void ElsedSegment::subtractPointFromModel(const cv::Point &p)
{
    const int independentCord = isHorizontal ? p.x : p.y;
    const int dependentCord = isHorizontal ? p.y : p.x;
    --N;
    sum_x_i -= independentCord;
    sum_y_i -= dependentCord;
    sum_x_i_2 -= independentCord * independentCord;
    sum_x_i_y_i -= independentCord * dependentCord;
    calculateLineEq();
}

void ElsedSegment::calculateLineEq()
{
    // ax+by+c=0：a = N·Σxy − Σx·Σy, b = (Σx)² − N·Σx², c = Σy·Σx² − Σx·Σxy
    //（x/y 依拟合方向互换）；int64 内环精确，出口 float 域归一化
    const float cf = (float)(sum_y_i * sum_x_i_2 - sum_x_i * sum_x_i_y_i);
    float af, bf;
    if (isHorizontal)
    {
        af = (float)((int64_t)N * sum_x_i_y_i - sum_x_i * sum_y_i);
        bf = (float)(sum_x_i * sum_x_i - (int64_t)N * sum_x_i_2);
    }
    else
    {
        bf = (float)((int64_t)N * sum_x_i_y_i - sum_x_i * sum_y_i);
        af = (float)(sum_x_i * sum_x_i - (int64_t)N * sum_x_i_2);
    }
    const float inv = rsqrtF(af * af + bf * bf);
    equation[0] = af * inv;
    equation[1] = bf * inv;
    equation[2] = cf * inv;
    // 符号统一惰性化：isInlier/getFitError 只用 |dist|，与符号无关；
    // 需要符号语义的场合（分支方向/端点投影）调用 ensureSigned()
}

// 符号统一（使方程沿 (dx,dy) 方向为正）。仅在消费符号语义时调用：
// 分支事件（directionFromLineEq）与段完结（calcSegmentEndpoints）
void ElsedSegment::ensureSigned()
{
    if (dx * -equation[1] + dy * equation[0] < 0)
    {
        equation[0] = -equation[0];
        equation[1] = -equation[1];
        equation[2] = -equation[2];
    }
}

void ElsedSegment::calcSegmentEndpoints()
{
    // 首末像素投影到拟合线：xp = w2²·x0 − w1·w2·y0 − w3·w1
    const float a1 = equation[1] * equation[1];
    const float a2 = equation[0] * equation[0];
    const float a3 = equation[0] * equation[1];
    const float a4 = equation[2] * equation[0];
    const float a5 = equation[2] * equation[1];
    {
        const int Px = firstPx.x, Py = firstPx.y;
        endpoints[0] = a1 * Px - a3 * Py - a4;
        endpoints[1] = a2 * Py - a3 * Px - a5;
    }
    {
        const int Px = lastPx.x, Py = lastPx.y;
        endpoints[2] = a1 * Px - a3 * Py - a4;
        endpoints[3] = a2 * Py - a3 * Px - a5;
    }
}

// 方向的反向
inline uint8_t elsedInverseDirection(uint8_t dir)
{
    switch (dir)
    {
    case ESED_RIGHT:
        return ESED_LEFT;
    case ESED_LEFT:
        return ESED_RIGHT;
    case ESED_UP:
        return ESED_DOWN;
    default:
        return ESED_UP; // ESED_DOWN
    }
}

// 由线方程导出行进方向
inline uint8_t elsedDirectionFromLineEq(const cv::Vec3f &eq)
{
    const float lineX = -eq[1];
    const float lineY = eq[0];
    if (std::fabs(lineX) > std::fabs(lineY))
    {
        return lineX > 0 ? ESED_RIGHT : ESED_LEFT;
    }
    return lineY > 0 ? ESED_DOWN : ESED_UP;
}

// 无初始行进像素时按方向虚拟回退一步
inline cv::Point elsedCalcLastPixelWithDirection(const cv::Point &px, uint8_t lastDir)
{
    switch (lastDir)
    {
    case ESED_UP:
        return {px.x, px.y + 1};
    case ESED_DOWN:
        return {px.x, px.y - 1};
    case ESED_LEFT:
        return {px.x + 1, px.y};
    case ESED_RIGHT:
        return {px.x - 1, px.y};
    default:
        return {-1, -1};
    }
}

// 周期角距离
inline double elsedCircularDist(double valueA, double valueB, double mod)
{
    double a, b;
    if (valueA < valueB)
    {
        a = valueA;
        b = valueB;
    }
    else
    {
        a = valueB;
        b = valueA;
    }
    const double dist_clockwise = b - a;
    const double dist_no_clockwise = a + (mod - b);
    return std::min(dist_clockwise, dist_no_clockwise);
}

// 线段与 X 轴夹角，值域 (−π/2, π/2]
inline float elsedSegAngle(const cv::Vec4f &s)
{
    if (s[2] > s[0])
        return std::atan2(s[3] - s[1], s[2] - s[0]);
    return std::atan2(s[1] - s[3], s[0] - s[2]);
}

// 线性/双线性插值
inline float elsedLerp(float s, float e, float t)
{
    return s + (e - s) * t;
}
inline float elsedBlerp(float c00, float c10, float c01, float c11, float tx, float ty)
{
    return elsedLerp(elsedLerp(c00, c10, tx), elsedLerp(c01, c11, tx), ty);
}

// 点到直线的投影
inline cv::Point2f elsedGetProjectionPtn(const cv::Vec3f &l, const cv::Point2f &p)
{
    const cv::Vec3f homoP(p.x, p.y, 1);
    if (l.dot(homoP) == 0)
        return p;
    // l 的方向为 (-l.b, l.a)，其旋转 90° 后为 (l.a, l.b)
    const cv::Vec2f v2(l[0], l[1]);
    const cv::Vec3f r2(v2[1], -v2[0], v2[0] * p.y - v2[1] * p.x);
    const cv::Vec3f p2 = l.cross(r2);
    return cv::Point2f(p2[0] / p2[2], p2[1] / p2[2]);
}

// Bresenham 直线栅格，结果写入复用缓冲
inline void elsedBresenham(int x0, int y0, int x1, int y1,
                           std::vector<cv::Point> &pixels)
{
    pixels.clear();
    int dx = x1 - x0, dy = y1 - y0;
    const int xIncrement = dx < 0 ? -1 : +1;
    const int yIncrement = dy < 0 ? -1 : +1;
    int x = x0, y = y0;
    dx = std::abs(dx);
    dy = std::abs(dy);

    if (dx >= dy)
    {
        int p = 2 * dy - dx;
        while (x != x1)
        {
            pixels.emplace_back(x, y);
            if (p >= 0)
            {
                y += yIncrement;
                p += 2 * dy - 2 * dx;
            }
            else
            {
                p += 2 * dy;
            }
            x += xIncrement;
        }
    }
    else
    {
        int p = 2 * dx - dy;
        while (y != y1)
        {
            pixels.emplace_back(x, y);
            if (p >= 0)
            {
                x += xIncrement;
                p += 2 * dx - 2 * dy;
            }
            else
            {
                p += 2 * dx;
            }
            y += yIncrement;
        }
    }
    pixels.emplace_back(x1, y1);
}

// 梯度：Gauss5x5(σ) → Sobel3（BORDER_REPLICATE）→
// gImg = |dx|+|dy|（< gradientThreshold 清零）、dirImg = 2 类方向图
// 高斯为可分离整数卷积：Gauss5x5 定点核 kq=cvRound(k·256)，
// 垂直→水平，全程 i32 无中间舍入，末端一次 (v + 2^15) >> 16，边界
// REFLECT_101；Sobel(3) 纯整数卷积，无舍入，边界 REPLICATE。
static inline int elsedRefl101(int i, int n)
{
    if (i < 0)
        return -i;
    if (i >= n)
        return 2 * (n - 1) - i;
    return i;
}

// 垂直 5 tap 单行（u8 → u16，scale 256）：dst[c] = Σ kq[i]·src[i][c]。
// 行指针由调用方按 REFLECT_101 钳位后传入
static void elsedBlurVRow(unsigned short *dst, const unsigned char *const src[5],
                          const int kq[5], int cols)
{
    int c = 0;
#if FML_AVX2
    const __m256i k02 = _mm256_set1_epi32(kq[0]);
    const __m256i k13 = _mm256_set1_epi32(kq[1]);
    const __m256i kc = _mm256_set1_epi32(kq[2]);
    for (; c + 16 <= cols; c += 16)
    {
        __m256i plo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[0] + c)));
        __m256i phi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[0] + c + 8)));
        __m256i v = _mm256_mullo_epi32(_mm256_add_epi32(plo,
                                                        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[4] + c)))),
                                       k02);
        __m256i v2 = _mm256_mullo_epi32(_mm256_add_epi32(phi,
                                                         _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[4] + c + 8)))),
                                        k02);
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(_mm256_add_epi32(
                                                       _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[1] + c))),
                                                       _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[3] + c)))),
                                                   k13));
        v2 = _mm256_add_epi32(v2, _mm256_mullo_epi32(_mm256_add_epi32(
                                                         _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[1] + c + 8))),
                                                         _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[3] + c + 8)))),
                                                     k13));
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(
                                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[2] + c))), kc));
        v2 = _mm256_add_epi32(v2, _mm256_mullo_epi32(
                                      _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src[2] + c + 8))), kc));
        __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v2), 0xD8);
        _mm256_storeu_si256((__m256i *)(dst + c), pk);
    }
#endif
    for (; c < cols; ++c)
        dst[c] = (unsigned short)(kq[0] * src[0][c] + kq[1] * src[1][c] + kq[2] * src[2][c] + kq[3] * src[3][c] + kq[4] * src[4][c]);
}

// t1（垂直模糊）行流式生成器：5 行环形缓冲，请求行号非递减、
// 每行至多计算一次
struct ElsedT1Gen
{
    const cv::Mat *gray;
    std::vector<unsigned short> ring; // 5 × cols
    const int *kq;
    int rows, cols;
    int ready; // 最高已产出行（初始 -1）
    void init(const cv::Mat *g, const int k[5])
    {
        gray = g;
        kq = k;
        rows = g->rows;
        cols = g->cols;
        ring.assign((size_t)5 * cols, 0);
        ready = -1;
    }
    const unsigned short *get(int sr)
    {
        if (sr < 0)
            sr = 0;
        if (sr >= rows)
            sr = rows - 1;
        if (ready < sr)
        {
            const int n = gray->rows;
            for (int r = ready + 1; r <= sr; ++r)
            {
                const unsigned char *ps[5] = {
                    gray->ptr<unsigned char>(elsedRefl101(r - 2, n)),
                    gray->ptr<unsigned char>(elsedRefl101(r - 1, n)),
                    gray->ptr<unsigned char>(r),
                    gray->ptr<unsigned char>(elsedRefl101(r + 1, n)),
                    gray->ptr<unsigned char>(elsedRefl101(r + 2, n))};
                elsedBlurVRow(&ring[(size_t)(r % 5) * cols], ps, kq, cols);
            }
            ready = sr;
        }
        return &ring[(size_t)(sr % 5) * cols];
    }
};

// 水平 5 tap 单行（u16 → u8，(v + 2^15) >> 16，REFLECT_101），
// 供融合 pass 逐行按需产模糊行
static void elsedBlurHRow(unsigned char *dp, const unsigned short *sp,
                          const int kq[5], int cols)
{
#if FML_AVX2
    int c = 2; // 窗口 [c-2, c+2] 全程在行内，避免越界读
    const __m256i k02 = _mm256_set1_epi32(kq[0]);
    const __m256i k13 = _mm256_set1_epi32(kq[1]);
    const __m256i kc = _mm256_set1_epi32(kq[2]);
    const __m256i vrnd = _mm256_set1_epi32(1 << 15);
    for (; c + 10 <= cols; c += 8)
    {
        __m256i p0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(sp + c - 2)));
        __m256i p1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(sp + c - 1)));
        __m256i p2 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(sp + c)));
        __m256i p3 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(sp + c + 1)));
        __m256i p4 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(sp + c + 2)));
        __m256i v = _mm256_mullo_epi32(_mm256_add_epi32(p0, p4), k02);
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(_mm256_add_epi32(p1, p3), k13));
        v = _mm256_add_epi32(v, _mm256_mullo_epi32(p2, kc));
        v = _mm256_srai_epi32(_mm256_add_epi32(v, vrnd), 16);
        __m128i p16 = _mm_packus_epi16(
            _mm256_castsi256_si128(_mm256_permute4x64_epi64(
                _mm256_packus_epi32(v, v), 0xD8)),
            _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(dp + c), p16);
    }
    for (int bc = 0; bc < 2 && bc < cols; ++bc)
    {
        int v = 0;
        for (int j = 0; j < 5; ++j)
            v += kq[j] * sp[elsedRefl101(bc + j - 2, cols)];
        dp[bc] = (unsigned char)((v + (1 << 15)) >> 16);
    }
    for (int bc = c; bc < cols; ++bc)
    {
        int v = 0;
        for (int j = 0; j < 5; ++j)
            v += kq[j] * sp[elsedRefl101(bc + j - 2, cols)];
        dp[bc] = (unsigned char)((v + (1 << 15)) >> 16);
    }
#else
    for (int c = 0; c < cols; ++c)
    {
        int v = 0;
        for (int j = 0; j < 5; ++j)
            v += kq[j] * sp[elsedRefl101(c + j - 2, cols)];
        dp[c] = (unsigned char)((v + (1 << 15)) >> 16);
    }
#endif
}

// 融合 pass：水平模糊与 Sobel/gImg/dirImg 单次遍历。
// 滚动缓冲按绝对行号惰性产模糊行（请求严格递增）
static void elsedBlurSobel(cv::Mat &dx, cv::Mat &dy, cv::Mat &g, cv::Mat &dir,
                           ElsedT1Gen &t1gen, const cv::Mat &grayPass,
                           bool haveBlur, const int kq[5], short gradientTh,
                           int r0, int r1)
{
    const int rows = grayPass.rows, cols = grayPass.cols;
    if (r1 < 0)
        r1 = rows;
    std::vector<unsigned char> brow((size_t)3 * cols); // 模糊行环形缓冲
    unsigned char *br[3] = {brow.data(), brow.data() + cols, brow.data() + 2 * (size_t)cols};
    int brRow[3] = {-1, -1, -1};
    // 按需产模糊行（σ<=0 时直接用灰度行）；请求严格递增
    auto blurGet = [&](int sr) -> const unsigned char *
    {
        const int s = sr < 0 ? 0 : (sr >= rows ? rows - 1 : sr);
        const int slot = s % 3;
        if (brRow[slot] != s)
        {
            if (haveBlur)
                elsedBlurHRow(br[slot], t1gen.get(s), kq, cols);
            else
                memcpy(br[slot], grayPass.ptr<unsigned char>(s), (size_t)cols);
            brRow[slot] = s;
        }
        return br[slot];
    };

    std::vector<short> t2s((size_t)3 * cols), t3s((size_t)3 * cols);
    short *s[3] = {t2s.data(), t2s.data() + cols, t2s.data() + 2 * (size_t)cols};
    short *s3[3] = {t3s.data(), t3s.data() + cols, t3s.data() + 2 * (size_t)cols};

    // 模糊行 sr 的横向差分（t2 缓冲）与 [1,2,1] 横向和（t3 缓冲）
    auto calcRow = [&](short *d2, short *d3, int sr)
    {
        const unsigned char *sp = blurGet(sr);
        int c = 1; // 窗口 [c-1, c+1] 行内；边界列走尾部
#if FML_AVX2
        for (; c + 17 <= cols; c += 16)
        {
            __m256i l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c - 1)));
            __m256i m = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c)));
            __m256i rt = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(sp + c + 1)));
            _mm256_storeu_si256((__m256i *)(d2 + c), _mm256_sub_epi16(rt, l));
            _mm256_storeu_si256((__m256i *)(d3 + c),
                                _mm256_add_epi16(_mm256_add_epi16(l, rt), _mm256_add_epi16(m, m)));
        }
#endif
        for (; c <= cols - 2; ++c)
        {
            d2[c] = (short)(sp[c + 1] - sp[c - 1]);
            d3[c] = (short)(sp[c - 1] + 2 * sp[c] + sp[c + 1]);
        }
        if (cols > 1)
        {
            d2[0] = (short)(sp[1] - sp[0]);
            d3[0] = (short)(3 * sp[0] + sp[1]);
            d2[cols - 1] = (short)(sp[cols - 1] - sp[cols - 2]);
            d3[cols - 1] = (short)(sp[cols - 2] + 3 * sp[cols - 1]);
        }
        else
        {
            d2[0] = 0;
            d3[0] = (short)(4 * sp[0]);
        }
    };

    // 由三行 t2/t3 产出第 r 行的 dx/dy/g/dir
    auto emitRow = [&](int r, const short *u2, const short *m2, const short *d2r,
                       const short *u3, const short *d3r)
    {
        short *ox = dx.ptr<short>(r);
        short *oy = dy.ptr<short>(r);
        short *gr = g.ptr<short>(r);
        unsigned char *dr = dir.ptr<unsigned char>(r);
        int c = 0;
#if FML_AVX2
        const __m256i vth = _mm256_set1_epi16(gradientTh);
        const __m256i v255 = _mm256_set1_epi16(0x00FF);
        for (; c + 16 <= cols; c += 16)
        {
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(u2 + c));
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(m2 + c));
            __m256i a2 = _mm256_loadu_si256((const __m256i *)(d2r + c));
            __m256i xv = _mm256_add_epi16(_mm256_add_epi16(a0, a2),
                                          _mm256_add_epi16(a1, a1));
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(u3 + c));
            __m256i b2 = _mm256_loadu_si256((const __m256i *)(d3r + c));
            __m256i yv = _mm256_sub_epi16(b2, b0);
            __m256i ax = _mm256_abs_epi16(xv);
            __m256i ay = _mm256_abs_epi16(yv);
            __m256i sum = _mm256_add_epi16(ax, ay); // ≤ 8160，i16 无溢出
            __m256i smallv = _mm256_cmpgt_epi16(vth, sum);
            __m256i gv = _mm256_andnot_si256(smallv, sum);
            // 方向：|dx|>=|dy| → VERTICAL(0)；|dy>|dx| → HORIZONTAL(255)
            __m256i hmask = _mm256_cmpgt_epi16(ay, ax);
            __m256i dirv = _mm256_and_si256(hmask, v255);
            _mm256_storeu_si256((__m256i *)(ox + c), xv);
            _mm256_storeu_si256((__m256i *)(oy + c), yv);
            _mm256_storeu_si256((__m256i *)(gr + c), gv);
            // 16×i16 → 16×u8
            _mm_storeu_si128((__m128i *)(dr + c),
                             _mm_packus_epi16(_mm256_castsi256_si128(dirv),
                                              _mm256_extracti128_si256(dirv, 1)));
        }
#endif
        for (; c < cols; ++c)
        {
            const short xv = (short)(u2[c] + 2 * m2[c] + d2r[c]);
            const short yv = (short)(d3r[c] - u3[c]);
            const int16_t adx = (int16_t)(xv >= 0 ? xv : -xv);
            const int16_t ady = (int16_t)(yv >= 0 ? yv : -yv);
            const int16_t sum = (int16_t)(adx + ady);
            ox[c] = xv;
            oy[c] = yv;
            gr[c] = sum < gradientTh ? (short)0 : sum;
            dr[c] = adx >= ady ? ESED_EDGE_VERTICAL : ESED_EDGE_HORIZONTAL;
        }
    };

    if (rows == 1)
    {
        if (r0 == 0 && r1 > 0)
        {
            calcRow(s[0], s3[0], 0);
            emitRow(0, s[0], s[0], s[0], s3[0], s3[0]);
        }
        return;
    }

    // 区间驱动（blurGet 使请求按行号递增且每行至多产一次）
    int iUp, iMid, iFree, rStart;
    if (r0 == 0)
    {
        calcRow(s[0], s3[0], 0);
        calcRow(s[1], s3[1], 1);
        emitRow(0, s[0], s[0], s[1], s3[0], s3[1]);
        iUp = 0;
        iMid = 1;
        iFree = 2;
        rStart = 1;
    }
    else
    {
        calcRow(s[0], s3[0], r0 - 1);
        calcRow(s[1], s3[1], r0);
        if (r0 + 1 < rows)
        {
            calcRow(s[2], s3[2], r0 + 1);
            emitRow(r0, s[0], s[1], s[2], s3[0], s3[2]);
            iUp = 1;
            iMid = 2;
            iFree = 0;
        }
        else
        {
            emitRow(r0, s[0], s[1], s[1], s3[0], s3[1]);
            iUp = 0;
            iMid = 1;
            iFree = 2;
        }
        rStart = r0 + 1;
    }
    for (int r = rStart; r < r1; ++r)
    {
        const bool hasDown = (r + 1 < rows);
        if (hasDown)
            calcRow(s[iFree], s3[iFree], r + 1);
        emitRow(r, s[iUp], s[iMid], s[hasDown ? iFree : iMid],
                s3[iUp], s3[hasDown ? iFree : iMid]);
        if (hasDown)
        {
            const int nu = iMid, nm = iFree, nf = iUp;
            iUp = nu;
            iMid = nm;
            iFree = nf;
        }
    }
}

// ELSED 梯度：Gauss5x5 → Sobel3 → gImg/dirImg（els_dx_/els_dy_/els_g_/els_dir_）
void FmlDetector::elsedComputeGradients(const cv::Mat &gray)
{
    const int rows = gray.rows, cols = gray.cols;
    els_w_ = cols;
    els_h_ = rows;

    // 5 点高斯定点核：cv::getGaussianKernel(5,σ) × 256 四舍五入；
    // σ<=0 用 δ 核（跳过平滑）
    cv::Mat gk;
    if (ec_.sigma > 0.0)
        gk = cv::getGaussianKernel(5, ec_.sigma);
    int kq[5] = {0, 0, 256, 0, 0}; // σ<=0：δ 核（无平滑）
    if (ec_.sigma > 0.0)
        for (int i = 0; i < 5; ++i)
            kq[i] = cvRound(gk.at<double>(i) * 256.0);

    // 垂直模糊按需流式产行（5 行环形）+ 水平模糊与 Sobel/gImg/dirImg
    // 单 pass 融合，不整面落中间平面；sigma<=0 时融合 pass 直接用灰度行
    els_dx_.create(rows, cols, CV_16S);
    els_dy_.create(rows, cols, CV_16S);
    els_g_.create(rows, cols, CV_16S);
    els_dir_.create(rows, cols, CV_8U);
    const short gradientTh = (short)ec_.grad_th; // float→short 传参
    const bool haveBlur = ec_.sigma > 0.0;
    ElsedT1Gen t1gen;
    if (haveBlur)
        t1gen.init(&gray, kq);
    elsedBlurSobel(els_dx_, els_dy_, els_g_, els_dir_, t1gen, gray,
                   haveBlur, kq, gradientTh, 0, rows);
}

// 锚点扫描：列优先、scanIntervals 间隔采样；
// 锚点 = 幅值高出垂直于边方向两邻居 anchorThresh 以上的像素。
// gImg 已阈值化（0 = 非边缘候选，直接跳过）
void FmlDetector::elsedComputeAnchorPoints(std::vector<cv::Point> &anchorPoints,
                                           int anchorThresh,
                                           int w0, int w1) const
{
    const int imageWidth = els_w_, imageHeight = els_h_;
    const int scanInterval = ec_.scan_intv; // 扫描间隔
    const short *gradImg = els_gImg_;
    const unsigned char *dirImg = els_dirImg_;

    if (w1 < 0)
        w1 = imageWidth - 2;
    const unsigned int pixelNum = imageWidth * imageHeight;
    // 容量按扫描密度一次预留，之后仅 push_back
    const size_t need = (size_t)((double)pixelNum / (2.5 * scanInterval)) + 1;
    if (anchorPoints.capacity() < need)
        anchorPoints.reserve(need);
    anchorPoints.clear();

    int nAnchors = 0;
    const int hLast = imageHeight - 2; //（w ≤ W-2，h ≤ H-2）

#if FML_AVX2
    // 列内 8 行一块的 gather 向量化（枚举序：w 外层升序 × h 内层升序）。
    // i32 gather 以 scale=2 读 gImg 相邻两元素，dir 以 scale=1 读 4 字节。
    // 列间判定相互独立（只读同列垂直邻居），支持 w 区间拆分
    {
        const __m256i vth = _mm256_set1_epi32(anchorThresh);
        const __m256i vzero = _mm256_setzero_si256();
        for (int w = w0; w <= w1; w += scanInterval)
        {
            int h = 1;
            for (; h + 7 * scanInterval <= hLast; h += 8 * scanInterval)
            {
                const int rowbase = h * imageWidth + w;
                // vindex = 行内元素下标（gather scale=2 → 字节偏移 2·idx）
                __m256i vidx = _mm256_set1_epi32(rowbase);
                vidx = _mm256_add_epi32(vidx, _mm256_setr_epi32(
                                                  0, scanInterval * imageWidth, 2 * scanInterval * imageWidth,
                                                  3 * scanInterval * imageWidth, 4 * scanInterval * imageWidth,
                                                  5 * scanInterval * imageWidth, 6 * scanInterval * imageWidth,
                                                  7 * scanInterval * imageWidth));
                __m256i mr = _mm256_i32gather_epi32((const int *)gradImg, vidx, 2);
                __m256i lf = _mm256_i32gather_epi32((const int *)(gradImg - 1), vidx, 2);
                __m256i up = _mm256_i32gather_epi32((const int *)(gradImg - imageWidth), vidx, 2);
                __m256i dn = _mm256_i32gather_epi32((const int *)(gradImg + imageWidth), vidx, 2);
                __m256i mid = _mm256_and_si256(mr, _mm256_set1_epi32(0xFFFF));
                __m256i right = _mm256_srli_epi32(mr, 16);
                __m256i left = _mm256_and_si256(lf, _mm256_set1_epi32(0xFFFF));
                __m256i gup = _mm256_and_si256(up, _mm256_set1_epi32(0xFFFF));
                __m256i gdn = _mm256_and_si256(dn, _mm256_set1_epi32(0xFFFF));
                // dir：取 4 字节组内第 (w&3) 字节
                __m256i dirv = _mm256_i32gather_epi32((const int *)dirImg, vidx, 1);
                __m256i isH = _mm256_cmpgt_epi32(
                    _mm256_and_si256(dirv, _mm256_set1_epi32(0xFF)), vzero);
                // keepV：mid≠0 且 mid ≥ 左+th 且 mid ≥ 右+th（⟺ !(左+th>mid)）
                __m256i keepV = _mm256_and_si256(
                    _mm256_cmpgt_epi32(mid, vzero),
                    _mm256_andnot_si256(
                        _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_add_epi32(left, vth), mid),
                                        _mm256_cmpgt_epi32(_mm256_add_epi32(right, vth), mid)),
                        _mm256_set1_epi32(-1)));
                __m256i keepH = _mm256_and_si256(
                    _mm256_cmpgt_epi32(mid, vzero),
                    _mm256_andnot_si256(
                        _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_add_epi32(gup, vth), mid),
                                        _mm256_cmpgt_epi32(_mm256_add_epi32(gdn, vth), mid)),
                        _mm256_set1_epi32(-1)));
                __m256i keep = _mm256_blendv_epi8(keepV, keepH, isH);
                unsigned bits = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(keep));
                while (bits)
                {
                    const int k = __builtin_ctz(bits);
                    bits &= bits - 1;
                    anchorPoints.emplace_back(w, h + k * scanInterval);
                    ++nAnchors;
                }
            }
            for (; h <= hLast; h += scanInterval)
            { // 尾部标量
                const int indexInArray = h * imageWidth + w;
                if (gradImg[indexInArray] == 0)
                    continue;
                if (dirImg[indexInArray] == ESED_EDGE_HORIZONTAL)
                {
                    if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
                        gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh)
                    {
                        anchorPoints.emplace_back(w, h);
                        ++nAnchors;
                    }
                }
                else
                {
                    if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
                        gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh)
                    {
                        anchorPoints.emplace_back(w, h);
                        ++nAnchors;
                    }
                }
            }
        }
        (void)nAnchors;
        return;
    }
#else
    // 标量回退（非 AVX2 构建）
    for (int w = w0; w <= w1; w += scanInterval)
    {
        for (int h = 1; h <= hLast; h += scanInterval)
        {
            const int indexInArray = h * imageWidth + w;
            if (gradImg[indexInArray] == 0)
                continue;
            if (dirImg[indexInArray] == ESED_EDGE_HORIZONTAL)
            {
                if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
                    gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh)
                {
                    anchorPoints.emplace_back(w, h);
                    ++nAnchors;
                }
            }
            else
            {
                if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
                    gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh)
                {
                    anchorPoints.emplace_back(w, h);
                    ++nAnchors;
                }
            }
        }
    }
#endif
}

// 逐锚点双向绘制
void FmlDetector::elsedDrawAnchorPoints(const std::vector<cv::Point> &anchorPoints)
{
    if (anchorPoints.empty())
        return;

    for (const auto &anchorPoint : anchorPoints)
    {
        const int indexInArray = anchorPoint.y * els_w_ + anchorPoint.x;
        if (els_edgeImg_[indexInArray])
            continue; // 已被先前绘制的边覆盖

        // 梯度水平 → 先向右扩展；竖直 → 先向下扩展
        const bool expandHorizontally = els_dirImg_[indexInArray] == ESED_EDGE_HORIZONTAL;
        const unsigned char lastDirection = expandHorizontally ? ESED_RIGHT : ESED_DOWN;
        elsedDrawEdgeInBothDirections(lastDirection, anchorPoint);
    }
}

// 双向绘制入口：先沿第一方向画，再沿反方向
// 画；反方向分支预置第一方向尾部像素（逆序）供拟合续接
void FmlDetector::elsedDrawEdgeInBothDirections(uint8_t direction, cv::Point anchor)
{
    std::vector<cv::Point> &anchorPixels = els_anchor_px_; // 复用缓冲
    anchorPixels.clear();

    els_edgeImg_[anchor.y * els_w_ + anchor.x] = ESED_ED_ANCHOR_PX;

    els_stack_.push_back(ElsedBranch{direction, anchor, true, {}});
    els_pixels_.push_back(anchor);
    elsedDrawEdgeTreeStack(anchor, anchorPixels, true);

    const uint8_t oppositeDirection = elsedInverseDirection(direction);
    els_stack_.push_back(ElsedBranch{oppositeDirection, anchor, true, anchorPixels});
    elsedDrawEdgeTreeStack(anchor, anchorPixels, false);
}

// 栈式边缘绘制主循环。
// 每个分支：路由游走；段未激活时滑窗增量拟合（fitError < 0.2 激活，窗口按
// SKIP_EDGE_PT 前移重试）；激活后逐像素内/外点判定。收口按状态派生分支：
//   BRANCH 3：外点超限 → 外点归还状态图，沿预测方向自由续走（复用本栈位）
//   BRANCH 1：沿拟合直线跳越交叉点续接（特征值+角度判据）
//   BRANCH 2.a：段含锚点 → 从锚点向反方向扩展另一侧
//   BRANCH 2.b：另一侧交叉点跳越；都不可行 → 本段完结落账
void FmlDetector::elsedDrawEdgeTreeStack(cv::Point anchor,
                                         std::vector<cv::Point> &initialPixels,
                                         bool firstAnchorDirection)
{
    uint8_t direction, gradDir, lineDirection;
    int i, indexInArray = -1, lastChekedPxIdx, initialPxIndex, nElements;
    bool addPixelsForTheFirstSide, inlierOverwritten, popStack, firstBranch,
        isAnchorFirstPx, wasExtended, localSegInitialized, segment;
    cv::Point px, lastPx;
    double fitError;
    std::vector<cv::Point> &extensionPixels = els_ext_px_; // 复用缓冲
    std::vector<cv::Point> &outliersList = els_out_px_;    // 复用缓冲

    extensionPixels.reserve((size_t)ec_.min_len);

    segment = false;
    firstBranch = true;
    // 是否正以锚点为段首扩展（是则起步索引回退一个像素）
    isAnchorFirstPx = els_pixels_.back() == anchor && firstAnchorDirection;

    // 新建本轮工作快照（未完结段，函数结尾移除）
    els_segments_.emplace_back();
    els_segments_.back().pixels = &els_pixels_;
    ElsedSegment *localSegment = &els_segments_.back();

    while (!els_stack_.empty())
    {
        popStack = true;
        const int stackBack = (int)els_stack_.size() - 1;
        ElsedBranch &branch = els_stack_[stackBack];
        // 本分支首像素在像素链中的索引
        initialPxIndex = (int)els_pixels_.size();
        if (firstBranch && isAnchorFirstPx)
            initialPxIndex--;
        direction = branch.direction;
        px = branch.px;
        addPixelsForTheFirstSide = branch.addPixelsForTheFirstSide;
        els_pixels_.insert(els_pixels_.end(), branch.pixels.begin(), branch.pixels.end());

        // 段拟合窗口起点
        lastChekedPxIdx = initialPxIndex;
        outliersList.clear();
        inlierOverwritten = false;

        lastPx = elsedCalcLastPixelWithDirection(px, direction);

        gradDir = (direction == ESED_DOWN || direction == ESED_UP)
                      ? ESED_EDGE_VERTICAL
                      : ESED_EDGE_HORIZONTAL;

        localSegInitialized = false;
        while (outliersList.size() <= (size_t)ESED_MAX_OUTLIERS_TH)
        {

            if (!elsedFindNextPxWithGradient(gradDir, px, lastPx))
            {
                // 幅值为 0 或抵达图像边界，止步
                break;
            }

            indexInArray = px.y * els_w_ + px.x;
            inlierOverwritten =
                els_edgeImg_[indexInArray] == ESED_ED_INLIER_PX ||
                els_edgeImg_[indexInArray] == ESED_ED_JUNCTION_PX ||
                els_edgeImg_[indexInArray] == ESED_ED_EDGE_PX;
            if (inlierOverwritten)
            {
                break;
            }

            if (segment)
            {
                // 已有拟合段：逐像素内/外点判定
                if (localSegment->isInlier(px.x, px.y, ec_.px_dist))
                {
                    els_edgeImg_[indexInArray] = ESED_ED_INLIER_PX;
                    els_pixels_.push_back(px);

                    outliersList.clear(); // 无条件清空

                    localSegment->addPixel(px.x, px.y, (int)els_pixels_.size() - 1,
                                           addPixelsForTheFirstSide);

                    // 本侧重新开放扩展
                    if (addPixelsForTheFirstSide)
                        localSegment->firstEndpointExtended = false;
                    else
                        localSegment->secondEndpointExtended = false;
                }
                else
                {
                    outliersList.push_back(px);
                    els_edgeImg_[indexInArray] = ESED_ED_OUTLIER_PX;
                }
            }
            else
            {
                // 尚无拟合段：收集像素，攒满 minLineLen 试拟合
                els_pixels_.push_back(px);
                els_edgeImg_[indexInArray] = ESED_ED_EDGE_PX;

                if (els_pixels_.size() - (size_t)lastChekedPxIdx >= (size_t)ec_.min_len)
                {
                    if (localSegInitialized)
                    {
                        localSegment->skipPositions();
                    }
                    else
                    {
                        localSegInitialized = true;
                        localSegment->init(els_pixels_, lastChekedPxIdx);
                    }

                    fitError = localSegment->getFitError();
                    if (fitError < ec_.fit_err)
                    {
                        // 段激活：窗口内像素划归本段
                        segment = true;
                        for (i = lastChekedPxIdx; i < lastChekedPxIdx + ec_.min_len; ++i)
                            els_edgeImg_[els_pixels_[i].y * els_w_ + els_pixels_[i].x] =
                                ESED_ED_INLIER_PX;
                    }
                    else
                    {
                        // 拟合失败：窗口前移 SKIP_EDGE_PT 后重试
                        lastChekedPxIdx += ESED_SKIP_EDGE_PT;
                    }
                }
            }

            // 跟随新像素的方向类
            gradDir = els_dirImg_[indexInArray];
        }

        // 首分支且锚点未被段吸收：留存本分支尾部像素供反方向分支预置
        if (firstBranch &&
            els_edgeImg_[anchor.y * els_w_ + anchor.x] != ESED_ED_INLIER_PX)
        {
            initialPixels.clear();
            nElements = std::min((int)(els_pixels_.size() - initialPxIndex),
                                 ec_.min_len - 1);
            for (i = initialPxIndex + nElements - 1; i >= initialPxIndex; --i)
            {
                // 尾部像素若已归属某段则停止回溯
                const int indexInImage = els_pixels_[i].y * els_w_ + els_pixels_[i].x;
                if (els_edgeImg_[indexInImage] == ESED_ED_INLIER_PX ||
                    els_edgeImg_[indexInImage] == ESED_ED_JUNCTION_PX)
                    break;
                initialPixels.push_back(els_pixels_[i]);
            }
        }

        if (segment)
        {
            if (!outliersList.empty())
            {
                // 止步时仍有外点：回收段上最后一个像素（回滚模型与状态图）
                if (addPixelsForTheFirstSide || localSegment->hasSecondSideElements())
                {
                    const cv::Point pxToDelete = addPixelsForTheFirstSide
                                                     ? localSegment->getLastPixel()
                                                     : localSegment->getFirstPixel();
                    els_edgeImg_[pxToDelete.y * els_w_ + pxToDelete.x] = ESED_ED_NO_EDGE_PX;
                    localSegment->removeLastPx(addPixelsForTheFirstSide);

                    outliersList.insert(outliersList.begin(), els_pixels_.back());
                    els_pixels_.pop_back();

                    if (!addPixelsForTheFirstSide &&
                        localSegment->getFirstPixel() == anchor)
                    {
                        // 段头像素被移空：该侧视为已扩展完
                        localSegment->secondEndpointExtended = true;
                    }
                }
            }

            // BRANCH 3：外点超限 → 外点归还状态图，沿预测方向自由续走
            if (outliersList.size() > (size_t)ESED_MAX_OUTLIERS_TH)
            {
                for (const cv::Point &outlier : outliersList)
                    els_edgeImg_[outlier.y * els_w_ + outlier.x] = ESED_ED_NO_EDGE_PX;

                uint8_t predictedLastDir;
                if (els_dirImg_[indexInArray] == ESED_EDGE_HORIZONTAL)
                    predictedLastDir = lastPx.x < px.x ? ESED_RIGHT : ESED_LEFT;
                else
                    predictedLastDir = lastPx.y < px.y ? ESED_DOWN : ESED_UP;

                popStack = false;
                branch.direction = predictedLastDir;
                branch.px.x = px.x;
                branch.px.y = px.y;
                branch.addPixelsForTheFirstSide = true;

                // 预置末内点 + 外点链，续走时回补模型
                branch.pixels.clear();
                branch.pixels.push_back(els_pixels_.back());
                for (const cv::Point &outlier : outliersList)
                    branch.pixels.push_back(outlier);
            }

            // BRANCH 1：沿段方向直线跳越交叉点续接
            wasExtended = localSegment->firstEndpointExtended;
            if ((outliersList.size() > (size_t)ESED_MAX_OUTLIERS_TH || inlierOverwritten) &&
                !wasExtended && ec_.treat_junc &&
                elsedCanSegmentBeExtended(*localSegment, true, extensionPixels))
            {
                localSegment->ensureSigned();
                lineDirection = elsedDirectionFromLineEq(localSegment->getLineEquation());
                elsedAddJunctionPixelsToSegment(extensionPixels, *localSegment,
                                                addPixelsForTheFirstSide);

                if (popStack)
                {
                    popStack = false;
                    branch.direction = lineDirection;
                    branch.px = extensionPixels.back();
                    branch.addPixelsForTheFirstSide = true;
                    branch.pixels.clear();
                }
                else
                {
                    els_stack_.emplace_back(
                        ElsedBranch{lineDirection, extensionPixels.back(), true, {}});
                }
            }
            else
            {
                // BRANCH 2.a：段含锚点 → 从锚点向直线反方向扩展另一侧
                localSegment->firstEndpointExtended = true;

                wasExtended = localSegment->secondEndpointExtended;
                if (!wasExtended && localSegment->getFirstPixel() == anchor)
                {
                    localSegment->ensureSigned();
                    const uint8_t oppositeLineDirection = elsedInverseDirection(
                        elsedDirectionFromLineEq(localSegment->getLineEquation()));
                    localSegment->secondEndpointExtended = true;

                    if (popStack)
                    {
                        popStack = false;
                        branch.direction = oppositeLineDirection;
                        branch.px = anchor;
                        branch.addPixelsForTheFirstSide = false;
                        branch.pixels.clear();
                    }
                    else
                    {
                        els_stack_.emplace_back(
                            ElsedBranch{oppositeLineDirection, anchor, false, {}});
                    }
                }
                else if (!wasExtended && ec_.treat_junc &&
                         elsedCanSegmentBeExtended(*localSegment, false,
                                                   extensionPixels))
                {
                    // BRANCH 2.b：另一侧交叉点跳越
                    elsedAddJunctionPixelsToSegment(extensionPixels, *localSegment, false);

                    localSegment->ensureSigned();
                    const uint8_t oppositeLineDirection = elsedInverseDirection(
                        elsedDirectionFromLineEq(localSegment->getLineEquation()));

                    if (popStack)
                    {
                        popStack = false;
                        branch.direction = oppositeLineDirection;
                        branch.px = extensionPixels.back();
                        branch.addPixelsForTheFirstSide = false;
                        branch.pixels.clear();
                    }
                    else
                    {
                        els_stack_.emplace_back(
                            ElsedBranch{oppositeLineDirection, extensionPixels.back(), false, {}});
                    }
                }
                else
                {
                    // 无后续分支：本段完结落账，新开工作快照
                    localSegment->secondEndpointExtended = true;
                    localSegment->finish();
                    els_segments_.emplace_back();
                    els_segments_.back().pixels = &els_pixels_;
                    localSegment = &els_segments_.back();
                    segment = false;
                }
            }
        }

        if (popStack)
        {
            // 未派生新分支：弹出本栈位
            els_stack_.resize(els_stack_.size() - 1);
        }

        firstBranch = false;
        extensionPixels.clear();
    }

    // 移除末尾未完结的工作快照
    els_segments_.pop_back();
}

// 梯度图 3 邻域择优路由：
// 按当前像素方向类与来向限制候选（Bresenham 式 2/3 邻域），取幅值最大者
//（严格大于才偏折，平局直行）；px 回写新像素，lastPx 回写移动前像素；
// 返回新像素幅值（0 = 止步）
bool FmlDetector::elsedFindNextPxWithGradient(uint8_t pxGradDirection,
                                              cv::Point &px, cv::Point &lastPx) const
{
    int16_t gValue1, gValue2, gValue3, gNext;
    const int lastX = lastPx.x, lastY = lastPx.y;
    lastPx = px;
    const int w = els_w_;

    // 末像素幅值检查复用本轮已加载的邻居值（gNext 即选中邻位或两侧
    // 比较值之一），不做二次寻址
    if (pxGradDirection == ESED_EDGE_HORIZONTAL)
    {
        // 梯度水平（边大致竖直）：

        // | Ok | X | Ok |
        // | Ok | X | Ok |
        // | Ok | X | Ok |

        if (lastX < px.x)
        {
            // 右行
            if (px.x == w - 1 || px.y == 0 || px.y == els_h_ - 1)
                return false; // 边界
            const int indexInArray = px.y * w + px.x + 1;
            px.x++;
            gValue1 = els_gImg_[indexInArray - w];
            gValue2 = els_gImg_[indexInArray];
            gValue3 = els_gImg_[indexInArray + w];
            if (__builtin_expect(gValue1 > gValue2 && gValue1 > gValue3, 0))
            {
                px.y = px.y - 1; // 右上
                gNext = gValue1;
            }
            else if (__builtin_expect(gValue3 > gValue2 && gValue3 > gValue1, 0))
            {
                px.y = px.y + 1; // 右下
                gNext = gValue3;
            }
            else
            {
                gNext = gValue2; // 直行（高频）
            }
        }
        else if (lastX > px.x)
        {
            // 左行
            if (px.x == 0 || px.y == 0 || px.y == els_h_ - 1)
                return false;
            const int indexInArray = px.y * w + px.x - 1;
            px.x--;
            gValue1 = els_gImg_[indexInArray - w];
            gValue2 = els_gImg_[indexInArray];
            gValue3 = els_gImg_[indexInArray + w];
            if (__builtin_expect(gValue1 > gValue2 && gValue1 > gValue3, 0))
            {
                px.y = px.y - 1; // 左上
                gNext = gValue1;
            }
            else if (__builtin_expect(gValue3 > gValue2 && gValue3 > gValue1, 0))
            {
                px.y = px.y + 1; // 左下
                gNext = gValue3;
            }
            else
            {
                gNext = gValue2;
            }
        }
        else if (lastY < px.y)
        { // lastX == px.x
            // 下行
            if (px.y == els_h_ - 1 || px.x == 0 || px.x == w - 1)
                return false;
            const int indexInArray = (px.y + 1) * w + px.x;
            px.y++;
            if (els_gImg_[indexInArray - 1] > els_gImg_[indexInArray + 1])
            {
                px.x--; // 左下
                gNext = els_gImg_[indexInArray - 1];
            }
            else
            {
                px.x++; // 右下
                gNext = els_gImg_[indexInArray + 1];
            }
        }
        else
        { // lastX == px.x && lastY > px.y
            // 上行
            if (px.y == 0 || px.x == 0 || px.x == w - 1)
                return false;
            const int indexInArray = (px.y - 1) * w + px.x;
            px.y--;
            if (els_gImg_[indexInArray - 1] > els_gImg_[indexInArray + 1])
            {
                px.x--; // 左上
                gNext = els_gImg_[indexInArray - 1];
            }
            else
            {
                px.x++; // 右上
                gNext = els_gImg_[indexInArray + 1];
            }
        }
    }
    else
    {
        // 梯度竖直（边大致水平）：

        // | Ok | Ok | Ok |
        // |  X |  X |  X |
        // | Ok | Ok | Ok |

        if (lastY < px.y)
        {
            // 下行
            if (px.y == els_h_ - 1 || px.x == 0 || px.x == w - 1)
                return false;
            const int indexInArray = (px.y + 1) * w + px.x;
            px.y++;
            gValue1 = els_gImg_[indexInArray + 1];
            gValue2 = els_gImg_[indexInArray];
            gValue3 = els_gImg_[indexInArray - 1];
            if (__builtin_expect(gValue1 > gValue2 && gValue1 > gValue3, 0))
            {
                px.x = px.x + 1; // 右下
                gNext = gValue1;
            }
            else if (__builtin_expect(gValue3 > gValue2 && gValue3 > gValue1, 0))
            {
                px.x = px.x - 1; // 左下
                gNext = gValue3;
            }
            else
            {
                gNext = gValue2;
            }
        }
        else if (lastY > px.y)
        {
            // 上行
            if (px.y == 0 || px.x == 0 || px.x == w - 1)
                return false;
            const int indexInArray = (px.y - 1) * w + px.x;
            px.y--;
            gValue1 = els_gImg_[indexInArray + 1];
            gValue2 = els_gImg_[indexInArray];
            gValue3 = els_gImg_[indexInArray - 1];
            if (__builtin_expect(gValue1 > gValue2 && gValue1 > gValue3, 0))
            {
                px.x = px.x + 1; // 右上
                gNext = gValue1;
            }
            else if (__builtin_expect(gValue3 > gValue2 && gValue3 > gValue1, 0))
            {
                px.x = px.x - 1; // 左上
                gNext = gValue3;
            }
            else
            {
                gNext = gValue2;
            }
        }
        else if (lastX < px.x)
        { // lastY == px.y
            // 右行
            if (px.x == w - 1 || px.y == 0 || px.y == els_h_ - 1)
                return false;
            const int indexInArray = px.y * w + px.x + 1;
            px.x++;
            if (els_gImg_[indexInArray - w] > els_gImg_[indexInArray + w])
            {
                px.y--; // 右上
                gNext = els_gImg_[indexInArray - w];
            }
            else
            {
                px.y++; // 右下
                gNext = els_gImg_[indexInArray + w];
            }
        }
        else
        { // lastY == px.y && lastX > px.x
            // 左行
            if (px.x == 0 || px.y == 0 || px.y == els_h_ - 1)
                return false;
            const int indexInArray = px.y * w + px.x - 1;
            px.x--;
            if (els_gImg_[indexInArray - w] > els_gImg_[indexInArray + w])
            {
                px.y--; // 左上
                gNext = els_gImg_[indexInArray - w];
            }
            else
            {
                px.y++; // 左下
                gNext = els_gImg_[indexInArray + w];
            }
        }
    }

    return gNext != 0;
}

// 投影外推：从 px 沿线方程方向外推
// stepSize 像素（先投影到线上再延展），投影点 2x2 邻域取幅值最大者为续接
// 起点；返回 false = 外推出界或四处幅值全 0
static bool elsedFindNextPxWithProjection(const short *gradImg, cv::Vec3f eq,
                                          bool invertLineDir, int imageWidth,
                                          int imageHeight, int stepSize, cv::Point &px)
{
    int16_t gValue1, gValue2, gValue3, gValue4;
    const cv::Point2f p((float)px.x, (float)px.y);
    const cv::Point2f lasPxReproj = elsedGetProjectionPtn(eq, p);

    cv::Point2f extendedPoint;
    if (!invertLineDir)
    {
        extendedPoint.x = lasPxReproj.x - stepSize * eq[1];
        extendedPoint.y = lasPxReproj.y + stepSize * eq[0];
    }
    else
    {
        extendedPoint.x = lasPxReproj.x + stepSize * eq[1];
        extendedPoint.y = lasPxReproj.y - stepSize * eq[0];
    }

    if (extendedPoint.x < 0 || extendedPoint.x >= imageWidth ||
        extendedPoint.y < 0 || extendedPoint.y >= imageHeight)
    {
        return false;
    }
    const int x = (int)extendedPoint.x;
    const int y = (int)extendedPoint.y;
    gValue1 = gradImg[y * imageWidth + x];
    gValue2 = gradImg[y * imageWidth + std::min(x + 1, imageWidth - 1)];
    gValue3 = gradImg[std::min(y + 1, imageHeight - 1) * imageWidth + x];
    gValue4 = gradImg[std::min(y + 1, imageHeight - 1) * imageWidth +
                      std::min(x + 1, imageWidth - 1)];

    // 取 4 者最大（钳位到图内，安全护栏）
    if (gValue2 > gValue1)
    {
        if (gValue3 > gValue2)
        {
            if (gValue4 > gValue3)
            {
                px.x = std::min(x + 1, imageWidth - 1);
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue4 != 0;
            }
            else
            {
                px.x = x;
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue3 != 0;
            }
        }
        else
        {
            if (gValue4 > gValue2)
            {
                px.x = std::min(x + 1, imageWidth - 1);
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue4 != 0;
            }
            else
            {
                px.x = std::min(x + 1, imageWidth - 1);
                px.y = y;
                return gValue2 != 0;
            }
        }
    }
    else
    {
        if (gValue3 > gValue1)
        {
            if (gValue4 > gValue3)
            {
                px.x = std::min(x + 1, imageWidth - 1);
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue4 != 0;
            }
            else
            {
                px.x = x;
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue3 != 0;
            }
        }
        else
        {
            if (gValue4 > gValue1)
            {
                px.x = std::min(x + 1, imageWidth - 1);
                px.y = std::min(y + 1, imageHeight - 1);
                return gValue4 != 0;
            }
            else
            {
                px.x = x;
                px.y = y;
                return gValue1 != 0;
            }
        }
    }
}

// 交叉点像素并入段：内点前半标记
// JUNCTION、后半标记 INLIER 并入模型；外点标记 OUTLIER 并丢弃
void FmlDetector::elsedAddJunctionPixelsToSegment(
    const std::vector<cv::Point> &junctionPixels, ElsedSegment &segment,
    bool addPixelsForTheFirstSide)
{
    if (addPixelsForTheFirstSide)
        segment.firstEndpointExtended = true;
    else
        segment.secondEndpointExtended = true;

    const int n = (int)junctionPixels.size();
    for (int i = 0; i < n; ++i)
    {
        const cv::Point &junctionPx = junctionPixels[i];
        if (segment.isInlier(junctionPx.x, junctionPx.y, ec_.px_dist))
        {
            if (i < n / 2)
                els_edgeImg_[junctionPx.y * els_w_ + junctionPx.x] = ESED_ED_JUNCTION_PX;
            else
                els_edgeImg_[junctionPx.y * els_w_ + junctionPx.x] = ESED_ED_INLIER_PX;
            els_pixels_.push_back(junctionPx);
            segment.addPixel(junctionPx.x, junctionPx.y, (int)els_pixels_.size() - 1,
                             addPixelsForTheFirstSide);
        }
        else
        {
            els_edgeImg_[junctionPx.y * els_w_ + junctionPx.x] = ESED_ED_OUTLIER_PX;
        }
    }
}

// 交叉点可续接判定：沿段方向外推 junctionSize
// 像素并要求 ED 路由全程内点；再以梯度自相关矩阵（2x2 SVD 手工展开）
// 特征值比 + 特征向量角度判定延伸区是直线延伸而非角点
bool FmlDetector::elsedCanSegmentBeExtended(ElsedSegment &segment,
                                            bool extendByTheEnd,
                                            std::vector<cv::Point> &pixelsInTheExtension)
{
    bool inlier, found, horizontalValidationDir, fitsEigenvaluesCond, fitsAngleCond;
    int validateDx, validateDy, xOffset, yOffset, indexInArray;
    double theta;
    float a, b, d, tmp, s1, s2, eigen_angle;
    cv::Point firstPx, lastPx, px;
    std::vector<cv::Point> pixelsToValidate;
    cv::Vec3d eq;

    firstPx = extendByTheEnd ? segment.getLastPixel() : segment.getFirstPixel();

    for (int GRAD_JUNCTION_SIZE : kElsedJunctionSizes)
    {
        if (segment.getNumOfPixels() <= GRAD_JUNCTION_SIZE)
            break;

        px = firstPx;
        // 沿线外推 GRAD_JUNCTION_SIZE 像素
        found = elsedFindNextPxWithProjection(els_gImg_, segment.getLineEquation(),
                                              !extendByTheEnd, els_w_, els_h_,
                                              GRAD_JUNCTION_SIZE, px);
        if (!found)
            continue;

        // 起点到外推点的 Bresenham 栅格
        elsedBresenham(firstPx.x, firstPx.y, px.x, px.y, pixelsInTheExtension);
        lastPx = pixelsInTheExtension[pixelsInTheExtension.size() - 2];

        // 沿段方向继续 ED 路由 GRAD_JUNCTION_SIZE 像素，要求全程内点
        //（pixelsToValidate 不在轮次间清空：失败尝试的残留像素
        //  会进入后续轮次的特征值统计）
        bool broken = false;
        for (int i = 0; i < GRAD_JUNCTION_SIZE; ++i)
        {
            if (!elsedFindNextPxWithGradient(
                    segment.horizontal() ? ESED_EDGE_HORIZONTAL : ESED_EDGE_VERTICAL,
                    px, lastPx))
            {
                pixelsInTheExtension.clear();
                broken = true;
                break;
            }
            inlier = segment.isInlier(px.x, px.y, ec_.px_dist);
            if (inlier)
            {
                pixelsInTheExtension.push_back(px);
                pixelsToValidate.push_back(px);
            }
            else
            {
                pixelsInTheExtension.clear();
                broken = true;
                break;
            }
        }
        if (broken || pixelsInTheExtension.empty())
            continue;

        // 段法向角 theta ∈ [0, π)
        eq = segment.getLineEquation();
        theta = std::atan2(eq[0], eq[1]) + CV_PI / 2.0;
        while (theta < 0)
            theta += CV_PI;
        while (theta >= CV_PI)
            theta -= CV_PI;

        // 延伸区梯度自相关矩阵 M = [[a, b], [b, d]]
        a = 0;
        b = 0;
        d = 0;
        validateDx = std::abs(pixelsToValidate.back().x - pixelsToValidate.front().x);
        validateDy = std::abs(pixelsToValidate.back().y - pixelsToValidate.front().y);
        horizontalValidationDir = validateDx >= validateDy;
        for (const cv::Point &extPixel : pixelsToValidate)
        {
            for (int offset : {-1, 0, 1})
            {
                // 按延伸区走向取纵向或横向邻居
                xOffset = horizontalValidationDir ? 0 : offset;
                yOffset = horizontalValidationDir ? offset : 0;
                indexInArray =
                    std::min(els_h_ - 1, std::max(0, extPixel.y + yOffset)) * els_w_ +
                    std::min(els_w_ - 1, std::max(0, extPixel.x + xOffset));
                a += els_pDxImg_[indexInArray] * els_pDxImg_[indexInArray];
                b += els_pDxImg_[indexInArray] * els_pDyImg_[indexInArray];
                d += els_pDyImg_[indexInArray] * els_pDyImg_[indexInArray];
            }
        }
        // 2x2 SVD 手工展开
        tmp = a * a - d * d;
        s1 = a * a + 2 * b * b + d * d;
        s2 = std::sqrt(tmp * tmp + 4 * b * b * (a + d) * (a + d));
        eigen_angle = -0.5f * std::atan2(2 * a * b + 2 * b * d, tmp);
        while (eigen_angle < 0)
            eigen_angle += (float)CV_PI;
        while (eigen_angle >= (float)CV_PI)
            eigen_angle -= (float)CV_PI;

        // 特征值条件：主特征值显著大于次特征值（非角点）
        fitsEigenvaluesCond =
            std::sqrt((s1 + s2) / (s1 - s2 + 0.00001)) > kElsedJunctionEigenvalsTh;
        // 角度条件：第一特征向量与段法向夹角足够小
        fitsAngleCond =
            elsedCircularDist(theta, eigen_angle, CV_PI) < kElsedJunctionAngleTh;

        if (!fitsEigenvaluesCond || !fitsAngleCond)
            continue;

        return true;
    }
    return false;
}

// ELSED 主流程，验证后接入共享后段（detector.cpp）
void FmlDetector::detectElsed(const cv::Mat &src, std::vector<Segment> &segments_all)
{
    imageheight = src.rows;
    imagewidth = src.cols;
    els_w_ = imagewidth;
    els_h_ = imageheight;

    elsedComputeGradients(src);

    els_edge_.create(els_h_, els_w_, CV_8U);
    els_edge_.setTo(cv::Scalar(0));
    els_pixels_.clear();
    els_segments_.clear();
    els_segments_.reserve(1024);
    els_stack_.clear();
    els_stack_.reserve(4096);

    els_gImg_ = els_g_.ptr<short>();
    els_dirImg_ = els_dir_.ptr();
    els_edgeImg_ = els_edge_.ptr();
    els_pDxImg_ = els_dx_.ptr<short>();
    els_pDyImg_ = els_dy_.ptr<short>();

    // 段验证核心（泛型）：段/像素访问器参数化，串行与双线程路径
    // 共用同一算术体
    auto validateLoop = [&](auto &&getSeg, auto &&getPx, int i0, int i1,
                            std::vector<Segment> &out)
    {
        for (int sIdx = i0; sIdx < i1; ++sIdx)
        {
            const auto &detectedSeg = getSeg(sIdx);
            bool valid = true;
            int nOriInliers = 0, nOriOutliers = 0;
            const cv::Vec4f s = detectedSeg.getEndpoints();
            Segment seg;
            seg.x1 = s[0];
            seg.y1 = s[1];
            seg.x2 = s[2];
            seg.y2 = s[3];
            seg.angle = 0.0f;
            seg.conf = 0.0f;
            seg.sup_n = 0;
            seg.sup_m = 0;
            int gdx = 0;

            if (detectedSeg.getNumOfPixels() < 2)
            {
                valid = false;
            }
            else
            {
                double theta = elsedSegAngle(s) + CV_PI / 2.0;
                while (theta < 0)
                    theta += CV_PI;
                while (theta >= CV_PI)
                    theta -= CV_PI;

                // 角度判据的平方变形：circularDist(θ,a,π) > th（th<π/2）⟺
                // |sin(θ−a)| > sin th；未归一化梯度下 |sin(θ−a)|·|g| =
                // |ldy·cosθ − ldx·sinθ|，两边平方免开方。逐段预计算三角量，
                // 逐像素 4 乘 2 加
                const float vth_cos = (float)std::cos(theta);
                const float vth_sin = (float)std::sin(theta);
                float vth_s2 = (float)std::sin(ec_.validate_th);
                vth_s2 *= vth_s2;
                if (ec_.validate_th >= CV_PI / 2.0)
                    vth_s2 = 1e30f; // 阈值≥90°：无外点

                // 端点叉积得线方程并归一化（float）
                cv::Vec3f l = cv::Vec3f(s[0], s[1], 1).cross(cv::Vec3f(s[2], s[3], 1));
                l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
                const float l0 = l[0], l1 = l[1], l2 = l[2];

                const int nPixelsToTrim =
                    (int)std::min(5.0, detectedSeg.getNumOfPixels() * 0.1);
                const cv::Point &firstPx = detectedSeg.getFirstPixel();
                const cv::Point &lastPx = detectedSeg.getLastPixel();

                // 验证 + 支撑统计（sup_n/sup_m/Σdx）融合为单遍遍历，行指针
                // 缓存免逐像素 y·w 寻址。统计计数与验证计数不互相干扰
                const float vx = s[2] - s[0], vy = s[3] - s[1];
                const float ang_e = std::atan2(vy, vx) + (float)CV_PI / 2.0f;
                const float ex = std::cos(ang_e), ey = std::sin(ang_e);
                const int w = els_w_;
                const short *pdxRow = nullptr;
                const short *pdyRow = nullptr;
                const unsigned char *edgeRow = nullptr;
                int lastY = -1;
                for (int pi = detectedSeg.firstPxIndex; pi <= detectedSeg.lastPxIndex; ++pi)
                {
                    const cv::Point &px = getPx(pi);
                    if (px.y != lastY)
                    {
                        pdxRow = els_pDxImg_ + (size_t)px.y * w;
                        pdyRow = els_pDyImg_ + (size_t)px.y * w;
                        edgeRow = els_edgeImg_ + (size_t)px.y * w;
                        lastY = px.y;
                    }
                    // 非内点不计入
                    if (edgeRow[px.x] != ESED_ED_INLIER_PX)
                        continue;

                    // 支撑统计：先于 trim 判定计数，所有内点都计入
                    const float gx = (float)pdxRow[px.x];
                    const float gy = (float)pdyRow[px.x];
                    gdx += (int)pdxRow[px.x];
                    ++seg.sup_n;
                    if (isAligned(gx, gy, ex, ey))
                        ++seg.sup_m;

                    // 段角度验证
                    const int endpointDist = detectedSeg.horizontal()
                                                 ? std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x))
                                                 : std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));
                    if (endpointDist < nPixelsToTrim)
                        continue;

                    // 重投影：p = px − (l0,l1)·(l0·px + l1·py + l2)
                    const float d = l0 * px.x + l1 * px.y + l2;
                    const float pfxx = px.x - l0 * d;
                    const float pfyy = px.y - l1 * d;
                    // 退化段（两端重合）NaN 防护
                    if (pfxx != pfxx || pfyy != pfyy)
                        continue;

                    // 双线性插值梯度 2x2 邻域
                    int x0 = pfxx < 0 ? 0 : (int)pfxx;
                    if (x0 >= w)
                        x0 = w - 1;
                    int y0 = pfyy < 0 ? 0 : (int)pfyy;
                    if (y0 >= els_h_)
                        y0 = els_h_ - 1;
                    int x1 = (int)(pfxx + 1.0f);
                    if (x1 >= w)
                        x1 = w - 1;
                    if (x1 < 0)
                        x1 = 0; // 安全护栏（内点约束下正常不可达）
                    int y1 = (int)(pfyy + 1.0f);
                    if (y1 >= els_h_)
                        y1 = els_h_ - 1;
                    if (y1 < 0)
                        y1 = 0; // 安全护栏（同上）

                    const float tx = pfxx - (int)pfxx, ty = pfyy - (int)pfyy;
                    const size_t ro = (size_t)y0 * w, ru = (size_t)y1 * w;
                    const float c00x = (float)els_pDxImg_[ro + x0], c10x = (float)els_pDxImg_[ro + x1];
                    const float c01x = (float)els_pDxImg_[ru + x0], c11x = (float)els_pDxImg_[ru + x1];
                    const float c00y = (float)els_pDyImg_[ro + x0], c10y = (float)els_pDyImg_[ro + x1];
                    const float c01y = (float)els_pDyImg_[ru + x0], c11y = (float)els_pDyImg_[ru + x1];
                    const float lerp_dx = (c00x + (c10x - c00x) * tx) * (1.f - ty) + (c01x + (c11x - c01x) * tx) * ty;
                    const float lerp_dy = (c00y + (c10y - c00y) * tx) * (1.f - ty) + (c01y + (c11y - c01y) * tx) * ty;

                    const float cr = lerp_dy * vth_cos - lerp_dx * vth_sin;
                    const float g2 = lerp_dx * lerp_dx + lerp_dy * lerp_dy;
                    cr *cr > vth_s2 *g2 ? ++nOriOutliers : ++nOriInliers;
                }

                valid = nOriInliers > nOriOutliers;
            }
            if (!valid)
                continue;

            seg.conf = (float)(-logNfa(seg.sup_n, seg.sup_m));
            // 边缘对几何先验平移
            const int off = gdx < 0 ? shift_px_ : gdx > 0 ? -shift_px_
                                                          : 0;
            seg.x1 += (float)off;
            seg.x2 += (float)off;
            out.push_back(seg);
        }
    };

    // 锚点扫描：阈值 8 起逐级减半直至出现锚点（阈值 0 收尾）
    std::vector<cv::Point> &anchors = els_anchors_; // 复用缓冲
    uint8_t anchorTh = (uint8_t)ec_.anchor_th;      // 初始阈值
    bool anchorThIsZero;
    do
    {
        anchorThIsZero = anchorTh == 0;
        if (nthr_ == 2 && imagewidth >= 64)
        {
            // 列拆分（按扫描网格索引取中点）：主线程低列段，工作线程高列段；
            // 列间判定独立，按 w 升序拼接
            const int si = ec_.scan_intv;
            const int nGrid = (imagewidth - 3) / si + 1; // 网格点 1, 1+si, ..., W-2
            const int kmid = nGrid / 2;
            els_anchors_wk_.clear();
            FmlPool *fp = FmlPool::instance();
            fp->post(0, 0, [&](int, int)
                     {
                         elsedComputeAnchorPoints(els_anchors_wk_, anchorTh,
                                                  1 + kmid * si, imagewidth - 2);
                     });
            elsedComputeAnchorPoints(anchors, anchorTh, 1, 1 + (kmid - 1) * si);
            fp->waitDone();
            anchors.insert(anchors.end(), els_anchors_wk_.begin(),
                           els_anchors_wk_.end());
        }
        else
        {
            elsedComputeAnchorPoints(anchors, anchorTh);
        }
        if (anchors.empty())
            anchorTh /= 2;
    } while (anchors.empty() && !anchorThIsZero);

    elsedDrawAnchorPoints(anchors);

    // 段角度验证（[3] 的段验证判据）
    // 重投影像素处双线性插值梯度角 vs 段法向角（trim = min(5, 10%N)），
    // 角误差 > validationTh 记外点；有效需内点数 > 外点数。
    // threads=2：段间相互独立（只读梯度平面/段自身），按下标二分并行、
    // 按序拼接（logNfa 查表为 thread_local，工作线程首次调用自建副本）。
    // 注：绘制与验证不可重叠——后续段会经交叉点吸收改写先前段像素的
    // 边缘状态，串行验证读到的才是终图状态。
    std::vector<Segment> &validated = els_validated_; // 复用缓冲
    validated.clear();
    validated.reserve(els_segments_.size());
    const int nSeg = (int)els_segments_.size();
    if (nthr_ == 2 && nSeg >= 16)
    {
        const int smid = nSeg / 2;
        els_validated_wk_.clear();
        FmlPool *fp = FmlPool::instance();
        fp->post(0, 0, [&](int, int)
                 {
                     validateLoop([this](int i) -> const ElsedSegment &
                                  {
                                      return els_segments_[i];
                                  },
                                  [this](int pi) -> const cv::Point &
                                  {
                                      return els_pixels_[pi];
                                  },
                                  smid, nSeg, els_validated_wk_);
                 });
        validateLoop([this](int i) -> const ElsedSegment &
                     {
                         return els_segments_[i];
                     },
                     [this](int pi) -> const cv::Point &
                     {
                         return els_pixels_[pi];
                     },
                     0, smid, validated);
        fp->waitDone();
        validated.insert(validated.end(), els_validated_wk_.begin(),
                         els_validated_wk_.end());
    }
    else
    {
        validateLoop([this](int i) -> const ElsedSegment &
                     {
                         return els_segments_[i];
                     },
                     [this](int pi) -> const cv::Point &
                     {
                         return els_pixels_[pi];
                     },
                     0, nSeg, validated);
    }

    // 管线后段：长度/贴边过滤 → 方向统一 → 合并
    std::vector<Segment> &kept = els_kept_; // 复用缓冲
    kept.clear();
    kept.reserve(validated.size());
    for (const Segment &seg0 : validated)
    {
        Segment seg = seg0;
        const float length = std::sqrt((seg.x1 - seg.x2) * (seg.x1 - seg.x2) +
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
        kept.push_back(seg);
    }
    if (do_merge)
        mergeAndCollect(src, kept, segments_all);
}
} // namespace fml
