// cluster.cpp —— fml 线段结果的车道线聚类实现
//
// 算法：节点化两段式——先局部成节点，再全局提线
//   节点层（图像纵向等分 bands 条扫描带，逐窗独立重做）：窗口内
//     线段按 x(yc) 排序后多开放组贪心分组为小簇（=节点），方向与
//     位置双约束（含成对发散检验防交叉混组）；节点携带窗口中心
//     测量 (x, yc)、平均方向单位向量、权重 w = Σ len·(0.5+饱和conf/3)
//     与组内平均原始置信度
//   提取层（全帧节点统一处理，全局视野）：
//     方向加权 MSAC 逐线提取：每轮在剩余节点上随机 3 点采样
//     （fit_quad=0 时 2 点直线），样本对先过方向相容预检、解出模型
//     后 3 样本自一致才全表计数；内点判定为双约束（横向残差
//     <= x_tol 且节点方向与模型切向夹角达标），按加权 MSAC 分数
//     Σ w·(τ²−r²) 择优；内点加权最小二乘精修后全表重收内点，再按
//     miss_max×窗口高 断线切分取最长连续子段（再精修），内点硬删除
//     （一节点只归一线）后提取下一条直至不足
//   输出：内点节点序列（按 y 降序）即车道线中心点链 centers；评分
//     = 0.28·链长 + 0.22·端到端延展 + 0.50·log10(1+平均置信度)；未达
//     min_length、居顶部噪声带（top_noise）或链呈锯齿形态（双缘混线，
//     连续折返 ≥3）的链 valid=false 且得分打折；线底端播种在顶部
//     top_noise_ratio 噪声带内 → top_noise
//
// 链拟合（描述性，fit_quad 开关）：MSAC 3 点（抛物线）/2 点（直线）
// 采样剔除外点 + 内点迭代最小一乘（L1/IRLS）精修；提取门限（x_tol，
// 宽）与链内精修门限（msac_thresh，严）两级分离：提取找全，精修剔噪。
//
// 预处理剔除：与竖直夹角超过 max_psi 的"太水平"线段在聚类前丢弃。
// 置信度为 fml 输出的原始绝对值（-log10 NFA，跨帧可比）。
//
// 性能设计：热路径零三角函数（角度门限化为点积门限）、零堆分配
// （节点表/成员池/内点表/工作区均为复用缓冲，链拟合全栈数组，输出
// 原位覆盖）、提取层全表计数 AVX2 8 宽 FMA、全局提取用帧级固定
// 种子 LCG 保证同输入同输出。
//
// 曲线参数（a2/a1/a0/rms）为中心点链的描述性拟合，不参与任何判据。
// 提取与拟合均为 x(y) 参数化（车道线 x(y) 单值的几何先验），近竖直
// 车道线效果最佳。

#include "cluster.hpp"
#include "utils.hpp"            // 快速数学工具（rsqrtF/log10F/hsum256，与 fml 共用）
#include "scorer_weights_data.h" // 内置打分权重（当前数据集训练）

#include <opencv2/imgproc.hpp> // cv::line/circle/LINE_AA（绘制接口）

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define CLU_AVX2 1
#endif

#ifdef FML_PROF_ROT
// 旋转阶段微基准诊断（仅 FML_PROF_ROT 编译时存在，生产构建无此代码）：
// thread_local 累计黄金分割搜索与最终精修的 TSC 周期与调用次数，
// 供 bench_cluster --rotprof 读取并换算成 ms。
struct RotProf
{
    unsigned long long gs_ticks = 0, final_ticks = 0, calls = 0;
    unsigned long long eval_ticks = 0, evals = 0; // 每次 E(θ) 平均
};
static thread_local RotProf g_rotProf;

extern "C" void fml_rot_prof(unsigned long long *gs, unsigned long long *fin,
                             unsigned long long *calls, bool reset)
{
    *gs = g_rotProf.gs_ticks;
    *fin = g_rotProf.final_ticks;
    *calls = g_rotProf.calls;
    if (reset)
    {
        g_rotProf.gs_ticks = 0;
        g_rotProf.final_ticks = 0;
        g_rotProf.calls = 0;
        g_rotProf.eval_ticks = 0;
        g_rotProf.evals = 0;
    }
}
extern "C" void fml_rot_prof_eval(unsigned long long *et, unsigned long long *en)
{
    *et = g_rotProf.eval_ticks;
    *en = g_rotProf.evals;
}
#endif

namespace
{

// 快速数学工具复用 utils.hpp（fml/cluster 同一份实现，跨 TU inline 共享）
inline float cluRsqrtF(float x) { return fml::rsqrtF(x); }
inline float cluLog10F(float x) { return fml::log10F(x); }
#if CLU_AVX2
inline float hsum256(__m256 v) { return fml::hsum256(v); }
#endif

constexpr float kMaxSlope = 64.f;  // 入桶排序键 x(yc) 外推的斜率钳位
                                   // （近水平线段防护）
constexpr int kMaxChainPts = 256;  // 中心链点数上限（交叉截面单线可达
                                   // 多节点/窗，超出仅防御性钳位；栈
                                   // 数组防堆分配）
// 全局提取层常量（内部标定，不进公开参数面）
constexpr int kGlobalIters = 64;   // 单条候选线的 MSAC 采样基准轮数
                                   // （空间局部化下模型收敛快；
                                   // 小线按反比补偿，后续 LO 式内点
                                   // 扩张再补质量）
constexpr int kRefNodes = 96;      // 轮数补偿的基准节点规模
constexpr int kMinNodeInl = 4;     // 候选线成立的最少内点节点数
                                   // （更少的线长度低于 min_length
                                   // 必 invalid，提取纯属输出噪声）

struct Seg
{
    float x1, y1, x2, y2;    // 端点（y1 <= y2，y1 为上端点）
    float ty;                // 下端点 y（top_noise 判播种位置用）
    float len;               // 长度
    float ux, uy;            // 单位方向向量（uy >= 0，指向下端）
    float tm;                // ux/uy，钳位 ±kMaxSlope（入桶排序键外推用）
    float conf;              // 原始置信度（-log10 NFA，可负）
    int wLo, wHi;            // 跨越的窗口带区间（含），-1 = 已剔除
};

// 全局节点（窗口局部聚类封口产物）：MSAC 提取的最小单元。
// 一条真实车道线在其覆盖的每个窗口产生 0~数个节点（交叉截面分段
// 时为多个）
struct Node
{
    float x, y;      // 窗口中心处测量 x（加权均值），窗口中心 y
    float ux, uy;    // 平均方向单位向量（uy >= 0）
    float w;         // 测量权重 Σ len·(0.5+饱和conf/3)
    float conf;      // 组内平均原始置信度
    float seed_y;    // 已记录成员的最大 ty（线底端播种位置，
                     // top_noise 判定用）
    int off, cnt;    // 成员线段下标在扁平池 nodePool 的区间（indices
                     // 诊断输出用）
};

// 候选线：一次全局提取的产物（内点集合 + 提取模型）
struct CandLine
{
    float a2, a1, a0; // 提取模型（加权 LSQ 精修后），描述性初值
    int off, cnt;     // 内点节点下标在 candInl 的区间
};

// 链拟合输出（描述性）：抛物线（fit_quad=1）或直线（fit_quad=0）
struct CurveFit
{
    float a2 = 0.f, a1 = 0.f, a0 = 0.f; // 竖直主轴抛物线 x = a2·y²+a1·y+a0
    float t_min = 0.f, t_max = 0.f;     // y 范围（MSAC 内点集）
    float rms = 0.f;                    // 横向 RMS 残差（竖直主轴参考）
    // 旋转主轴描述（fit_rot 开启时有效；θ=0 时 A/B/C 与 a2/a1/a0 一致）
    float theta = 0.f;                   // 最优旋转角（度）
    float ra = 0.f, rb = 0.f, rc = 0.f; // u = ra·v² + rb·v + rc
    // 以上 (u,v) 与图像坐标 (x,y) 满足旋转矩阵 R(θ)：u = x·cosθ + y·sinθ，
    // v = -x·sinθ + y·cosθ（θ=0 时 u=x、v=y，退化为竖直主轴参考系）
};

// 权重加权最小二乘求解器（定义见下）：提取层精修与链拟合 IRLS 各步
// 共用，此处前向声明
void wlsFit(const float *px, const float *py, const float *pw,
            const int *inl, int nInl, int fit_quad,
            float &a2, float &a1, float &a0);

// 中心点链描述性拟合（仅用于对外输出，不参与判据）：
//   MSAC（抛物线 3 点 / 直线 2 点采样，Cramer 精确解）模型评分按
//   最长连续内点段 → 内点数 → sse 字典序，全链连续内点提前退出；
//   内点上迭代最小一乘（L1/IRLS）精修，对残留偏差稳健。
// 模型：c = a2·t² + a1·t + a0，t 恒为 y（车道线 x(y) 单值的几何先验），
// 直车道线的 a2 自然收敛为 ≈0。全栈数组，零堆分配。
// fit_rot 开启时，在竖直主轴结果基础上追加旋转搜索：以 θ 为变量在
// [-60°,60°] 范围内做黄金分割极小化 L1 成本 E(θ)（每次 E(θ) 调用在
// 旋转局部系中做快装 IRLS 拟合），最终在最优 θ* 处做完整 IRLS 精修
// 输出 ra/rb/rc（可贴合急弯或斜向车道线的真实走向）。
void describeCurvePts(const cv::Point2f *pts, int n,
                      float msac_thresh, int msac_iters, int fit_quad,
                      int fit_rot, CurveFit &out)
{
    float a2 = 0.f, a1 = 0.f, a0 = 0.f, rms = 0.f;
    float t_min = 0.f, t_max = 0.f;
    if (n <= 0)
    {
        out.t_min = out.t_max = 0.f;
        return;
    }
    if (n > kMaxChainPts)
        n = kMaxChainPts; // 防御性钳位（正常单线节点数远低于此）
    // 几何先验：车道线在图像内 x(y) 单值（内点节点序列 y 非严格单调
    // ——交叉截面同窗多节点/同 y 并列，MSAC 与 IRLS 求解均不依赖
    // 单调性），恒用 x(y) 参数化（抛物线开口向左右）

    // 点表 float 化（float 累积误差对拟合输出的影响低于阈值粒度；
    // 换来 MSAC 残差评估 AVX2 8 宽 FMA）
    float vt[kMaxChainPts], vc[kMaxChainPts];
    for (int i = 0; i < n; ++i)
    {
        vt[i] = pts[i].y;
        vc[i] = pts[i].x;
    }
    if (n == 1)
    {
        out.a0 = vc[0];
        out.t_min = out.t_max = vt[0];
        return;
    }

    // 内点集选择（MSAC）+ 内点上迭代最小一乘精修，全在栈数组上完成。
    // st/sc 为内点上一次项和（均权起步解用），随内点表重建同遍累加
    int inl[kMaxChainPts];
    int nInl = 0;
    float st = 0, sc = 0;
    if (n >= 4)
    {
        // 确定性伪随机（LCG），同输入同结果
        unsigned seed = 0x9E3779B9u;
        auto rnd = [&]()
        {
            seed = seed * 1664525u + 1013904223u;
            return seed >> 8;
        };
        float bestSse = 1e18f;
        float bestA[3] = {0.f, 0.f, 0.f}; // 最优模型系数（延迟重建内点表）
        int bestCnt = 0;
        int bestRun = 0;                  // 最优模型的最长连续内点段
        // 逐点残差与内点掩码缓冲：残差评估 AVX2 8 宽 FMA（显式
        // intrinsics，不依赖编译开关），统计同序标量累加
        float rr[kMaxChainPts];
        unsigned char mk[kMaxChainPts];
        for (int it = 0; it < msac_iters; ++it)
        {
            const int i0 = (int)(rnd() % (unsigned)n);
            const int i1 = (int)(rnd() % (unsigned)n);
            float rA2 = 0.f, rA1 = 0.f, rA0 = 0.f;
            if (fit_quad)
            {
                // 过 3 点的抛物线（克拉默法则；范德蒙行列式退化为共线/重
                // t）。float 域退行列式阈值放宽到 1e-3（量纲 ~1e7，仅
                // 用于剔除共线退化样本）
                const int i2 = (int)(rnd() % (unsigned)n);
                if (i0 == i1 || i0 == i2 || i1 == i2)
                    continue;
                const float t0 = vt[i0], t1 = vt[i1], t2 = vt[i2];
                const float c0 = vc[i0], c1 = vc[i1], c2 = vc[i2];
                const float t00 = t0 * t0, t10 = t1 * t1, t20 = t2 * t2;
                const float det = t00 * (t1 - t2) - t0 * (t10 - t20) +
                                  (t10 * t2 - t1 * t20);
                if (std::fabs(det) < 1e-3f)
                    continue;
                rA2 = (c0 * (t1 - t2) - t0 * (c1 - c2) +
                       (c1 * t2 - t1 * c2)) / det;
                rA1 = (t00 * (c1 - c2) - c0 * (t10 - t20) +
                       (t10 * c2 - c1 * t20)) / det;
                rA0 = (t00 * (t1 * c2 - c1 * t2) -
                       t0 * (t10 * c2 - c1 * t20) +
                       c0 * (t10 * t2 - t1 * t20)) / det;
            }
            else
            {
                // 过 2 点的直线
                if (i0 == i1)
                    continue;
                const float dt = vt[i1] - vt[i0];
                if (std::fabs(dt) < 1e-6f)
                    continue;
                rA1 = (vc[i1] - vc[i0]) / dt;
                rA0 = vc[i0] - rA1 * vt[i0];
            }
#if CLU_AVX2
            // 第一遍（向量化）：逐点残差 + 内点掩码，8 点/迭代
            {
                const __m256 vA2 = _mm256_set1_ps(rA2);
                const __m256 vA1 = _mm256_set1_ps(rA1);
                const __m256 vA0 = _mm256_set1_ps(rA0);
                const __m256 vTh = _mm256_set1_ps(msac_thresh);
                const __m256 sgn = _mm256_set1_ps(-0.f);
                int i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    const __m256 vt8 = _mm256_loadu_ps(vt + i);
                    const __m256 vc8 = _mm256_loadu_ps(vc + i);
                    // pred = a2·t² + (a1·t + a0)，r = c - pred
                    const __m256 pred = _mm256_fmadd_ps(
                        vA2, _mm256_mul_ps(vt8, vt8),
                        _mm256_fmadd_ps(vA1, vt8, vA0));
                    const __m256 r = _mm256_sub_ps(vc8, pred);
                    _mm256_storeu_ps(rr + i, r);
                    const __m256 ar = _mm256_andnot_ps(sgn, r);
                    const unsigned msk = (unsigned)_mm256_movemask_ps(
                        _mm256_cmp_ps(ar, vTh, _CMP_LE_OS));
                    for (int q = 0; q < 8; ++q)
                        mk[i + q] = (unsigned char)((msk >> q) & 1);
                }
                for (; i < n; ++i)
                {
                    const float r = vc[i] - (rA2 * vt[i] * vt[i] +
                                             rA1 * vt[i] + rA0);
                    rr[i] = r;
                    mk[i] = std::fabs(r) <= msac_thresh;
                }
            }
#else
            for (int i = 0; i < n; ++i)
            {
                const float r = vc[i] - (rA2 * vt[i] * vt[i] +
                                         rA1 * vt[i] + rA0);
                rr[i] = r;
                mk[i] = std::fabs(r) <= msac_thresh;
            }
#endif
            // 第二遍：同序累加——最长连续内点段（连续性优先）+ 内点数
            //（内点率）+ sse
            int run = 0, maxRun = 0, cnt = 0;
            float sse = 0;
            for (int i = 0; i < n; ++i)
            {
                if (mk[i])
                {
                    ++cnt;
                    ++run;
                    if (run > maxRun)
                        maxRun = run;
                    sse += rr[i] * rr[i];
                }
                else
                {
                    run = 0;
                }
            }
            // 模型评分（字典序）：最长连续段 → 内点数（率） → sse
            if (maxRun > bestRun ||
                (maxRun == bestRun &&
                 (cnt > bestCnt || (cnt == bestCnt && sse < bestSse))))
            {
                bestRun = maxRun;
                bestCnt = cnt;
                bestSse = sse;
                bestA[0] = rA2;
                bestA[1] = rA1;
                bestA[2] = rA0;
            }
            if (maxRun == n)
                break; // 全链连续内点，提前退出（干净链首个有效采样即命中）
        }
        // 用最优模型重建内点表，并同遍累加边界与一次项和（同序累加，
        // 省两遍内点遍历）
        if (bestCnt >= 2 && bestRun >= 2)
        {
            t_min = 1e9f;
            t_max = -1e9f;
            for (int i = 0; i < n; ++i)
            {
                const float r = vc[i] - (bestA[0] * vt[i] * vt[i] +
                                         bestA[1] * vt[i] + bestA[2]);
                if (std::fabs(r) <= msac_thresh)
                {
                    inl[nInl++] = i;
                    st += vt[i];
                    sc += vc[i];
                    if (vt[i] < t_min)
                        t_min = vt[i];
                    if (vt[i] > t_max)
                        t_max = vt[i];
                }
            }
        }
    }
    if (nInl == 0)
    {
        t_min = 1e9f;
        t_max = -1e9f;
        for (int i = 0; i < n; ++i)
        {
            inl[nInl++] = i;
            st += vt[i];
            sc += vc[i];
            if (vt[i] < t_min)
                t_min = vt[i];
            if (vt[i] > t_max)
                t_max = vt[i];
        }
    }
    // ---- 内点上迭代最小一乘（L1/IRLS）精修（float 域）----
    // 第 1 步：以最小二乘解为基础（均权起步解，沿用完整二次法方程，
    // 病态时降为直线解）。中心化后 Σu=0、Σv=0，但 u² 列与常数列并不正交（Σu²≠0）——二次
    // 拟合必须保留常数项 b0，否则曲率系统性减半。消去 b0
    // （b0 = −su2/n·b2）得二元方程组：
    //   (su4 − su2²/n)·b2 + su3·b1 = su2c
    //   su3·b2 + su2·b1 = suc
    // 直线模型（fit_quad=0）：b2=0，一元解 b1 = suc/su2。
    // tc/cc 为内点中心（中心化后 Σu=0、Σv=0）
    const float tc = st / nInl, cc = sc / nInl;
    float su2 = 0, su3 = 0, su4 = 0, suc = 0, su2c = 0;
    for (int i = 0; i < nInl; ++i)
    {
        const float u = vt[inl[i]] - tc, v = vc[inl[i]] - cc;
        su2 += u * u;
        su3 += u * u * u;
        su4 += u * u * u * u;
        suc += u * v;
        su2c += u * u * v;
    }
    const float nn = (float)nInl;
    const float k1 = su2 > 1e-9f ? suc / su2 : 0.f;
    a2 = 0.f;
    a1 = k1;
    a0 = cc - k1 * tc;
    if (fit_quad)
    {
        const float s22 = su2 - su2 * su2 / nn; // = n·Var(u)
        float m11 = su4 - su2 * su2 / nn;
        m11 += 1e-6f * s22; // 岭正则化（相对量级，防病态）
        const float det = m11 * su2 - su3 * su3;
        if (std::fabs(det) > 1e-6f)
        {
            const float qq2 = (su2c * su2 - suc * su3) / det;
            const float qq1 = (m11 * suc - su3 * su2c) / det;
            const float qq0 = -(su2 / nn) * qq2;
            a2 = qq2;
            a1 = qq1 - 2.f * qq2 * tc;
            a0 = cc - qq1 * tc + qq2 * tc * tc + qq0;
        }
    }
    // 第 2 步：在最小二乘基础上做最小一乘（IRLS）迭代精修——目标
    // min Σ|r|，对残留外点/尾部偏差比最小二乘稳健。权重
    // w_i = 1/max(|r_i|, δ)：残差小的内点主导模型。固定迭代上限 +
    // 系数早退，确定性同 MSAC（同输入同输出）；每轮仅对少数内点做
    // 两遍累加，开销与 MSAC 采样轮同量级且链短
    {
        constexpr int kL1Iters = 6;      // IRLS 迭代上限（链内点数十余，
                                         // 干净链 2~3 轮即收敛）
        constexpr float kL1Floor = 0.1f; // 残差权重下限（px）：|r|<δ 视为
                                         // 精确贴合，权重上限 1/δ，防除零
        constexpr float kL1Conv = 1e-3f; // 系数收敛早退阈值（px 量纲）
        float iw[kMaxChainPts];
        for (int it = 0; it < kL1Iters; ++it)
        {
            for (int i = 0; i < nInl; ++i)
            {
                const float t = vt[inl[i]];
                const float r = vc[inl[i]] -
                                (a2 * t * t + a1 * t + a0);
                iw[inl[i]] = 1.f / std::max(std::fabs(r), kL1Floor);
            }
            float na2 = a2, na1 = a1, na0 = a0;
            wlsFit(vc, vt, iw, inl, nInl, fit_quad, na2, na1, na0);
            const bool conv =
                std::fabs(na2 - a2) <= kL1Conv &&
                std::fabs(na1 - a1) <= kL1Conv &&
                std::fabs(na0 - a0) <= kL1Conv;
            a2 = na2;
            a1 = na1;
            a0 = na0;
            if (conv)
                break;
        }
    }
    float sres = 0;
    float fixedL1 = 0.f;
    for (int i = 0; i < nInl; ++i)
    {
        const float vf = vt[inl[i]];
        const float r = vc[inl[i]] - (a2 * vf * vf + a1 * vf + a0);
        sres += r * r;
        fixedL1 += std::fabs(r);
    }
    rms = std::sqrt(sres / nInl);
    fixedL1 /= nInl;
    // ---- 旋转主轴搜索（fit_rot 开启）：θ ∈ [-θmax,θmax] 黄金分割极小化 ----
    // 局部参考系 (u,v)：u = x·cosθ + y·sinθ，v = -x·sinθ + y·cosθ；
    // 拟合 u = ra·v² + rb·v + rc（L1/IRLS），E(θ) = Σ|r|（内点上）。
    // 黄金分割每次仅需旋转 + 加权求解，单链 nInl ≤ kMaxChainPts，
    // 代价可忽略不计。
    float thetaDeg = 0.f, rotA = 0.f, rotB = 0.f, rotC = 0.f;
    // 早退：竖直主轴拟合已足够贴合（平均 |r| 低于阈值）时，旋转带来的
    // 增益可忽略（直/近直车道），直接跳过旋转搜索，输出保持 θ=0 且
    // ra/rb/rc 全 0，绘制自动回退到竖直主轴。
    constexpr float kSkipRotL1 = 0.30f; // 平均 |r| 阈值（px）
    if (fit_rot && nInl >= 4 && fixedL1 >= kSkipRotL1)
    {
        constexpr float kThetaMax = 60.f * (float)CV_PI / 180.f; // ±60°
        constexpr int kGSCnt = 15;       // 黄金分割迭代：0.618^14×120°≈0.4°
                                         // 精度对描述性 θ 足够，多于无益
        constexpr int kSearchIrls = 0;   // 搜索期 IRLS 轮数：0 = 仅均权 LSQ
                                         // 排序（最终在 θ* 处单独做完整 L1）
        constexpr int kFinalIrls = 6;    // 最终精修 IRLS 轮数（同竖直主轴）
        constexpr float kL1Floor = 0.1f; // 残差权重下限（px）
        constexpr float kL1Conv = 1e-3f; // 系数收敛早退阈值
        // 旋转坐标临时数组（仅占栈，nInl ≤ kMaxChainPts）
        float vu[kMaxChainPts]; // u 坐标（旋转后 x 方向）
        float vv[kMaxChainPts]; // v 坐标（旋转后 y 方向）
        int io[kMaxChainPts];   // 连续索引 [0,1,...,nInl-1]
        for (int q = 0; q < nInl; ++q)
            io[q] = q;
        float iw[kMaxChainPts]; // IRLS 权重
        // 快速旋转拟合评估：给定 θ → 旋转内点 → IRLS → 返回 Σ|r|
        auto rotEval = [&](float th, int irlsIters, float &oA, float &oB,
                           float &oC) -> float
        {
#ifdef FML_PROF_ROT
            unsigned _px; const unsigned long long _e0 = __rdtscp(&_px);
#endif
            const float cs = std::cos(th), sn = std::sin(th);
            for (int q = 0; q < nInl; ++q)
            {
                const float x = vc[inl[q]], y = vt[inl[q]];
                vu[q] = x * cs + y * sn;
                vv[q] = -x * sn + y * cs;
            }
            // 均权 LSQ 起步
            for (int q = 0; q < nInl; ++q)
                iw[q] = 1.f;
            float fA = 0.f, fB = 0.f, fC = 0.f;
            wlsFit(vu, vv, iw, io, nInl, fit_quad, fA, fB, fC);
            // IRLS 迭代（与竖直主轴完全同构）
            for (int it = 0; it < irlsIters; ++it)
            {
                for (int q = 0; q < nInl; ++q)
                {
                    const float v = vv[q];
                    const float r = vu[q] - (fA * v * v + fB * v + fC);
                    iw[q] = 1.f / std::max(std::fabs(r), kL1Floor);
                }
                float nA = fA, nB = fB, nC = fC;
                wlsFit(vu, vv, iw, io, nInl, fit_quad, nA, nB, nC);
                const bool conv =
                    std::fabs(nA - fA) <= kL1Conv &&
                    std::fabs(nB - fB) <= kL1Conv &&
                    std::fabs(nC - fC) <= kL1Conv;
                fA = nA;
                fB = nB;
                fC = nC;
                if (conv)
                    break;
            }
            oA = fA;
            oB = fB;
            oC = fC;
            // L1 成本（Σ|r|）
            float cost = 0.f;
            for (int q = 0; q < nInl; ++q)
            {
                const float v = vv[q];
                cost += std::fabs(vu[q] - (fA * v * v + fB * v + fC));
            }
#ifdef FML_PROF_ROT
            g_rotProf.eval_ticks += __rdtscp(&_px) - _e0;
            g_rotProf.evals++;
#endif
            return cost;
        };
        // 黄金分割：区间 [lo,hi]，每次 f(θ1) < f(θ2) 则缩右，否则缩左
        const float gr = 0.6180339887498949f; // φ-1
        float lo = -kThetaMax, hi = kThetaMax;
        float t1 = hi - gr * (hi - lo);
        float t2 = lo + gr * (hi - lo);
        float f1A, f1B, f1C;
        float f1 = rotEval(t1, kSearchIrls, f1A, f1B, f1C);
        float f2A, f2B, f2C;
        float f2 = rotEval(t2, kSearchIrls, f2A, f2B, f2C);
#ifdef FML_PROF_ROT
        unsigned _px; const unsigned long long _gs0 = __rdtscp(&_px);
#endif
        for (int it = 0; it < kGSCnt; ++it)
        {
            if (f1 < f2)
            {
                hi = t2;
                t2 = t1;
                f2A = f1A;
                f2B = f1B;
                f2C = f1C;
                f2 = f1;
                t1 = hi - gr * (hi - lo);
                f1 = rotEval(t1, kSearchIrls, f1A, f1B, f1C);
            }
            else
            {
                lo = t1;
                t1 = t2;
                f1A = f2A;
                f1B = f2B;
                f1C = f2C;
                f1 = f2;
                t2 = lo + gr * (hi - lo);
                f2 = rotEval(t2, kSearchIrls, f2A, f2B, f2C);
            }
        }
        // 最优角：取 f1/f2 较小侧
        const float thOpt = (f1 < f2) ? t1 : t2;
#ifdef FML_PROF_ROT
        g_rotProf.gs_ticks += __rdtscp(&_px) - _gs0;
        unsigned long long _f0 = __rdtscp(&_px);
#endif
        // 最终精修（完整 IRLS 轮数）；rotEval 结束时 vu/vv 即 θ* 系
        // 内点坐标，直接复用做形态校验
        rotEval(thOpt, kFinalIrls, rotA, rotB, rotC);
#ifdef FML_PROF_ROT
        g_rotProf.final_ticks += __rdtscp(&_px) - _f0;
        g_rotProf.calls++;
#endif
        // 病态回折（hairpin）防护：车道线在任意旋转系下不可回折——
        // 抛物线顶点落在内点 v 域内部且极值超出内点 u 包络时，弧线
        // 在支撑区内鼓出数百像素而残差仍可为零（内点分居 v 域两端，
        // 弧从中间绕过去）。检出即拒绝旋转解，回退竖直主轴（ra/rb/rc
        // 清零 + θ=0，绘制自动走竖直分支按 t_min/t_max 截断）
        thetaDeg = thOpt * 180.f / (float)CV_PI;
        if (rotA != 0.f)
        {
            float vmin = 1e30f, vmax = -1e30f, umin = 1e30f, umax = -1e30f;
            for (int q = 0; q < nInl; ++q)
            {
                vmin = std::min(vmin, vv[q]);
                vmax = std::max(vmax, vv[q]);
                umin = std::min(umin, vu[q]);
                umax = std::max(umax, vu[q]);
            }
            const float vStar = -rotB / (2.f * rotA);
            if (vStar > vmin && vStar < vmax)
            {
                const float uStar = rotA * vStar * vStar + rotB * vStar + rotC;
                constexpr float kHairpinMargin = 10.f; // px
                if (uStar > umax + kHairpinMargin ||
                    uStar < umin - kHairpinMargin)
                {
                    rotA = 0.f;
                    rotB = 0.f;
                    rotC = 0.f;
                    thetaDeg = 0.f;
                }
            }
        }
    }
    // ---- 抛物线/直线结果拷出 ----
    out.a2 = a2;
    out.a1 = a1;
    out.a0 = a0;
    out.t_min = t_min;
    out.t_max = t_max;
    out.rms = rms;
    out.theta = thetaDeg;
    out.ra = rotA;
    out.rb = rotB;
    out.rc = rotC;
}

// 单簇 indices 去重用的标记位图（版本号打点，免清零）
struct IdStamp
{
    std::vector<int> mark;
    int tag = 0;
};

// 位图去重：升序输出唯一成员下标（与 sort+unique 结果完全一致），
// 复杂度 O(成员数 + 最大下标)，替代逐簇拷贝排序
inline void dedupIndices(std::vector<int> &out, const int *ids, int cnt,
                         IdStamp &st, int n)
{
    if ((int)st.mark.size() < n)
        st.mark.assign(n, 0);
    if (++st.tag == 0) // int 回绕防护（工程上不可达）
    {
        std::fill(st.mark.begin(), st.mark.end(), 0);
        st.tag = 1;
    }
    int maxId = -1; // 扫描上界收紧到实际触达的最大下标
    for (int q = 0; q < cnt; ++q)
    {
        const int id = ids[q];
        st.mark[id] = st.tag;
        if (id > maxId)
            maxId = id;
    }
    out.clear();
    for (int id = 0; id <= maxId; ++id)
        if (st.mark[id] == st.tag)
            out.push_back(id);
}

// 内点上的权重加权最小二乘（中心化消去常数列 + 相对量级岭正则；
// 解奇异保持入参模型不动）。两处共用同一法方程：
//   - 提取层精修：权重取节点测量权重 w，长而置信高的线段主导模型；
//   - 链拟合 describeCurvePts 的 IRLS 各步：权重取 1/max(|r|, δ)。
// 节点数据经 px/py/pw（SoA 连续存储）与 inl 下标索引（同域）。
// fit_quad=1 抛物线 / 0 直线
void wlsFit(const float *px, const float *py, const float *pw,
            const int *inl, int nInl, int fit_quad,
            float &a2, float &a1, float &a0)
{
    float sw = 0, st = 0, sc = 0;
    for (int i = 0; i < nInl; ++i)
    {
        const float w = pw[inl[i]];
        sw += w;
        st += w * py[inl[i]];
        sc += w * px[inl[i]];
    }
    if (!(sw > 0.f) || nInl < 2)
        return;
    // 加权中心（中心化后 Σw·u = 0、Σw·v = 0，常数项解耦）
    const float tc = st / sw, cc = sc / sw;
    float u2 = 0, u3 = 0, u4 = 0, uc = 0, u2c = 0;
    for (int i = 0; i < nInl; ++i)
    {
        const float w = pw[inl[i]];
        const float u = py[inl[i]] - tc, v = px[inl[i]] - cc;
        u2 += w * u * u;
        u3 += w * u * u * u;
        u4 += w * u * u * u * u;
        uc += w * u * v;
        u2c += w * u * u * v;
    }
    if (fit_quad)
    {
        const float s22 = u2 - u2 * u2 / sw;
        float m11 = u4 - u2 * u2 / sw;
        m11 += 1e-6f * s22; // 岭正则化（相对量级，防病态）
        const float det = m11 * u2 - u3 * u3;
        if (std::fabs(det) > 1e-6f)
        {
            const float b2 = (u2c * u2 - uc * u3) / det;
            const float b1 = (m11 * uc - u3 * u2c) / det;
            const float b0 = -(u2 / sw) * b2;
            a2 = b2;
            a1 = b1 - 2.f * b2 * tc;
            a0 = cc - b1 * tc + b2 * tc * tc + b0;
        }
    }
    else
    {
        const float b1 = u2 > 1e-9f ? uc / u2 : 0.f;
        a2 = 0.f;
        a1 = b1;
        a0 = cc - b1 * tc;
    }
}

// 单个候选线的输出构建（独立于其他候选线）：直写输出对象，conf
// 均值直接从内点节点读取（不落地中间副本）
void buildClu(FmlSegCluster &c, const CandLine &cl,
              const std::vector<Node> &nodes, const int *inl,
              const std::vector<int> &pool, const fml_cluster_params &p,
              int k, int H, IdStamp &st, int nSeg)
{
    // 本函数专属复用缓冲（线程本地，容量跨帧保持）
    static thread_local std::vector<int> idsBuf, orderBuf;
    // indices：内点节点成员线段去重升序（诊断/回退渲染用）
    idsBuf.clear();
    for (int q = 0; q < cl.cnt; ++q)
    {
        const Node &nd = nodes[inl[q]];
        idsBuf.insert(idsBuf.end(), pool.begin() + nd.off,
                      pool.begin() + nd.off + nd.cnt);
    }
    dedupIndices(c.indices, idsBuf.data(), (int)idsBuf.size(), st, nSeg);
    // centers：内点节点按 y 降序（自底向上，与链渲染/延展语义一致；
    // 同 y 并列按 x 升序，保证确定性输出）
    orderBuf.assign(inl, inl + cl.cnt);
    std::sort(orderBuf.begin(), orderBuf.end(),
              [&](int a, int b)
              {
                  if (nodes[a].y != nodes[b].y)
                      return nodes[a].y > nodes[b].y;
                  return nodes[a].x < nodes[b].x;
              });
    c.centers.clear();
    for (int q : orderBuf)
        c.centers.push_back({nodes[q].x, nodes[q].y});
    // 链锯齿形态门限（拒绝双缘混线）：真实车道线 x(y) 平滑，个别窗口
    // 因干扰出现孤立折返可容忍；两条近平行边被同一模型收编时，链在
    // 左右两簇间持续交替——"相邻差分反号且双侧幅值超阈"的折返连续
    // 出现。连续折返 ≥3（4 个相邻差分左右交替）判为锯齿链，
    // valid=false（灰显 + 打分打折，与 top_noise 同处理；不删除，
    // 保持其余簇排名稳定）。阈值不入参数面：4px 幅阈高于节点抖动
    // （≤2px）；正常干扰线最长连续折返 ≤2，双缘混线 ≥3
    bool zigzag = false;
    {
        const int nC = (int)c.centers.size();
        if (nC >= 6)
        {
            int run = 0, bestRun = 0;
            for (int i = 1; i + 1 < nC; ++i)
            {
                const float d0 = c.centers[i].x - c.centers[i - 1].x;
                const float d1 = c.centers[i + 1].x - c.centers[i].x;
                if (d0 * d1 < 0.f && std::fabs(d0) > 4.f &&
                    std::fabs(d1) > 4.f)
                {
                    ++run;
                    bestRun = std::max(bestRun, run);
                }
                else
                    run = 0;
            }
            zigzag = bestRun >= 3;
        }
    }
    // top_noise：底端节点（y 最大，orderBuf 首元）的播种位置（成员
    // 最大 ty）在顶部噪声带内 ⟺ 整线居顶部噪声带（链底端即播种端）
    c.top_noise =
        p.top_noise_ratio > 0.f &&
        nodes[orderBuf.front()].seed_y < (float)H * p.top_noise_ratio;
    CurveFit f;
    describeCurvePts(c.centers.data(), (int)c.centers.size(),
                     p.msac_thresh, p.msac_iters, p.fit_quad, p.fit_rot, f);
    c.a2 = f.a2;
    c.a1 = f.a1;
    c.a0 = f.a0;
    c.t_min = f.t_min;
    c.t_max = f.t_max;
    c.rms = f.rms;
    c.theta = f.theta;
    c.ra = f.ra;
    c.rb = f.rb;
    c.rc = f.rc;
    // 评分：0.28 链长 + 0.22 端到端延展 + 0.50 log10(1+conf 均值)
    c.length = (float)c.centers.size() * (float)k;
    const cv::Point2f &p0 = c.centers.front();
    const cv::Point2f &p1 = c.centers.back();
    const float ex = p0.x - p1.x, ey = p0.y - p1.y;
    const float extent = std::sqrt(ex * ex + ey * ey);
    float cm = 0;
    for (int q = 0; q < cl.cnt; ++q)
        cm += nodes[inl[q]].conf;
    cm /= (float)std::max(cl.cnt, 1);
    c.conf = cm;
    const float s_sup = std::min(1.f, c.length / 130.f);
    const float s_ext = std::min(1.f, extent / 190.f);
    const float conf_t = cluLog10F(1.f + std::max(cm, 0.f));
    c.score = 0.28f * s_sup + 0.22f * s_ext + 0.50f * conf_t;
    c.valid = c.length >= p.min_length && !c.top_noise && !zigzag;
    if (!c.valid)
        c.score *= 0.25f;
}

// ---- 外部打分器（lane_scorer 三档）----
// 特征布局与训练一致（FML_LANE_SCORER_NFEAT=24），索引固定：
//   0 npts  1 length  2 score(加法口径)  3 conf  4 sumc  5 theta  6 |a1|
//   7 a2    8 |a2|·ySpan  9 |a0-中心|  10 ySpan  11 cx_mean  12 cx_std
//   13 cx_range  14 xTop  15 xBot  16 tilt  17 maxYgap  18 nbGap
//   19 ptDensity  20 relScore  21 relSumc  22 rankNorm  23 tiltAbs
//
// centers 为自底向上（centers[0] 是 y 最大的底部节点）。
static void extractScorerFeats(const FmlSegCluster &c, int rank, int nclu,
                               float maxScore, float maxSumc,
                               float imgW, float imgH, float bands,
                               float *f)
{
    const int n = (int)c.centers.size();
    f[0] = (float)n;                 // npts
    f[1] = c.length;                 // length
    f[2] = c.score;                  // score（加法口径）
    f[3] = c.conf;                   // conf
    const float sumc = c.conf * (float)std::max(n, 0);
    f[4] = sumc;                     // sumc
    f[5] = c.theta;                  // theta
    f[6] = std::fabs(c.a1);          // |a1|
    f[7] = c.a2;                     // a2
    // ySpan / x 统计
    float yBot = 0, yTop = 0, xBot = 0, xTop = 0;
    float cxmin = 1e30f, cxmax = -1e30f;
    double cxsum = 0, cx2sum = 0;
    float maxYgap = 0; int nbGap = 0;
    const float bandH = std::max(1.f, imgH / std::max(1.f, bands));
    for (int i = 0; i < n; ++i)
    {
        const cv::Point2f &pt = c.centers[i];
        if (i == 0)
        {
            yBot = pt.y;
            xBot = pt.x;
        }
        if (i == n - 1)
        {
            yTop = pt.y;
            xTop = pt.x;
        }
        cxmin = std::min(cxmin, pt.x);
        cxmax = std::max(cxmax, pt.x);
        cxsum += pt.x;
        cx2sum += (double)pt.x * pt.x;
        if (i + 1 < n)
        {
            const float dy = c.centers[i].y - c.centers[i + 1].y;
            maxYgap = std::max(maxYgap, dy);
            if (dy > 2.f * bandH)
                ++nbGap;
        }
    }
    const float ySpan = std::max(1.f, yBot - yTop);
    const float cxmean = (float)(cxsum / std::max(1, n));
    const float cxstd = (float)std::sqrt(std::max(0.0,
        cx2sum / (double)std::max(1, n) - (double)cxmean * cxmean));
    const float tilt = (ySpan >= 1.f) ? (xBot - xTop) / ySpan : 0.f;
    f[8] = std::fabs(c.a2) * ySpan;          // |a2|·ySpan
    f[9] = std::fabs(c.a0 - 0.5f * imgW);    // |a0 - 中心|
    f[10] = ySpan;                            // ySpan
    f[11] = cxmean;                           // cx_mean
    f[12] = cxstd;                            // cx_std
    f[13] = std::max(0.f, cxmax - cxmin);     // cx_range
    f[14] = xTop;                             // xTop
    f[15] = xBot;                             // xBot
    f[16] = tilt;                             // tilt_dxdy
    f[17] = maxYgap;                          // maxYgap
    f[18] = (float)nbGap;                     // nbGap
    f[19] = (float)n / ySpan;                 // ptDensity
    f[20] = maxScore > 1e-6f ? c.score / maxScore : 0.f;  // relScore
    f[21] = maxSumc > 1e-6f ? sumc / maxSumc : 0.f;       // relSumc
    f[22] = nclu > 0 ? (float)rank / (float)nclu : 0.f;   // rankNorm
    f[23] = std::fabs(tilt);                  // tiltAbs
}

// M5 乘法核打分（可解释/轻量档）
static float score_M5(const fml_lane_scorer_weights &w, const float *f)
{
    const float sumc = std::max(f[4], 0.f);
    const float lenw = f[1] / std::max(1.f, 0.5f * w.img_h);
    const float capp = std::min(lenw, w.m5.cap);
    const float logc = std::log10(1.f + sumc);
    float s = std::pow(logc, w.m5.a) * std::pow(capp, w.m5.b);
    // rank 软罚：rankNorm 超过 1/3（= 帧内第 4 条起）减分
    const float rnPen = std::max(0.f, f[22] - (1.f / 3.f));
    s -= w.m5.rank_w * rnPen;
    return s;
}

// 线性白盒打分（LR / LinearSVM 共用 +b）
// AVX2：24 维点积 = 8×3 无余数；非 AVX2 回退标量循环
static float score_linear(const fml_lane_scorer_weights &w, const float *f)
{
#ifdef CLU_AVX2
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < 24; i += 8)
    {
        const __m256 vf = _mm256_loadu_ps(f + i);
        const __m256 vm = _mm256_loadu_ps(w.mu + i);
        const __m256 vi = _mm256_loadu_ps(w.istd + i);
        const __m256 vw = _mm256_loadu_ps(w.w + i);
        s = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(vf, vm), vi), vw, s);
    }
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    __m128 h = _mm_add_ps(lo, hi);
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h) + w.b;
#else
    float s = w.b;
    for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
        s += w.w[i] * ((f[i] - w.mu[i]) * w.istd[i]);
    return s;
#endif
}

// RBF-SVM 打分（效果最强档）
// AVX2：24 维平方距离 = 8×3 无余数，横向化简后单次 expf；其余同标量
static float score_rbf(const fml_lane_scorer_weights &w, const float *f)
{
#ifdef CLU_AVX2
    const float z[24] = {
        (f[0] - w.mu[0]) * w.istd[0],  (f[1] - w.mu[1]) * w.istd[1],
        (f[2] - w.mu[2]) * w.istd[2],  (f[3] - w.mu[3]) * w.istd[3],
        (f[4] - w.mu[4]) * w.istd[4],  (f[5] - w.mu[5]) * w.istd[5],
        (f[6] - w.mu[6]) * w.istd[6],  (f[7] - w.mu[7]) * w.istd[7],
        (f[8] - w.mu[8]) * w.istd[8],  (f[9] - w.mu[9]) * w.istd[9],
        (f[10] - w.mu[10]) * w.istd[10], (f[11] - w.mu[11]) * w.istd[11],
        (f[12] - w.mu[12]) * w.istd[12], (f[13] - w.mu[13]) * w.istd[13],
        (f[14] - w.mu[14]) * w.istd[14], (f[15] - w.mu[15]) * w.istd[15],
        (f[16] - w.mu[16]) * w.istd[16], (f[17] - w.mu[17]) * w.istd[17],
        (f[18] - w.mu[18]) * w.istd[18], (f[19] - w.mu[19]) * w.istd[19],
        (f[20] - w.mu[20]) * w.istd[20], (f[21] - w.mu[21]) * w.istd[21],
        (f[22] - w.mu[22]) * w.istd[22], (f[23] - w.mu[23]) * w.istd[23],
    };
    const __m256 vz0 = _mm256_loadu_ps(z);
    const __m256 vz1 = _mm256_loadu_ps(z + 8);
    const __m256 vz2 = _mm256_loadu_ps(z + 16);
    const __m256 vg = _mm256_set1_ps(-w.gamma);
    float s = w.b;
    for (int k = 0; k < w.n_sv; ++k)
    {
        const float *sv = w.sv.data() + (size_t)k * 24;
        const __m256 d0 = _mm256_sub_ps(vz0, _mm256_loadu_ps(sv));
        const __m256 d1 = _mm256_sub_ps(vz1, _mm256_loadu_ps(sv + 8));
        const __m256 d2 = _mm256_sub_ps(vz2, _mm256_loadu_ps(sv + 16));
        const __m256 q = _mm256_fmadd_ps(
            d0, d0, _mm256_fmadd_ps(d1, d1, _mm256_mul_ps(d2, d2)));
        __m128 lo = _mm256_castps256_ps128(q), hi = _mm256_extractf128_ps(q, 1);
        __m128 h = _mm_add_ps(lo, hi);
        h = _mm_hadd_ps(h, h);
        h = _mm_hadd_ps(h, h);
        const float dist = _mm_cvtss_f32(h);
        s += w.sv_coef[k] * std::exp(-w.gamma * dist);
    }
    return s;
#else
    float z[FML_LANE_SCORER_NFEAT];
    for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
        z[i] = (f[i] - w.mu[i]) * w.istd[i];
    float s = w.b;
    const int nsv = w.n_sv;
    for (int k = 0; k < nsv; ++k)
    {
        const float *sv = w.sv.data() + (size_t)k * FML_LANE_SCORER_NFEAT;
        float d = 0;
        for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
        {
            const float t = z[i] - sv[i];
            d += t * t;
        }
        s += w.sv_coef[k] * std::exp(-w.gamma * d);
    }
    return s;
#endif
}

// 按打分机型覆盖 score（若启用），输出帧内 max 供 rel 特征用
static void applyLaneScorer(std::vector<FmlSegCluster> &cls,
                            const fml_lane_scorer_weights *w,
                            float imgW, float imgH, float bands)
{
    if (cls.empty())
        return;
    const int nclu = (int)cls.size();
    // 帧内 max（用重排前的加法 score / sumc 计算 rel 特征，先后置）
    float maxScore = 0, maxSumc = 0;
    for (const FmlSegCluster &c : cls)
    {
        maxScore = std::max(maxScore, c.score);
        maxSumc = std::max(maxSumc,
                           c.conf * (float)std::max((int)c.centers.size(), 0));
    }
    float f[FML_LANE_SCORER_NFEAT];
    for (int i = 0; i < nclu; ++i)
    {
        FmlSegCluster &c = cls[i];
        extractScorerFeats(c, i, nclu, maxScore, maxSumc, imgW, imgH, bands, f);
        switch (w->mode)
        {
        case 1:
            c.score = score_M5(*w, f);
            break;
        case 2:
            c.score = score_linear(*w, f);
            break;
        case 3:
            c.score = score_rbf(*w, f);
            break;
        default:
            break;
        }
        // 打分器生效时分数即"是/否"判据：负分 = 判定为否，
        // 不再视为有效车道线（避免负 R 值仍显示绿色）
        c.valid = c.valid && c.score > 0.f;
    }
    // 重排
    std::sort(cls.begin(), cls.end(),
              [](const FmlSegCluster &a, const FmlSegCluster &b)
              {
                  return a.score > b.score;
              });
}

} // namespace

void fml_lane_scorer_init(fml_lane_scorer_weights &w, int mode,
                          const fml_m5_scorer_params &m5)
{
    w.mode = mode;
    w.img_h = 240.f;
    w.m5 = m5;
    for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
    {
        w.mu[i] = 0.f;
        w.istd[i] = 1.f;
        w.w[i] = 0.f;
    }
    w.b = 0.f;
    w.gamma = 0.f;
    w.n_sv = 0;
    w.sv.clear();
    w.sv_coef.clear();
    if (mode == 2) // LR-3（内置稀疏权重）
    {
        for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
        {
            w.mu[i] = kScorerLrMu24[i];
            w.istd[i] = kScorerLrIstd24[i];
            w.w[i] = kScorerLrW24[i];
        }
        w.b = kScorerLrB;
    }
    else if (mode == 3) // RBF-24（内置）
    {
        for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
        {
            w.mu[i] = kScorerRbfMu24[i];
            w.istd[i] = kScorerRbfIstd24[i];
        }
        w.gamma = kScorerRbfGamma;
        w.b = kScorerRbfB;
        w.n_sv = kScorerRbfNsv;
        w.sv.assign(kScorerRbfSv, kScorerRbfSv + kScorerRbfNsv * FML_LANE_SCORER_NFEAT);
        w.sv_coef.assign(kScorerRbfCoef, kScorerRbfCoef + kScorerRbfNsv);
    }
}

const fml_lane_scorer_weights &fml_lane_scorer_default(int mode)
{
    // 静态缓存内置权重（进程生命周期内仅初始化一次，按模式缓存）
    static bool done[4] = {false, false, false, false};
    static fml_lane_scorer_weights cache[4];
    const int m = (mode >= 1 && mode <= 3) ? mode : 1;
    if (!done[m])
    {
        fml_lane_scorer_init(cache[m], m);
        done[m] = true;
    }
    return cache[m];
}

int fml_lane_scorer_load(fml_lane_scorer_weights &w, const char *path)
{
    FILE *fp = std::fopen(path, "r");
    if (!fp)
        return 1;
    int mode = w.mode;
    float imgH = w.img_h;
    float b = w.b, gamma = w.gamma;
    float m5a = w.m5.a, m5b = w.m5.b, m5cap = w.m5.cap, m5rk = w.m5.rank_w;
    int nsv = w.n_sv;
    std::vector<float> mu(FML_LANE_SCORER_NFEAT, 0.f);
    std::vector<float> istd(FML_LANE_SCORER_NFEAT, 1.f);
    std::vector<float> ww(FML_LANE_SCORER_NFEAT, 0.f);
    for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
    {
        mu[i] = w.mu[i];
        istd[i] = w.istd[i];
        ww[i] = w.w[i];
    }
    std::vector<float> sv, svc;
    char key[64];
    while (std::fscanf(fp, " %63s", key) == 1)
    {
        if (0 == std::strcmp(key, "mode"))
            std::fscanf(fp, " %d", &mode);
        else if (0 == std::strcmp(key, "img_h"))
            std::fscanf(fp, " %f", &imgH);
        else if (0 == std::strcmp(key, "b"))
            std::fscanf(fp, " %f", &b);
        else if (0 == std::strcmp(key, "gamma"))
            std::fscanf(fp, " %f", &gamma);
        else if (0 == std::strcmp(key, "m5_a"))
            std::fscanf(fp, " %f", &m5a);
        else if (0 == std::strcmp(key, "m5_b"))
            std::fscanf(fp, " %f", &m5b);
        else if (0 == std::strcmp(key, "m5_cap"))
            std::fscanf(fp, " %f", &m5cap);
        else if (0 == std::strcmp(key, "m5_rank_w"))
            std::fscanf(fp, " %f", &m5rk);
        else if (0 == std::strcmp(key, "nsv"))
            std::fscanf(fp, " %d", &nsv);
        else if (0 == std::strcmp(key, "mu"))
            for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
                std::fscanf(fp, " %f", &mu[i]);
        else if (0 == std::strcmp(key, "istd"))
            for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
                std::fscanf(fp, " %f", &istd[i]);
        else if (0 == std::strcmp(key, "w"))
            for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
                std::fscanf(fp, " %f", &ww[i]);
        else if (0 == std::strcmp(key, "sv"))
        {
            sv.resize((size_t)std::max(0, nsv) * FML_LANE_SCORER_NFEAT);
            for (size_t i = 0; i < sv.size(); ++i)
                std::fscanf(fp, " %f", &sv[i]);
        }
        else if (0 == std::strcmp(key, "svc"))
        {
            svc.resize((size_t)std::max(0, nsv));
            for (size_t i = 0; i < svc.size(); ++i)
                std::fscanf(fp, " %f", &svc[i]);
        }
        else
        {
            // 未知键：跳过本行
            std::fscanf(fp, " %*[^\n]");
        }
    }
    std::fclose(fp);
    if (mode < 1 || mode > 3)
        return 2;
    w.mode = mode;
    w.img_h = imgH;
    w.m5.a = m5a;
    w.m5.b = m5b;
    w.m5.cap = m5cap;
    w.m5.rank_w = m5rk;
    for (int i = 0; i < FML_LANE_SCORER_NFEAT; ++i)
    {
        w.mu[i] = mu[i];
        w.istd[i] = istd[i];
        w.w[i] = ww[i];
    }
    w.b = b;
    w.gamma = gamma;
    w.n_sv = std::max(0, nsv);
    w.sv.swap(sv);
    w.sv_coef.swap(svc);
    return 0;
}

int fml_cluster_lines(const std::vector<cv::Vec4f> &lines,
                      const std::vector<float> &confs,
                      int width, int height,
                      std::vector<FmlSegCluster> &clusters,
                      const fml_cluster_params &p, fml_cluster_debug *dbg)
{
    (void)width; // 预留：当前判据仅使用高度（bands/top_noise 按高度换算）
    if (dbg)
    {
        dbg->y_lo.clear();
        dbg->y_hi.clear();
        dbg->centers.clear();
        dbg->matched.clear();
    }
    const int n = (int)lines.size();
    if (n == 0)
    {
        clusters.clear();
        return 0;
    }
    const int H = std::max(1, height);
    const int k = std::max(1, H / std::max(1, p.bands));
    const int nBands = H / k + 2;
    const float cosAng = std::cos(p.ang_tol * (float)CV_PI / 180.f);

    // ---- 复用缓冲（容量跨帧保持，稳态零堆分配）----
    static thread_local std::vector<Seg> segsBuf;
    static thread_local std::vector<int> bOff, bItem, bFill;
    // 全局节点表与成员扁平池（各窗口局部聚类封口产物；oMem 为窗口
    // 栈数组，出窗即失效，成员下标封口时拷入扁平池）
    static thread_local std::vector<Node> nodes;
    static thread_local std::vector<int> nodePool;
    // 提取产物：候选线表与内点节点下标表（CandLine.off/cnt 索引后者）
    static thread_local std::vector<CandLine> candLines;
    static thread_local std::vector<int> candInl;
    // 提取工作区：SoA 热数组（x/y/方向/权重连续存储 + nodes 下标
    // 映射，随硬删除收缩）与精修集/内点标记/打包排序表/断线切分
    // 缓冲（槽位域）
    static thread_local std::vector<float> eX, eY, eUx, eUy, eW;
    static thread_local std::vector<int> eIdx, inlBuf, inlTmp, packOrder;
    static thread_local std::vector<unsigned char> remark;
    // 各扫描带中心 y（入桶时求排序键 x(yc) 与窗口循环共用）
    static thread_local std::vector<float> ycs;
    // bItem 同位排序键（x(yc)，入桶时求出）
    static thread_local std::vector<float> bKeys;
    // 输出 indices 位图去重工作区
    static thread_local IdStamp idStamp;

    nodes.clear();
    nodePool.clear();
    candLines.clear();
    candInl.clear();

    segsBuf.resize(n);
    // ---- 预处理：上端点归一、单位方向、太水平剔除、带区间 ----
    // 太水平门限的判据 |ψ| <= max_psi 等价于单位向量 uy >= cos(max_psi)
    const float cosMaxPsi = std::cos(std::max(0.f, std::min(90.f,
                                                    p.max_psi)) *
                                     (float)CV_PI / 180.f);
    for (int i = 0; i < n; ++i)
    {
        Seg &s = segsBuf[i];
        s.x1 = lines[i][0];
        s.y1 = lines[i][1];
        s.x2 = lines[i][2];
        s.y2 = lines[i][3];
        if (s.y1 > s.y2) // 归一：y1 为上端点
        {
            std::swap(s.x1, s.x2);
            std::swap(s.y1, s.y2);
        }
        s.ty = s.y2;
        const float d2 = (s.x2 - s.x1) * (s.x2 - s.x1) +
                         (s.y2 - s.y1) * (s.y2 - s.y1);
        s.len = std::sqrt(d2);
        s.conf = i < (int)confs.size() ? confs[i] : 0.f;
        const float dx = s.x2 - s.x1, dy = s.y2 - s.y1;
        // 单位方向（uy >= 0，快速 rsqrt：2 除 → 2 乘）；太水平线段
        // 提前剔除（置空长度后不进入任何窗口聚类）
        if (s.len > 1e-3f)
        {
            const float inv = cluRsqrtF(d2);
            s.ux = dx * inv;
            s.uy = dy * inv;
            if (s.uy < cosMaxPsi)
                s.len = 0.f;
        }
        else
        {
            s.ux = 0.f;
            s.uy = 1.f;
        }
        s.tm = std::fabs(dy) > 1e-3f ? dx / dy
                                     : (dx >= 0 ? kMaxSlope : -kMaxSlope);
        s.tm = std::max(-kMaxSlope, std::min(kMaxSlope, s.tm));
    }

    // ---- 段按跨越窗口带计数排序入桶（替代逐窗全量扫描）----
    // 各带中心 y 与窗口循环内同一表达式预计算（排序键 x(yc) 在入桶时
    // 一并求出，窗口内免二次遍历）
    bOff.assign(nBands + 1, 0);
    ycs.resize(nBands);
    for (int w = 0; w < nBands; ++w)
    {
        const float ylo = H - (w + 1) * (float)k;
        ycs[w] = ylo + 0.5f * k;
    }
    int totItems = 0;
    for (int i = 0; i < n; ++i)
    {
        Seg &s = segsBuf[i];
        if (s.len <= 1e-3f)
        {
            s.wLo = s.wHi = -1;
            continue;
        }
        const auto bandLo = [&](float y) // 上端点：y1 < yhi（半开区间）
        {
            int b = (int)((float)(H - 1.f - y) / (float)k);
            return std::max(0, std::min(nBands - 1, b));
        };
        const auto bandHi = [&](float y) // 下端点：y2 >= ylo
        {
            int b = (int)((float)(H - y) / (float)k);
            return std::max(0, std::min(nBands - 1, b));
        };
        s.wHi = bandHi(s.ty);            // 下端点（低带号）
        s.wLo = bandLo(s.y1);            // 上端点（高带号）
        if (s.wLo > s.wHi)
            std::swap(s.wLo, s.wHi);
        totItems += s.wHi - s.wLo + 1;
        for (int w = s.wLo; w <= s.wHi; ++w)
            ++bOff[w + 1];
    }
    for (int w = 0; w < nBands; ++w)
        bOff[w + 1] += bOff[w];
    bItem.resize(totItems);
    bKeys.resize(totItems);
    {
        bFill.assign(bOff.begin(), bOff.end() - 1);
        for (int i = 0; i < n; ++i)
        {
            const Seg &s = segsBuf[i];
            if (s.wLo < 0)
                continue;
            for (int w = s.wLo; w <= s.wHi; ++w)
            {
                const int pos = bFill[w]++;
                bItem[pos] = i;
                bKeys[pos] = s.x1 + s.tm * (ycs[w] - s.y1);
            }
        }
    }

    // 单线段测量权重：长度 × 饱和置信度
    auto segW = [&](const Seg &s)
    {
        return s.len * (0.5f + std::min(std::max(s.conf, 0.f), 3.f) / 3.f);
    };

    // ---- 自底向上逐窗口：局部聚类成节点（收集进全局节点表）----
    for (int w = 0; w < nBands; ++w)
    {
        const float ylo = H - (w + 1) * (float)k;
        const float yhi = H - w * (float)k;
        const float yc = ylo + 0.5f * k;
        if (dbg)
        {
            dbg->y_lo.push_back(ylo);
            dbg->y_hi.push_back(yhi);
        }
        const int b0 = bOff[w], b1 = bOff[w + 1];
        if (b1 == b0)
            continue; // 无观测窗口，直接跳过

        // 1. 窗口内局部聚类：按 x(yc) 排序后多开放组贪心分配——新线段
        //    加入"方向点积 >= cosAng 且（x 间隙 <= match_x_tol 或 x 落在组的
        //    x 跨度内）"的最优开放组，无兼容组才新开。同窗交叉的两条线
        //    （x 交错、方向不同）因此各回各组，不会被单开组的顺序封口
        //    切成碎片（X 交叉/匝道汇入/急弯局部重叠场景）
        constexpr int kMaxOpen = 16;   // 同时开放组上限（须大于窗口内可能
                                       // 出现的方向簇数：超限后的兜底并入
                                       // 会污染组方向均值）
        constexpr int kMaxGrpMem = 24; // 单组单窗成员下标上限（超出仅不再
                                       // 记录下标——indices 为诊断输出；
                                       // 测量权重照常累计）
        int nOpen = 0;
        int oMem[kMaxOpen][kMaxGrpMem];
        int oCnt[kMaxOpen];
        float oSw[kMaxOpen], oSx[kMaxOpen], oSu[kMaxOpen], oSv[kMaxOpen],
            oCf[kMaxOpen]; // 全 float 累积
        float oXmin[kMaxOpen], oXmax[kMaxOpen];
        float oSeed[kMaxOpen]; // 组内已记录成员的最大 ty（增量维护）
        float oInv[kMaxOpen];  // 组方向和的归一化系数（随吸收增量重算，
                               // 免候选扫描内循环 sqrt）
        float oMx[kMaxOpen];   // 组加权均值 x（免候选扫描内循环除法）
        // 每吸收一个成员后刷新缓存（一次 sqrt/除法摊到一次成员添加）
        auto grpRefresh = [&](int o)
        {
            oMx[o] = oSx[o] / oSw[o];
            oInv[o] = cluRsqrtF(oSu[o] * oSu[o] + oSv[o] * oSv[o] +
                                1e-12f);
        };
        {
            // 桶内按 x(yc) 插入排序（元素极少，免 std::sort 开销）；
            // 排序键已在入桶时求出（bKeys），窗口内免二次遍历
            const int cnt = b1 - b0;
            for (int a = 1; a < cnt; ++a)
            {
                const int key = bItem[b0 + a];
                const float xa = bKeys[b0 + a];
                int b2 = a - 1;
                while (b2 >= 0 && bKeys[b0 + b2] > xa)
                {
                    bKeys[b0 + b2 + 1] = bKeys[b0 + b2];
                    bItem[b0 + b2 + 1] = bItem[b0 + b2];
                    --b2;
                }
                bKeys[b0 + b2 + 1] = xa;
                bItem[b0 + b2 + 1] = key;
            }
            for (int bi = b0; bi < b1; ++bi)
            {
                const int i = bItem[bi];
                const Seg &s = segsBuf[i];
                const float xi = bKeys[bi];
                const float wgt = segW(s);
                // 最优开放组：方向相容且 x 可归属（间隙达标或落在组的
                // x 跨度内——后者是交叉交错场景的回收通道），取 x 间隙
                // 最小者。谓词按"先便宜后贵"排序（x 间隙/跨度 → 方向
                // 点积 → 成对发散），纯过滤与最终归属无关
                int best = -1;
                float bestGap = 1e9f;
                const float lim = 2.f * p.match_x_tol;
                const float sxBot = s.x1 + s.tm * (s.y2 - s.y1);
                for (int o = 0; o < nOpen; ++o)
                {
                    const float gap = std::fabs(xi - oMx[o]);
                    if (gap > p.match_x_tol &&
                        (xi < oXmin[o] || xi > oXmax[o]))
                        continue; // x 间隙超限且不在组 x 跨度内
                    if (s.ux * oSu[o] * oInv[o] +
                            s.uy * oSv[o] * oInv[o] <
                        cosAng)
                        continue; // 与组均值方向相容
                    // 成对发散检验（核心）：与组内每一成员在两者 4 个
                    // 端点 y 处比较横向 x——任一处差值 > 2·match_x_tol 即为
                    // 交叉发散对，拒绝入组（保证组内不存在"某 y 处
                    // x 差值很大"的成员对；同线成员差值恒定不触发）。
                    // 四处差值取 max 后一次比较（无短路径分支，判定
                    // 与逐处比较等价）；只扫描已记录成员（oCnt 超限
                    // 部分无下标，不会读越界）
                    bool pairOK = true;
                    const int mo = std::min(oCnt[o], kMaxGrpMem);
                    for (int j2 = 0; j2 < mo; ++j2)
                    {
                        const Seg &m = segsBuf[oMem[o][j2]];
                        const float mxBot = m.x1 + m.tm * (m.y2 - m.y1);
                        const float d0 =
                            std::fabs(s.x1 -
                                      (m.x1 + m.tm * (s.y1 - m.y1)));
                        const float d1 =
                            std::fabs(sxBot -
                                      (m.x1 + m.tm * (s.y2 - m.y1)));
                        const float d2 =
                            std::fabs((s.x1 + s.tm * (m.y1 - s.y1)) -
                                      m.x1);
                        const float d3 =
                            std::fabs((s.x1 + s.tm * (m.y2 - s.y1)) -
                                      mxBot);
                        if (std::max(std::max(d0, d1),
                                     std::max(d2, d3)) > lim)
                        {
                            pairOK = false;
                            break;
                        }
                    }
                    if (!pairOK)
                        continue;
                    if (gap < bestGap)
                    {
                        bestGap = gap;
                        best = o;
                    }
                }
                if (best < 0 && nOpen < kMaxOpen)
                {
                    best = nOpen++;
                    oSw[best] = oSx[best] = oSu[best] = oSv[best] =
                        oCf[best] = 0.f;
                    oCnt[best] = 0;
                    oXmin[best] = 1e9f;
                    oXmax[best] = -1e9f;
                    oSeed[best] = -1e9f;
                }
                if (best < 0)
                {
                    // 开放组满：并入方向点积最高者（兜底，测量不丢）
                    float bestDot = -2.f;
                    for (int o = 0; o < nOpen; ++o)
                    {
                        const float dot = s.ux * oSu[o] * oInv[o] +
                                          s.uy * oSv[o] * oInv[o];
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            best = o;
                        }
                    }
                }
                oSw[best] += wgt;
                oSx[best] += wgt * xi;
                oSu[best] += wgt * s.ux;
                oSv[best] += wgt * s.uy;
                oCf[best] += s.conf;
                oXmin[best] = std::min(oXmin[best], xi);
                oXmax[best] = std::max(oXmax[best], xi);
                // seed_y 只累计会被记录的成员（与封口时扫描已记录成员
                // 的 max 语义逐位一致）
                if (oCnt[best] < kMaxGrpMem)
                {
                    oMem[best][oCnt[best]] = i;
                    oSeed[best] = std::max(oSeed[best], s.ty);
                }
                ++oCnt[best];
                grpRefresh(best);
            }
        }
        // 封口：开放组 → 全局节点（MSAC 提取的最小单元）。成员下标
        // 拷入扁平池（oMem 是窗口栈数组，出窗即失效）；dbg 的
        // centers/matched 与节点表同序同步追加（matched 提取阶段回填）
        for (int o = 0; o < nOpen; ++o)
        {
            if (oCnt[o] == 0)
                continue;
            const float inv = oInv[o];
            Node nd;
            nd.x = oMx[o];
            nd.y = yc;
            nd.ux = oSu[o] * inv;
            nd.uy = oSv[o] * inv;
            nd.w = oSw[o];
            nd.conf = oCf[o] / (float)oCnt[o];
            nd.seed_y = oSeed[o];
            nd.off = (int)nodePool.size();
            nd.cnt = std::min(oCnt[o], kMaxGrpMem);
            nodePool.insert(nodePool.end(), oMem[o], oMem[o] + nd.cnt);
            nodes.push_back(nd);
            if (dbg)
            {
                dbg->centers.push_back({nd.x, yc});
                dbg->matched.push_back(0);
            }
        }
    }

    // ---- 全局方向加权 MSAC：逐线提取候选 ----
    // 每轮在全部剩余节点上拟合一条最优线（内点判定 = 横向残差
    // <= x_tol 且方向与模型切向夹角达标的硬双约束；择优用加权
    // MSAC 分数 Σ w·(τ²−r²)，同内点数下残差小者胜），加权最小二乘
    // 精修后硬删除其内点，再在剩余节点上提取下一条。交叉线的节点
    // 对在方向预检/内点判据处即被拒绝，不以位置相近混入同线；断线
    // 跨度由 miss_max 切分约束
    //
    // 提取层热数据 SoA 化：采样与全表计数只触 x/y/ux/uy/w 五个
    // float 数组，顺序访问无双重间接；冷字段（conf/seed_y/成员区间）
    // 留在 nodes 供输出阶段。初始按 x 升序排布（局部化采样的前提：
    // 伴点 s1 在锚点 x 邻域内随机——同线节点 x 上聚集成带，邻域命中
    // 率远高于全局乱采；排序键附 y/w 保证确定性）。排序仅初始一次，
    // 删除后顺序自然演化
    int nRem = (int)nodes.size();
    packOrder.resize(nRem);
    for (int i = 0; i < nRem; ++i)
        packOrder[i] = i;
    std::sort(packOrder.begin(), packOrder.begin() + nRem,
              [&](int a, int b)
              {
                  if (nodes[a].x != nodes[b].x)
                      return nodes[a].x < nodes[b].x;
                  if (nodes[a].y != nodes[b].y)
                      return nodes[a].y < nodes[b].y;
                  return nodes[a].w > nodes[b].w;
              });
    eX.resize(nRem);
    eY.resize(nRem);
    eUx.resize(nRem);
    eUy.resize(nRem);
    eW.resize(nRem);
    eIdx.resize(nRem);
    float yMin = 1e9f, yMax = -1e9f; // 节点 y 范围（值域剪枝用）
    for (int i = 0; i < nRem; ++i)
    {
        const Node &nd = nodes[packOrder[i]];
        eX[i] = nd.x;
        eY[i] = nd.y;
        eUx[i] = nd.ux;
        eUy[i] = nd.uy;
        eW[i] = nd.w;
        eIdx[i] = packOrder[i];
        yMin = std::min(yMin, nd.y);
        yMax = std::max(yMax, nd.y);
    }
    // 帧级固定种子 LCG：同输入同输出（与 describeCurvePts 同款生成
    // 器）。每次取高 24 位拆两个 12 位随机数使用（s1/s2 各一），采样
    // 循环内 LCG 调用减半（依赖链是预检段的主要延迟之一）
    unsigned rngSeed = 0x9E3779B9u;
    auto rnd2 = [&]() -> unsigned
    {
        rngSeed = rngSeed * 1664525u + 1013904223u;
        const unsigned r = rngSeed >> 8;
        return r; // 高 24 位；调用方按 12 位拆分使用
    };
    // 方向门限的平方形式（sin² 的补）：dot 与 cosAng·√(1+m²) 均非负，
    // dot² >= cos²·(1+m²) 与点积门限数学等价，且免除法/rsqrt、纯
    // 乘加——向量化与标量共用同一谓词形态
    const float cos2 = cosAng * cosAng;
    const float tau2 = p.x_tol * p.x_tol; // MSAC 截断阈（提取门限平方）
#if CLU_AVX2
    // 帧级 SIMD 常量（循环不变量，一次 broadcast 全帧复用）
    const __m256 vTolC = _mm256_set1_ps(p.x_tol);
    const __m256 vCos2C = _mm256_set1_ps(cos2);
    const __m256 vSgnC = _mm256_set1_ps(-0.f);
    const __m256 vOneC = _mm256_set1_ps(1.f);
    const __m256 vZeroC = _mm256_setzero_ps();
#endif
    // 节点-模型内点谓词（SoA 槽位域，标量路径：尾部/自一致/精修集
    // 共用）。全比较谓词：异常模型（数值发散）自然落入拒绝路径
    auto sInl = [&](int s, float a2, float a1, float a0)
    {
        const float y = eY[s];
        const float pred = (a2 * y + a1) * y + a0;
        if (std::fabs(eX[s] - pred) > p.x_tol)
            return false; // 位置：横向残差超提取门限
        const float m = 2.f * a2 * y + a1;
        const float dot = eUx[s] * m + eUy[s];
        return dot >= 0.f && dot * dot >= cos2 * (1.f + m * m);
    };
    // 内点收集（向量化判定 + 位展开；dst 得到升序槽位表——与 SoA
    // 热数组同域，调用方按需转 nodes 下标）
    auto collectInl = [&](float a2, float a1, float a0,
                          std::vector<int> &dst)
    {
        dst.clear();
        int i = 0;
#if CLU_AVX2
        {
            const __m256 vA2 = _mm256_set1_ps(a2);
            const __m256 vA1 = _mm256_set1_ps(a1);
            const __m256 vA0 = _mm256_set1_ps(a0);
            const __m256 vK2 = _mm256_set1_ps(2.f * a2);
            const __m256 vTol = vTolC;
            const __m256 vCos2 = vCos2C;
            const __m256 vSgn = vSgnC;
            const __m256 vOne = vOneC;
            const __m256 vZero = vZeroC;
            for (; i + 8 <= nRem; i += 8)
            {
                const __m256 vy = _mm256_loadu_ps(eY.data() + i);
                const __m256 t = _mm256_fmadd_ps(vA2, vy, vA1);
                const __m256 pred = _mm256_fmadd_ps(t, vy, vA0);
                const __m256 res = _mm256_andnot_ps(
                    vSgn,
                    _mm256_sub_ps(_mm256_loadu_ps(eX.data() + i), pred));
                __m256 ok = _mm256_cmp_ps(res, vTol, _CMP_LE_OS);
                if (_mm256_movemask_ps(ok) == 0)
                    continue;
                const __m256 m = _mm256_fmadd_ps(vK2, vy, vA1);
                const __m256 dot = _mm256_fmadd_ps(
                    _mm256_loadu_ps(eUx.data() + i), m,
                    _mm256_loadu_ps(eUy.data() + i));
                ok = _mm256_and_ps(
                    ok, _mm256_cmp_ps(dot, vZero, _CMP_GE_OS));
                ok = _mm256_and_ps(
                    ok, _mm256_cmp_ps(_mm256_mul_ps(dot, dot),
                                      _mm256_mul_ps(
                                          vCos2,
                                          _mm256_fmadd_ps(m, m, vOne)),
                                      _CMP_GE_OS));
                int msk = _mm256_movemask_ps(ok);
                while (msk)
                {
                    const int b = __builtin_ctz((unsigned)msk);
                    dst.push_back(i + b);
                    msk &= msk - 1;
                }
            }
        }
#endif
        for (; i < nRem; ++i)
            if (sInl(i, a2, a1, a0))
                dst.push_back(i);
    };

    while (nRem >= kMinNodeInl)
    {
        // 1. MSAC 采样：锚点轮转（i0 = it%nRem，均匀覆盖每节点，
        //    免同锚重复浪费，确定性）+ 随机采伴点。样本对先过方向
        //    相容预检（不同线的节点几乎必然方向不相容，以 1 次点积
        //    的代价拒绝大量伪样本），解出模型后样本自一致（自身必须
        //    是模型内点）才值得全表计数
        float rA2 = 0.f, rA1 = 0.f, rA0 = 0.f;
        float bestA2 = 0.f, bestA1 = 0.f, bestA0 = 0.f;
        float bestScore = 0.f; // 最优模型 MSAC 分数（择优指标）
        int bestCnt = 0;
        // 采样轮数：基准值按节点规模反比放大（小线每轮全表计数便宜，
        // 轮数补偿找回小内点率线的采样置信；上限 2 倍封顶）
        int kNeed = kGlobalIters;
        if (nRem < kRefNodes)
            kNeed = std::min(kGlobalIters * kRefNodes / std::max(nRem, 1),
                             3 * kGlobalIters / 2);
        // 采样循环零除法：锚点轮转用条件减法、伴点用 Lemire 乘法
        // 取模（随机性近似均匀；运行时 % 除法 ~20+ 周期是轮均热点）
        int s0 = 0;
        const int s1Span = nRem / 4 + 1;
        int it = 0;
        for (; it < kNeed; ++it, ++s0)
        {
            if (s0 >= nRem)
                s0 -= nRem; // 锚点轮转（等价 it % nRem）
            const unsigned r12 = rnd2();
            // 伴点 s1：锚点 x 邻域（环形，半宽 nRem/8 槽）——同线节点
            // x 上聚集成带，邻域采样命中率远高于全局；s2 全局随机，
            // 保证 det 的 y 跨度（两个 12 位随机数拆自同一次 LCG）
            int s1 = s0 + 1 +
                     (int)(((r12 & 0xFFFu) * (unsigned)s1Span) >> 12) -
                     (int)(nRem / 8);
            if (s1 < 0)
                s1 += nRem;
            else if (s1 >= nRem)
                s1 -= nRem;
            // 方向相容预检（样本对，SoA 直接读）
            if (eUx[s0] * eUx[s1] + eUy[s0] * eUy[s1] < cosAng)
                continue;
            if (p.fit_quad)
            {
                // 过 3 点的抛物线（克拉默法则；同 y/共线退化为 det≈0）
                const int s2 =
                    (int)(((r12 >> 12) * (unsigned)nRem) >> 12);
                const float t0 = eY[s0], t1 = eY[s1], t2 = eY[s2];
                const float c0 = eX[s0], c1 = eX[s1], c2 = eX[s2];
                const float t00 = t0 * t0, t10 = t1 * t1, t20 = t2 * t2;
                const float det = t00 * (t1 - t2) - t0 * (t10 - t20) +
                                  (t10 * t2 - t1 * t20);
                if (std::fabs(det) < 1e-3f)
                    continue;
                rA2 = (c0 * (t1 - t2) - t0 * (c1 - c2) +
                       (c1 * t2 - t1 * c2)) / det;
                rA1 = (t00 * (c1 - c2) - c0 * (t10 - t20) +
                       (t10 * c2 - c1 * t20)) / det;
                rA0 = (t00 * (t1 * c2 - c1 * t2) -
                       t0 * (t10 * c2 - c1 * t20) +
                       c0 * (t10 * t2 - t1 * t20)) / det;
                // 自一致：3 样本自身必须都是模型内点（矛盾样本即弃）
                if (!sInl(s0, rA2, rA1, rA0) ||
                    !sInl(s1, rA2, rA1, rA0) ||
                    !sInl(s2, rA2, rA1, rA0))
                    continue;
            }
            else
            {
                // 过 2 点的直线
                const float dt = eY[s1] - eY[s0];
                if (std::fabs(dt) < 1e-6f)
                    continue;
                rA2 = 0.f;
                rA1 = (eX[s1] - eX[s0]) / dt;
                rA0 = eX[s0] - rA1 * eY[s0];
                if (!sInl(s0, rA2, rA1, rA0) ||
                    !sInl(s1, rA2, rA1, rA0))
                    continue;
            }
            // 2. 全表内点计数 + 加权 MSAC 分数（AVX2 8 宽，平方形式
            //    全乘加；计数 popcount、分数按掩码选通后水平累加）。
            //    MSAC：内点贡献 w·(τ² − r²)（横向残差越小收益越高），
            //    外点贡献 0——同内点数下残差小的模型胜出，精修起点
            //    更稳；方向约束保持硬门限（切向失配即出局）。
            //    值域剪枝：模型对全帧 y 范围的预测值域 [xlo,xhi]（抛
            //    物线含顶点），x 窗外节点横向残差必然超限——x 有序
            //    SoA 上二分出扫描区间，计数只触窗口（车道线只覆盖
            //    全图 x 的一小带，是计数成本的主削减项）
            float sw = 0.f;
            int cnt = 0;
            float xlo = 0.f, xhi = -1.f; // 窗口无效（NaN 防护回退全表）
            {
                const float p0 = (rA2 * yMin + rA1) * yMin + rA0;
                const float p1 = (rA2 * yMax + rA1) * yMax + rA0;
                xlo = std::min(p0, p1);
                xhi = std::max(p0, p1);
                if (rA2 != 0.f) // 顶点落在 y 范围内则并入值域（yv 计算
                                // 对 rA2=0 的除零由此规避）
                {
                    const float yv = -rA1 / (2.f * rA2);
                    if (yv > yMin && yv < yMax) // NaN 时恒 false，安全
                    {
                        const float pv = (rA2 * yv + rA1) * yv + rA0;
                        xlo = std::min(xlo, pv);
                        xhi = std::max(xhi, pv);
                    }
                }
            }
            // 值域剪枝（xlo/xhi）预留未启用：本数据规模（节点数十至百、
            // 窗口窄）下二分+尾部标量的开销反大于收益；大节点规模场景
            // 可在 eX 有序前提下恢复 lower_bound 窗口
            (void)xlo;
            (void)xhi;
            const int ilo = 0, ihi = nRem;

#if CLU_AVX2
            {
                const __m256 vA2 = _mm256_set1_ps(rA2);
                const __m256 vA1 = _mm256_set1_ps(rA1);
                const __m256 vA0 = _mm256_set1_ps(rA0);
                const __m256 vK2 = _mm256_set1_ps(2.f * rA2);
                const __m256 vTau2 = _mm256_set1_ps(tau2);
                const __m256 vTol = vTolC;
                const __m256 vCos2 = vCos2C;
                const __m256 vSgn = vSgnC;
                const __m256 vOne = vOneC;
                const __m256 vZero = vZeroC;
                int i = ilo;
                for (; i + 8 <= ihi; i += 8)
                {
                    const __m256 vy = _mm256_loadu_ps(eY.data() + i);
                    const __m256 t = _mm256_fmadd_ps(vA2, vy, vA1);
                    const __m256 pred = _mm256_fmadd_ps(t, vy, vA0);
                    const __m256 res = _mm256_andnot_ps(
                        vSgn, _mm256_sub_ps(_mm256_loadu_ps(eX.data() + i),
                                            pred));
                    // 位置先行：整块位置全拒则跳过方向计算（位置通过
                    // 率仅三四成，分支收益为正）
                    __m256 ok = _mm256_cmp_ps(res, vTol, _CMP_LE_OS);
                    if (_mm256_movemask_ps(ok) == 0)
                        continue;
                    const __m256 m = _mm256_fmadd_ps(vK2, vy, vA1);
                    const __m256 dot = _mm256_fmadd_ps(
                        _mm256_loadu_ps(eUx.data() + i), m,
                        _mm256_loadu_ps(eUy.data() + i));
                    ok = _mm256_and_ps(
                        ok, _mm256_cmp_ps(dot, vZero, _CMP_GE_OS));
                    ok = _mm256_and_ps(
                        ok, _mm256_cmp_ps(_mm256_mul_ps(dot, dot),
                                          _mm256_mul_ps(
                                              vCos2,
                                              _mm256_fmadd_ps(m, m, vOne)),
                                          _CMP_GE_OS));
                    const int msk = _mm256_movemask_ps(ok);
                    if (msk == 0)
                        continue; // 整块方向/位置全拒：免权重累加
                    cnt += __builtin_popcount((unsigned)msk);
                    // 内点增益 w·(τ² − r²)，外点被掩码清零（r² 截断
                    // 到 τ² 保证增益非负，掩掉也不出负数）
                    const __m256 gain = _mm256_sub_ps(
                        vTau2, _mm256_min_ps(_mm256_mul_ps(res, res),
                                             vTau2));
                    sw += hsum256(_mm256_and_ps(
                        _mm256_mul_ps(_mm256_loadu_ps(eW.data() + i),
                                      gain),
                        ok));
                }
                for (; i < ihi; ++i)
                {
                    const float y = eY[i];
                    const float pred = (rA2 * y + rA1) * y + rA0;
                    const float res = std::fabs(eX[i] - pred);
                    if (res > p.x_tol)
                        continue;
                    const float m = 2.f * rA2 * y + rA1;
                    const float dot = eUx[i] * m + eUy[i];
                    if (dot < 0.f || dot * dot < cos2 * (1.f + m * m))
                        continue;
                    ++cnt;
                    sw += eW[i] * (tau2 - res * res);
                }
            }
#else
            for (int i = ilo; i < ihi; ++i)
            {
                const float y = eY[i];
                const float pred = (rA2 * y + rA1) * y + rA0;
                const float res = std::fabs(eX[i] - pred);
                if (res > p.x_tol)
                    continue;
                const float m = 2.f * rA2 * y + rA1;
                const float dot = eUx[i] * m + eUy[i];
                if (dot < 0.f || dot * dot < cos2 * (1.f + m * m))
                    continue;
                ++cnt;
                sw += eW[i] * (tau2 - res * res);
            }
#endif
            if (sw > bestScore || (sw == bestScore && cnt > bestCnt))
            {
                bestScore = sw;
                bestCnt = cnt;
                bestA2 = rA2;
                bestA1 = rA1;
                bestA0 = rA0;
                // 自适应置信停止：内点率 ε 下 3 点全内点的单采样命中
                // 率 ≈ ε³，ln(0.01)/ln(1−ε³) 轮即达 99% 置信（上限收
                // 缩到该值）；仅在 best 更新时算一次 log 摊多轮
                const float eps = (float)bestCnt / (float)nRem;
                const float p3 = eps * eps * eps;
                if (p3 > 1e-9f && p3 < 0.999f)
                {
                    const int kneed =
                        (int)std::ceil(std::log(0.01f) /
                                       std::log(1.f - p3));
                    if (kneed < kNeed)
                        kNeed = kneed;
                }
            }
            if (cnt == nRem)
                break; // 全节点内点：不可能更优，提前退出
        }
        if (bestCnt < kMinNodeInl)
            break; // 剩余节点撑不起一条可靠线：提取结束（防止 3 节点
                   // 随机共线产生的垃圾候选输出）
        // 3. MSAC 最优模型的内点集 → 加权最小二乘精修（解奇异则
        //    保持 MSAC 模型）。inlBuf 存 SoA 槽位（与热数组同域）
        collectInl(bestA2, bestA1, bestA0, inlBuf);
        float mA2 = bestA2, mA1 = bestA1, mA0 = bestA0;
        wlsFit(eX.data(), eY.data(), eW.data(), inlBuf.data(),
               (int)inlBuf.size(), p.fit_quad, mA2, mA1, mA0);
        // 4. 精修模型全表重收内点（模型更准，允许集合较 MSAC 轮
        //    扩张；单轮精修不迭代——下游链拟合还有最小二乘）
        collectInl(mA2, mA1, mA0, inlBuf);
        // 4.5 最大容许断线距离：内点按 y 升序扫描相邻纵向间隙，超
        //     过 miss_max×窗口高 处切断；仅节点数最多的连续子段作为
        //     本线输出（子段重新加权精修后登记），其余子段节点放回
        //     剩余池参与后续迭代——足够长则独立成线，太短因不足最小
        //     内点门限自然丢弃。防止把跨大间隙（长虚线空档/遮挡）的
        //     虚假长链拟合为一条曲线。复用 miss_max 参数，无新增参
        //     数面；切分为每线一次小数组排序+线性扫描，性能可忽略
        {
            // y 桶计数排序：桶宽 = 窗口高 k（同桶内间隙必然 ≤ k，不
            // 会切断，桶内无需排序）——替代比较排序，O(nInl + 桶数)
            const int nInl = (int)inlBuf.size();
            const float invK = 1.f / (float)k;
            int binCnt[48] = {0};
            const int nBin = nBands + 1 < 48 ? nBands + 1 : 47;
            for (int q = 0; q < nInl; ++q)
            {
                int b = (int)(eY[inlBuf[q]] * invK);
                b = b < 0 ? 0 : (b >= nBin ? nBin - 1 : b);
                ++binCnt[b];
            }
            int acc = 0;
            for (int b = 0; b < nBin; ++b)
            {
                const int c = binCnt[b];
                binCnt[b] = acc;
                acc += c;
            }
            inlTmp.assign(inlBuf.begin(), inlBuf.end());
            for (int q = 0; q < nInl; ++q)
            {
                int b = (int)(eY[inlTmp[q]] * invK);
                b = b < 0 ? 0 : (b >= nBin ? nBin - 1 : b);
                inlBuf[binCnt[b]++] = inlTmp[q];
            }
            const float maxGap = (float)p.miss_max * (float)k;
            int segStart = 0, bestStart = 0, bestCnt = 0;
            float bestW = -1.f;
            for (int i = 1; i <= nInl; ++i)
            {
                if (i < nInl &&
                    eY[inlBuf[i]] - eY[inlBuf[i - 1]] <= maxGap)
                    continue;
                // 连续子段 [segStart, i) 收口：节点数最多者胜，平局
                // 比权重和（确定性）
                const int cnt = i - segStart;
                float sw = 0.f;
                for (int q = segStart; q < i; ++q)
                    sw += eW[inlBuf[q]];
                if (cnt > bestCnt || (cnt == bestCnt && sw > bestW))
                {
                    bestCnt = cnt;
                    bestStart = segStart;
                    bestW = sw;
                }
                segStart = i;
            }
            if (bestCnt < kMinNodeInl)
                break; // 最大连续子段仍不足最小门限：剩余内点为零散
                       // 垃圾聚簇，整线作废并终止提取
            for (int q = 0; q < bestCnt; ++q) // 原地收拢为选中子段
                inlBuf[q] = inlBuf[bestStart + q];
            inlBuf.resize(bestCnt);
            // 子段重新加权精修（内点集即子段本身，不再重收——子段
            // 内纵向无大间隙，LSQ 后残差只会更小）
            wlsFit(eX.data(), eY.data(), eW.data(), inlBuf.data(),
                   bestCnt, p.fit_quad, mA2, mA1, mA0);
        }
        const int clOff = (int)candInl.size();
        remark.assign(nRem, 0);
        for (int s : inlBuf)
        {
            remark[s] = 1;
            candInl.push_back(eIdx[s]);
        }
        const int inlCnt = (int)candInl.size() - clOff;
        if (inlCnt < kMinNodeInl)
            continue; // 防御：精修回退致内点流失——该线作废，回收
                      // candInl 空间（截断登记长度）
        CandLine cl;
        cl.a2 = mA2;
        cl.a1 = mA1;
        cl.a0 = mA0;
        cl.off = clOff;
        cl.cnt = inlCnt;
        candLines.push_back(cl);
        // dbg 回填：内点节点标记为已归属（与节点表同序）
        if (dbg)
            for (int q = clOff; q < (int)candInl.size(); ++q)
                dbg->matched[candInl[q]] = 1;
        // 5. 硬删除内点（一节点只归一线）：SoA 保序紧缩——eX 的 x
        // 有序性必须保持（值域剪枝的二分窗口依赖它）
        int keep = 0;
        for (int i = 0; i < nRem; ++i)
        {
            if (remark[i])
                continue;
            eX[keep] = eX[i];
            eY[keep] = eY[i];
            eUx[keep] = eUx[i];
            eUy[keep] = eUy[i];
            eW[keep] = eW[i];
            eIdx[keep] = eIdx[i];
            ++keep;
        }
        nRem = keep;
    }

    // ---- 输出组装：候选线内点节点序列即车道线中心线（直写调用方
    // clusters；原位覆盖、不清空、保留元素容量，稳态零堆分配）----
    const size_t hi = clusters.size();
    size_t j = 0;
    for (const CandLine &cl : candLines)
    {
        FmlSegCluster *dst;
        if (j < hi)
        {
            dst = &clusters[j]; // 复用调用方元素容量
        }
        else
        {
            clusters.emplace_back();
            dst = &clusters.back();
        }
        buildClu(*dst, cl, nodes, candInl.data() + cl.off, nodePool, p, k,
                 H, idStamp, n);
        ++j;
    }
    clusters.resize(j);
    std::sort(clusters.begin(), clusters.end(),
              [](const FmlSegCluster &a, const FmlSegCluster &b)
              {
                  return a.score > b.score;
              });
    // 数据训练打分器（lane_scorer 三档）：用外部/内置权重重打分重排
    if (p.lane_scorer >= 1 && p.lane_scorer <= 3)
    {
        fml_lane_scorer_weights sw;
        const fml_lane_scorer_weights *swp = p.lane_scorer_w;
        if (swp == nullptr)
        {
            fml_lane_scorer_init(sw, p.lane_scorer);
            swp = &sw;
        }
        applyLaneScorer(clusters, swp, (float)width, (float)H,
                        (float)std::max(1, p.bands));
    }
    int nvalid = 0;
    for (const FmlSegCluster &c : clusters)
        nvalid += c.valid ? 1 : 0;
    return nvalid;
}

// ---- 聚类结果绘制（与 fml_draw_lines 同风格：只画不弹窗）----

// 无效/噪声簇固定灰色
static const cv::Scalar kClusterNoiseDraw = {110, 110, 110};

// 有效簇固定纯绿
static const cv::Scalar kClusterValidDraw = {0, 255, 0};

// R 标签（排名 + 得分）固定蓝色
static const cv::Scalar kClusterTagDraw = {255, 0, 0};

void fml_draw_clusters(cv::Mat &img,
                       const std::vector<FmlSegCluster> &clusters,
                       double scale_x, double scale_y, int thickness,
                       int mode, const fml_cluster_debug *dbg)
{
    const int thValid = std::max(1, thickness);
    const int thNoise = std::max(1, thickness / 2);

    // ---- SPLIT 模式：窗口划分细线 + 小簇中心点（基底由调用方画）----
    if (mode == FML_CLUSTER_DRAW_SPLIT)
    {
        if (dbg == nullptr)
            return;
        for (size_t i = 0; i < dbg->y_lo.size(); ++i)
            cv::line(img,
                     {0, cvRound(dbg->y_lo[i] * scale_y)},
                     {img.cols, cvRound(dbg->y_lo[i] * scale_y)},
                     {170, 170, 170}, 1, cv::LINE_AA);
        if (!dbg->y_hi.empty())
            cv::line(img,
                     {0, cvRound(dbg->y_hi.back() * scale_y)},
                     {img.cols, cvRound(dbg->y_hi.back() * scale_y)},
                     {170, 170, 170}, 1, cv::LINE_AA);
        for (size_t i = 0; i < dbg->centers.size(); ++i)
            cv::circle(img,
                       cv::Point(cvRound(dbg->centers[i].x * scale_x),
                                 cvRound(dbg->centers[i].y * scale_y)),
                       3,
                       dbg->matched[i] ? cv::Scalar(0, 255, 255)
                                       : cv::Scalar(0, 0, 255),
                       -1, cv::LINE_AA);
        return;
    }

    // ---- CHAINS / FIT 模式：逐簇绘制 ----
    int rank = -1;
    for (const FmlSegCluster &c : clusters)
    {
        ++rank;
        const bool ok = c.valid;
        const cv::Scalar col = ok ? kClusterValidDraw : kClusterNoiseDraw;
        const int th = ok ? thValid : thNoise;
        if (c.centers.empty())
            continue;

        if (mode == FML_CLUSTER_DRAW_FIT)
        {
            // 拟合曲线（粗线）+ 中心点（同色小点）
            // 旋转主轴有效时按局部 (v,u) 参数化圆弧绘制；否则
            // 沿用竖直主轴抛物线 a2/a1/a0 + t_min/t_max
            const bool rot =
                (c.ra != 0.f || c.rb != 0.f || c.rc != 0.f);
            cv::Point prev;
            bool first = true;
            if (rot)
            {
                const float th = c.theta * (float)CV_PI / 180.f;
                const float cs = std::cos(th), sn = std::sin(th);
                // 从旋转后的中心点计算 v 范围：仅限拟合支撑区间
                //（y∈[t_min,t_max]，竖直主轴帧 MSAC 内点集）内的节点。
                // ra/rb/rc 只在内点上有效，病态抛物线（大 |ra|）外插到
                // 支撑外节点处会二次发散；支撑外节点不参与范围计算
                float vlo = 1e9f, vhi = -1e9f;
                for (const cv::Point2f &p : c.centers)
                {
                    if (p.y < c.t_min || p.y > c.t_max)
                        continue;
                    const float v = -p.x * sn + p.y * cs;
                    vlo = std::min(vlo, v);
                    vhi = std::max(vhi, v);
                }
                if (vlo > vhi)
                { // 防御：支撑内无节点（理论不可达，t 范围即取自内点）
                    vlo = 1e9f;
                    vhi = -1e9f;
                    for (const cv::Point2f &p : c.centers)
                    {
                        const float v = -p.x * sn + p.y * cs;
                        vlo = std::min(vlo, v);
                        vhi = std::max(vhi, v);
                    }
                }
                for (float vq = vlo; vq <= vhi; vq += 2.f)
                {
                    const float u =
                        c.ra * vq * vq + c.rb * vq + c.rc;
                    const float x = u * cs - vq * sn;
                    const float y = u * sn + vq * cs;
                    const cv::Point q(cvRound(x * scale_x),
                                      cvRound(y * scale_y));
                    if (!first)
                        cv::line(img, prev, q, col, thickness,
                                 cv::LINE_AA);
                    prev = q;
                    first = false;
                }
            }
            else
            {
                for (float t2 = c.t_min; t2 <= c.t_max; t2 += 2.f)
                {
                    const float v =
                        c.a2 * t2 * t2 + c.a1 * t2 + c.a0;
                    const cv::Point q(cvRound(v * scale_x),
                                      cvRound(t2 * scale_y));
                    if (!first)
                        cv::line(img, prev, q, col, thickness,
                                 cv::LINE_AA);
                    prev = q;
                    first = false;
                }
            }
            for (const cv::Point2f &pt : c.centers)
                cv::circle(img,
                           cv::Point(cvRound(pt.x * scale_x),
                                     cvRound(pt.y * scale_y)),
                           2, col, -1, cv::LINE_AA);
        }
        else
        {
            // 中心点链折线（单点画圆）
            if (c.centers.size() == 1)
            {
                cv::circle(img,
                           cv::Point(cvRound(c.centers[0].x * scale_x),
                                     cvRound(c.centers[0].y * scale_y)),
                           std::max(2, th), col, -1, cv::LINE_AA);
            }
            else
            {
                cv::Point prev;
                bool first = true;
                for (const cv::Point2f &pt : c.centers)
                {
                    const cv::Point q(cvRound(pt.x * scale_x),
                                      cvRound(pt.y * scale_y));
                    if (!first)
                        cv::line(img, prev, q, col, th, cv::LINE_AA);
                    prev = q;
                    first = false;
                }
            }
        }

        // R 标签（排名 + 得分，蓝色），画在曲线参数范围起点处
        if (ok)
        {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "R%d %.2f", rank,
                          (double)c.score);
            float lx, ly;
            const bool rot =
                (c.ra != 0.f || c.rb != 0.f || c.rc != 0.f);
            if (rot)
            {
                const float th = c.theta * (float)CV_PI / 180.f;
                const float cs = std::cos(th), sn = std::sin(th);
                // 标签定位与弧绘制同口径：v 下限仅限拟合支撑区间内节点
                float vlo = 1e9f;
                for (const cv::Point2f &p : c.centers)
                {
                    if (p.y < c.t_min || p.y > c.t_max)
                        continue;
                    const float v = -p.x * sn + p.y * cs;
                    vlo = std::min(vlo, v);
                }
                if (vlo > 1e8f)
                { // 防御：同弧绘制回退全集合
                    vlo = 1e9f;
                    for (const cv::Point2f &p : c.centers)
                        vlo = std::min(vlo, -p.x * sn + p.y * cs);
                }
                const float u =
                    c.ra * vlo * vlo + c.rb * vlo + c.rc;
                lx = u * cs - vlo * sn;
                ly = u * sn + vlo * cs;
            }
            else
            {
                const float t0 = c.t_min;
                lx = c.a2 * t0 * t0 + c.a1 * t0 + c.a0;
                ly = t0;
            }
            cv::putText(img, tag,
                        cv::Point(cvRound(lx * scale_x) + 6,
                                  cvRound(ly * scale_y) + thickness),
                        cv::FONT_HERSHEY_SIMPLEX, 1.1, kClusterTagDraw,
                        2, cv::LINE_AA);
        }
    }
}