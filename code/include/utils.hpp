// 项目名: fml_for_x86_64
// 文件名: utils.hpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// 内部基础工具库（header-only）：跨模块复用的底层组件，全部以 inline /
// 模板形式实现在头文件内，无对应 .cpp 编译单元，供 detector/fld/elsed/
// cluster 各编译单元零成本复用。不对外公开，用户只需包含 fml.hpp。
//
// 内容：
//   FML_AVX2 编译开关与 fmlMallocTune / 线程池 FmlPool（poolRows/poolPost/
//   poolWait）/ NFA 显著性 lgamma、log 查表（lgT/logT）/ 快速数学
//   （rsqrtF、log10F、hsum256 SIMD 水平求和）/ 直线几何小工具
//   （cross3/normalizeLine/distPointLine/lineFromDir）

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define FML_AVX2 1
#endif

namespace fml
{
#if defined(__GLIBC__)
#include <malloc.h>
// threads=2 半区宽度逐帧随内容变化，cv::Mat 平面随之重分配。
// 默认 glibc 对 >128KB 的分配走 mmap（新页清零 + munmap），每帧数次
// 页错误代价显著。提高 mmap 阈值让平面分配复用堆块（进程级一次性设置）
static inline void fmlMallocTune()
{
    static int done = 0;
    if (!done)
    {
        done = 1;
        mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024);
        mallopt(M_TRIM_THRESHOLD, 32 * 1024 * 1024);
    }
}
#else
static inline void fmlMallocTune()
{
}
#endif

// 双线程持久工作池（threads=2 时启用）
// 目标场景：单物理核的两个逻辑 CPU（超线程兄弟）。池由首次使用方懒创建
// 一个工作线程，进程生命周期内常驻（免每帧线程创建开销）。
// 约定（对外文档已声明）：fml_* API 的双线程模式仅允许单应用线程调用。
//
// 握手协议（免锁，自旋 + pause）：
//   主线程：seq++ → 写 job → posted.release(seq) → 跑自己那半 →
//           自旋 done.acquire() == seq
//   工作线程：自旋 posted.acquire() != seen → seen = posted → 跑 job →
//             done.release(seen)
// 序号单调递增，无 ABA；job 的写/读被 posted/done 的 acquire-release 定序。
struct FmlPool
{
    std::thread th;
    std::function<void(int, int)> job; // 小捕获（this+整数），SBO 栈内无堆分配
    std::atomic<uint32_t> posted{0};
    std::atomic<uint32_t> done{0};
    std::atomic<bool> quit{false};
    uint32_t seq = 0;  // 仅主线程读写
    uint32_t seen = 0; // 仅工作线程读写

    static inline void spinPause()
    {
#if defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    void run()
    {
        for (;;)
        {
            while (posted.load(std::memory_order_acquire) == seen)
            {
                if (quit.load(std::memory_order_relaxed))
                    return;
                spinPause();
            }
            seen = posted.load(std::memory_order_relaxed);
            int b = jobBegin, e = jobEnd;
            job(b, e);
            done.store(seen, std::memory_order_release);
        }
    }

    int jobBegin = 0, jobEnd = 0; // 与 job 配套的区间参数

    // 单例：故意泄漏（new 后不 delete），规避静态析构顺序问题；
    // 工作线程 detach，随进程退出
    static FmlPool *instance()
    {
        static FmlPool *p = nullptr;
        if (!p)
        {
            p = new FmlPool();
            std::thread t([](FmlPool *q)
                          {
                              q->run();
                          },
                          p);
            t.detach();
            p->th = std::move(t);
        }
        return p;
    }

    // 提交工作线程任务并等待完成（主线程自己的那半由调用方执行）
    void post(int b, int e, std::function<void(int, int)> f)
    {
        job = std::move(f);
        jobBegin = b;
        jobEnd = e;
        ++seq;
        posted.store(seq, std::memory_order_release);
    }

    void waitDone()
    {
        while (done.load(std::memory_order_acquire) != seq)
            spinPause();
    }
};

// 行区间二分并行：主线程 [0,mid)，工作线程 [mid,rows)，阻塞至全部完成。
// rows 太小（并行收益低于握手开销）时退化为串行整段
template <class F>
inline void poolRows(int rows, F &&f)
{
    if (rows < 64)
    {
        f(0, rows);
        return;
    }
    FmlPool *p = FmlPool::instance();
    const int mid = rows / 2;
    p->post(mid, rows, [&f](int b, int e)
            {
                f(b, e);
            });
    f(0, mid);
    p->waitDone();
}

// 无区间任务提交（洪泛等）：工作线程执行 f()，主线程随后可做自己那份，
// 完成后调用 poolWait() 阻塞等待
template <class F>
inline void poolPost(F &&f)
{
    FmlPool *p = FmlPool::instance();
    p->post(0, 0, [f](int, int) mutable
            {
                f();
            });
}
inline void poolWait()
{
    FmlPool::instance()->waitDone();
}
// float 域快速倒数平方根：rsqrtss + 1 次 Newton 迭代（相对误差 ~1e-7）。
// x=0 时结果为 +inf。非 x86 回退 1/sqrt
inline float rsqrtF(float x)
{
#if defined(__SSE2__)
    const __m128 v = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(v);
    const __m128 half = _mm_set_ss(0.5f);
    const __m128 three = _mm_set_ss(1.5f);
    const __m128 vr2 = _mm_mul_ss(_mm_mul_ss(v, r), r);
    r = _mm_mul_ss(r, _mm_sub_ss(three, _mm_mul_ss(half, vr2)));
    return _mm_cvtss_f32(r);
#else
    return 1.0f / std::sqrt(x);
#endif
}

// float 域快速 log10：指数位提取 + 64 段线性插值查表（最大误差 ~1e-4，
// 仅用于评分的置信度压缩项，不影响几何判据）。与 lgT/logT 查表风格一致。
// x<=0 回退 0（调用侧 max(conf,0) 保证定义域）
inline float log10F(float x)
{
    if (x <= 1e-20f)
        return 0.f;
    static const float LOG2_TAB[65] = {
        0.00000000f, 0.02236781f, 0.04439412f, 0.06608919f, 0.08746284f,
        0.10852446f, 0.12928302f, 0.14974712f, 0.16992500f, 0.18982456f,
        0.20945337f, 0.22881869f, 0.24792751f, 0.26678654f, 0.28540222f,
        0.30378075f, 0.32192809f, 0.33985000f, 0.35755200f, 0.37503943f,
        0.39231742f, 0.40939094f, 0.42626475f, 0.44294350f, 0.45943162f,
        0.47573343f, 0.49185310f, 0.50779464f, 0.52356196f, 0.53915881f,
        0.55458885f, 0.56985561f, 0.58496250f, 0.59991284f, 0.61470984f,
        0.62935662f, 0.64385619f, 0.65821148f, 0.67242534f, 0.68650053f,
        0.70043972f, 0.71424552f, 0.72792045f, 0.74146699f, 0.75488750f,
        0.76818432f, 0.78135971f, 0.79441587f, 0.80735492f, 0.82017896f,
        0.83289001f, 0.84549005f, 0.85798100f, 0.87036472f, 0.88264305f,
        0.89481776f, 0.90689060f, 0.91886324f, 0.93073734f, 0.94251451f,
        0.95419631f, 0.96578428f, 0.97727992f, 0.98868469f, 1.00000000f};
    int ei;
    const float m = std::frexp(x, &ei); // x = m·2^ei，m ∈ [0.5,1)
    const float u = (2.f * m - 1.f) * 64.f; // [0,64) 查表位置
    int k = (int)u;
    if (k > 63)
        k = 63;
    if (k < 0)
        k = 0;
    const float fr = u - (float)k;
    // log2(m) = -1 + log2(1+t)，t = 2m-1 ∈ [0,1)
    const float l2 = LOG2_TAB[k] + fr * (LOG2_TAB[k + 1] - LOG2_TAB[k]) -
                     1.f;
    return ((float)ei + l2) * 0.30102999566f;
}

#if FML_AVX2
// __m256i 8×i32 水平求和（精确）
inline long long hsum256_epi32(__m256i v)
{
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(v),
                               _mm256_extracti128_si256(v, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (long long)_mm_cvtsi128_si32(s4) + (long long)_mm_extract_epi32(s4, 1);
}

// __m256 水平求和（两级 shuffle 归并，浮点近似序）
inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}
#endif

// 3 维向量叉积 c = a × b
inline void cross3(const double a[3], const double b[3], double c[3])
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

// 直线归一化（构造后调用一次，此后点线距离为纯点积）
inline void normalizeLine(double l[3])
{
    double w = std::sqrt(l[0] * l[0] + l[1] * l[1]);
    double inv = 1.0 / w;
    l[0] *= inv;
    l[1] *= inv;
    l[2] *= inv;
}

// 点到已归一化直线的带符号距离
inline double distPointLine(const double p[3], const double l[3])
{
    return l[0] * p[0] + l[1] * p[1] + l[2] * p[2];
}

// 过点 (x0,y0) 方向 (vx,vy) 的直线一般式（齐次坐标，构造即归一化）。
// (vx,vy) 已单位化，(l0,l1)=(−vy,vx) 即单位法向，直接写出系数
inline void lineFromDir(float x0, float y0, float vx, float vy, double l[3])
{
    l[0] = (double)-vy;
    l[1] = (double)vx;
    l[2] = (double)vy * x0 - (double)vx * y0;
}
// NFA 查表：lgamma(k+1) 与 log(k)，k=0..LNFA_TAB；超出回退直接计算
constexpr int LNFA_TAB = 16384;
inline std::vector<double> &lgammaTable()
{
    static thread_local std::vector<double> tab;
    if (tab.empty())
    {
        tab.resize(LNFA_TAB + 1);
        for (int k = 0; k <= LNFA_TAB; ++k)
            tab[k] = std::lgamma((double)k + 1.0);
    }
    return tab;
}
inline std::vector<double> &logTable()
{
    static thread_local std::vector<double> tab;
    if (tab.empty())
    {
        tab.resize(LNFA_TAB + 1);
        tab[0] = 0.0;
        for (int k = 1; k <= LNFA_TAB; ++k)
            tab[k] = std::log((double)k);
    }
    return tab;
}
inline double lgT(int k)
{
    return k <= LNFA_TAB ? lgammaTable()[k] : std::lgamma((double)k + 1.0);
}
inline double logT(int k)
{
    return k <= LNFA_TAB ? logTable()[k] : std::log((double)k);
}
} // namespace fml
