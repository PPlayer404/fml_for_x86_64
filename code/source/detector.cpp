// 项目名: fml_for_x86_64
// 文件名: detector.cpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// 后端公共仓库：FmlDetector 构造、顶层调度入口（detect / 双线程空间拆分）、
// 两端共享的后处理流水线（NFA 显著性与重算、共线合并、方向统一、
// 长度/贴边过滤、先验平移），以及 fml.hpp 公开接口的实现
// （fml_detect / fml_detect_elsed / fml_draw_lines）。
//
// 两前端实现见 fld.cpp（传统 Canny/FLT）与 elsed.cpp（ELSED），
// 类定义与共享结构见 detector.hpp。
//
// 参考文献
//   [1] Lee et al., Outdoor place recognition ... using straight lines, ICRA 2014.
//   [2] von Gioi et al., LSD: A Line Segment Detector, IPL 2012.
//   [3] Suárez et al., ELSED: Enhanced Line SEgment Drawing, Pattern Recognition 2022.
//   [4] Akinlar & Topal, EDLines, Pattern Recognition Letters 2011.

#include "detector.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <opencv2/imgproc.hpp>

using fml::FmlDetector;   // 公开接口实现中直接指代内部检测器类
using fml::ElsedCfg;      // 同上（构造传参用）
using fml::fmlMallocTune; // 同上（进程级 malloc 调优）
namespace fml
{
FmlDetector::FmlDetector(int _length_threshold, float _distance_threshold,
                         double _canny_th1, double _canny_th2,
                         float _grad_sigma, bool _do_merge,
                         float _nfa_eps, float _nfa_ang_tol_deg,
                         int _shift_px, bool _elsed, const ElsedCfg &_ec,
                         int _threads)
    : threshold_length(_length_threshold),
      threshold_dist(_distance_threshold),
      canny_th1(_canny_th1),
      canny_th2(_canny_th2),
      grad_sigma(_grad_sigma),
      do_merge(_do_merge),
      nfa_eps(_nfa_eps),
      nfa_ang_tol(_nfa_ang_tol_deg * (float)CV_PI / 180.0f),
      tan_tol_(std::tan(nfa_ang_tol)),
      shift_px_(_shift_px),
      log10_nfa_eps_(std::log10((double)_nfa_eps)),
      nfa_ang_tol_deg_(_nfa_ang_tol_deg),
      elsed_(_elsed),
      ec_(_ec),
      nthr_(_threads == 2 ? 2 : 1)
{
}

void FmlDetector::detect(const cv::Mat &_image, std::vector<cv::Vec4f> &lines,
                         std::vector<float> *confidences)
{
    CV_Assert(!_image.empty() && _image.type() == CV_8UC1);

    lines.clear();
    if (confidences)
        confidences->clear();
    std::vector<Segment> segments;
    // threads=2：图像空间拆分（左右两半并行独立检测 + 接缝缝合）；
    // threads=1：整图单线程
    if (nthr_ == 2 && _image.cols >= 128)
        detectSegmentsDual(_image, segments);
    else
        detectSegments(_image, segments);
    if (confidences)
        confidences->reserve(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        const Segment &seg = segments[i];
        lines.push_back(cv::Vec4f(seg.x1, seg.y1, seg.x2, seg.y2));
        if (confidences)
            confidences->push_back(seg.conf);
    }
}

// 单图完整管线：梯度 → canny/ELSED 绘制 → 链提取 → 合并
void FmlDetector::detectSegments(const cv::Mat &_image,
                                 std::vector<Segment> &segments)
{
    segments.reserve(256);
    if (elsed_)
    {
        // ELSED 管线：梯度图锚点 + 边缘绘制游走，不经过 canny
        detectElsed(_image, segments);
    }
    else
    {
        computeGradientAngle(_image); // Sobel 一遍，canny 与梯度角共用
        cannyFromGradient((int)canny_th1, (int)canny_th2);
        cv::Mat canny(nms_map_, cv::Rect(1, 1, _image.cols, _image.rows));
        // 四角区域清零，避免图像角落产生虚假边缘
        for (int rr = 0; rr < 6; ++rr)
            memset(canny.ptr(rr), 0, 6);
        for (int rr = _image.rows - 5; rr < _image.rows; ++rr)
            memset(canny.ptr(rr) + _image.cols - 5, 0, 5);
        lineDetection(_image, segments, canny);
    }
}

// threads=2 空间拆分：左右两半并行各自跑完整管线（工作线程用独立子检测
// 器实例，内部串行），拼接后按 do_merge 做一次接缝缝合——跨缝被切断的
// 共线段由现有合并逻辑重新接上。确定性输出（无竞态时序依赖）。
// 两半各带 SEAM_MARGIN 像素重叠余量：跨缝线在两半中各有一份完整拷贝，
// 重叠共线段经合并还原为整线（无余量则跨缝线只能保持切断状态）
void FmlDetector::detectSegmentsDual(const cv::Mat &img,
                                     std::vector<Segment> &out)
{
    const int W = img.cols, H = img.rows;
    const int Wl = (W + 1) / 2;
    const int SEAM_MARGIN = 16; // 两半的重叠余量（像素）
    if (!wk_det_)
    {
        // 工作线程子检测器：同参数、内部串行；懒创建常驻（与线程池同策略）
        wk_det_ = new FmlDetector(threshold_length, threshold_dist,
                                  canny_th1, canny_th2, grad_sigma, do_merge,
                                  nfa_eps, nfa_ang_tol_deg_, shift_px_,
                                  elsed_, ec_, 1);
    }
    cv::Mat L = img(cv::Rect(0, 0, std::min(W, Wl + SEAM_MARGIN), H));
    cv::Mat R = img(cv::Rect(std::max(0, Wl - SEAM_MARGIN), 0,
                             W - std::max(0, Wl - SEAM_MARGIN), H));
    dual_seg_l_.clear();
    dual_seg_r_.clear();
    FmlPool *fp = FmlPool::instance();
    fp->post(0, 0, [&](int, int)
             {
                 wk_det_->detectSegments(R, dual_seg_r_);
             });
    const int saved = nthr_;
    nthr_ = 1; // 本实例内层子检测走串行（防池重入）
    detectSegments(L, dual_seg_l_);
    nthr_ = saved;
    fp->waitDone();
    // 右半坐标还原：ROI 内为局部坐标，补回 x 偏移
    const int rx = std::max(0, Wl - SEAM_MARGIN);
    for (size_t i = 0; i < dual_seg_r_.size(); ++i)
    {
        dual_seg_r_[i].x1 += (float)rx;
        dual_seg_r_[i].x2 += (float)rx;
    }
    // 拼接 + 接缝缝合：跨缝切断的共线段由现有合并逻辑重新接上
    if (do_merge)
    {
        dual_seg_l_.insert(dual_seg_l_.end(), dual_seg_r_.begin(),
                           dual_seg_r_.end());
        mergeAndCollect(img, dual_seg_l_, out);
    }
    else
    {
        out = dual_seg_l_;
        out.insert(out.end(), dual_seg_r_.begin(), dual_seg_r_.end());
    }
}

// 返回 log10(NFA)；NFA = (NM)^(5/2) * sum_{k=m..n} C(n,k) p^k (1-p)^(n-k)
// （a-contrario 虚警数判据，定义见 [2]，检测框架用法见 [1]）
// lgamma/log 查表；尾和按峰值归一，exp 项过小即早停
double FmlDetector::logNfa(int n, int m) const
{
    if (n <= 0)
        return 0.0;
    double p = (double)nfa_ang_tol / CV_PI;
    // 防御：容差退化（<=0 或 >=180°）时钳位，避免 log(0)/log(负) 产生 NaN
    if (p < 1e-9)
        p = 1e-9;
    else if (p > 1.0 - 1e-9)
        p = 1.0 - 1e-9;
    double l1p = std::log(1.0 - p);
    double lp = std::log(p);
    double lc = lgT(n) - lgT(m) - lgT(n - m);
    double logterm = lc + (double)m * lp + (double)(n - m) * l1p;
    double maxlog = logterm;
    // 固定容量栈缓冲；项数超限时回退堆分配
    double stack_logs[2048];
    const bool use_stack = (n - m) < 2047;
    double *const logs = use_stack ? stack_logs : new double[n - m + 1];
    int nlogs = 0;
    logs[nlogs++] = logterm;
    bool past_peak = false;
    for (int k = m; k < n; ++k)
    {
        logterm += logT(n - k) - logT(k + 1) + lp - l1p;
        if (logterm > maxlog)
        {
            maxlog = logterm;
            past_peak = true;
        }
        else if (past_peak && logterm < maxlog - 45.0)
        {
            break; // 后续项 exp 下溢，可截断
        }
        logs[nlogs++] = logterm;
    }
    double sum = 0.0;
    for (int i = 0; i < nlogs; ++i)
        sum += std::exp(logs[i] - maxlog);
    if (!use_stack)
        delete[] logs;
    double log_tail = maxlog + std::log(sum);
    double nfa = 2.5 * std::log10((double)imagewidth * (double)imageheight) + log_tail / std::log(10.0);
    return nfa;
}

// 按长度加权合并两条线段：角度取加权平均，端点取合并方向上的投影极值
void FmlDetector::mergeLines(const Segment &seg1, const Segment &seg2, Segment &seg_merged)
{
    double xg = 0.0, yg = 0.0;
    float ax = seg1.x1, ay = seg1.y1;
    float bx = seg1.x2, by = seg1.y2;
    float cx = seg2.x1, cy = seg2.y1;
    float dx = seg2.x2, dy = seg2.y2;

    float dlix = (bx - ax), dliy = (by - ay);
    float dljx = (dx - cx), dljy = (dy - cy);
    double li = std::sqrt((double)dlix * dlix + (double)dliy * dliy);
    double lj = std::sqrt((double)dljx * dljx + (double)dljy * dljy);

    xg = (li * ((double)ax + bx) + lj * ((double)cx + dx)) / (2.0 * (li + lj));
    yg = (li * ((double)ay + by) + lj * ((double)cy + dy)) / (2.0 * (li + lj));

    double thi = (dlix == 0.0f) ? CV_PI / 2.0 : std::atan((double)dliy / dlix);
    double thj = (dljx == 0.0f) ? CV_PI / 2.0 : std::atan((double)dljy / dljx);
    double thr;
    if (std::fabs(thi - thj) <= CV_PI / 2.0)
    {
        thr = (li * thi + lj * thj) / (li + lj);
    }
    else
    {
        double tmp = thj - CV_PI * (thj / std::fabs(thj));
        thr = (li * thi + lj * tmp) / (li + lj);
    }

    double axg = ((double)ay - yg) * std::sin(thr) + ((double)ax - xg) * std::cos(thr);
    double bxg = ((double)by - yg) * std::sin(thr) + ((double)bx - xg) * std::cos(thr);
    double cxg = ((double)cy - yg) * std::sin(thr) + ((double)cx - xg) * std::cos(thr);
    double dxg = ((double)dy - yg) * std::sin(thr) + ((double)dx - xg) * std::cos(thr);

    double d1 = std::min(axg, std::min(bxg, std::min(cxg, dxg)));
    double d2 = std::max(axg, std::max(bxg, std::max(cxg, dxg)));

    seg_merged.x1 = (float)(d1 * std::cos(thr) + xg);
    seg_merged.y1 = (float)(d1 * std::sin(thr) + yg);
    seg_merged.x2 = (float)(d2 * std::cos(thr) + xg);
    seg_merged.y2 = (float)(d2 * std::sin(thr) + yg);
}

// 合并判定：seg2 中点到 seg1 直线的距离、两中点间距、角度差均在阈值内
bool FmlDetector::mergeSegments(const Segment &seg1, const Segment &seg2, Segment &seg_merged)
{
    double o[3] = {(seg2.x1 + seg2.x2) / 2.0, (seg2.y1 + seg2.y2) / 2.0, 1.0};
    double a[3] = {(double)seg1.x1, (double)seg1.y1, 1.0};
    double b[3] = {(double)seg1.x2, (double)seg1.y2, 1.0};
    double l1[3];
    cross3(a, b, l1);
    normalizeLine(l1);

    cv::Point2f seg1mid, seg2mid;
    seg1mid.x = (seg1.x1 + seg1.x2) / 2.0f;
    seg1mid.y = (seg1.y1 + seg1.y2) / 2.0f;
    seg2mid.x = (seg2.x1 + seg2.x2) / 2.0f;
    seg2mid.y = (seg2.y1 + seg2.y2) / 2.0f;

    float seg1len = std::sqrt((seg1.x1 - seg1.x2) * (seg1.x1 - seg1.x2) +
                              (seg1.y1 - seg1.y2) * (seg1.y1 - seg1.y2));
    float seg2len = std::sqrt((seg2.x1 - seg2.x2) * (seg2.x1 - seg2.x2) +
                              (seg2.y1 - seg2.y2) * (seg2.y1 - seg2.y2));
    float middist = std::sqrt((seg1mid.x - seg2mid.x) * (seg1mid.x - seg2mid.x) +
                              (seg1mid.y - seg2mid.y) * (seg1mid.y - seg2mid.y));
    float angdiff = std::fabs(seg1.angle - seg2.angle);
    float dist = (float)distPointLine(o, l1);

    if (std::fabs(dist) <= threshold_dist * 2.0f &&
        middist <= seg1len / 2.0f + seg2len / 2.0f + 20.0f &&
        angdiff <= (float)CV_PI / 180.0f * 5.0f)
    {
        mergeLines(seg1, seg2, seg_merged);
        // 置信度按联合支撑点重算 NFA（两端置信度相加相当于 NFA 相乘，
        // 会重复计入全图试验次数因子）
        seg_merged.sup_n = seg1.sup_n + seg2.sup_n;
        seg_merged.sup_m = seg1.sup_m + seg2.sup_m;
        seg_merged.conf = (float)(-logNfa(seg_merged.sup_n, seg_merged.sup_m));
        return true;
    }
    return false;
}

// pt 在直线 l 上的垂足，并钳位到图像范围内
void FmlDetector::incidentPoint(const double l[3], cv::Point2f &pt)
{
    double xk[3] = {(double)pt.x, (double)pt.y, 1.0};
    double lh[3] = {l[0], l[1], 0.0};
    double lk[3], x2[3];
    cross3(xk, lh, lk);
    cross3(lk, l, x2);
    double inv = 1.0 / x2[2];
    float xf = (float)(x2[0] * inv);
    float yf = (float)(x2[1] * inv);

    cv::Point2f pt_tmp;
    pt_tmp.x = xf < 0.0f                   ? 0.0f
               : xf >= (imagewidth - 1.0f) ? (imagewidth - 1.0f)
                                           : xf;
    pt_tmp.y = yf < 0.0f                    ? 0.0f
               : yf >= (imageheight - 1.0f) ? (imageheight - 1.0f)
                                            : yf;
    pt = pt_tmp;
}
void FmlDetector::pointInboardTest(const cv::Size srcSize, cv::Point2i &pt)
{
    pt.x = pt.x <= 5 ? 5 : pt.x >= srcSize.width - 5 ? srcSize.width - 5
                                                     : pt.x;
    pt.y = pt.y <= 5 ? 5 : pt.y >= srcSize.height - 5 ? srcSize.height - 5
                                                      : pt.y;
}

// 合并阶段（传统 canny 管线与 ELSED 管线共用）：ith 从尾向前，jth 向左扫描
// 尝试两两合并；合并成功则结果写入 ith、jth 标记删除并重置游标，
// 存活数 <2 停止；存活线段按序回填 segments_all
void FmlDetector::mergeAndCollect(const cv::Mat &src,
                                  std::vector<Segment> &segments_tmp,
                                  std::vector<Segment> &segments_all)
{
    if ((int)segments_tmp.size() >= 2)
    {
        const int n = (int)segments_tmp.size();
        std::vector<char> &alive = alive_buf_;
        std::vector<int> &left = left_buf_;
        alive.assign(n, 1);
        left.resize(n);
        for (int i = 0; i < n; ++i)
            left[i] = i - 1; // 物理左邻居

        // 向左找第一个存活（含哨兵 -1）
        auto stepLeft = [&](int i)
        {
            while (i >= 0 && !alive[i])
                i = left[i];
            return i;
        };
        // i 严格左侧最近的存活元素
        auto prevAlive = [&](int i)
        {
            return stepLeft(left[i]);
        };

        int aliveCount = n;
        int ith = n - 1;
        int jth = prevAlive(ith);
        while (aliveCount > 1 && ith >= 0 && jth >= 0)
        {
            Segment seg_merged;
            if (mergeSegments(segments_tmp[ith], segments_tmp[jth], seg_merged))
            {
                additionalOperationsOnSegment(src, seg_merged);
                segments_tmp[ith] = seg_merged;
                alive[jth] = 0;
                --aliveCount;
                jth = prevAlive(ith);
            }
            else
            {
                jth = prevAlive(jth);
            }
            if (jth < 0)
            {
                ith = prevAlive(ith);
                jth = ith >= 0 ? prevAlive(ith) : -1;
            }
        }
        for (int i = 0; i < n; ++i)
            if (alive[i])
                segments_all.push_back(segments_tmp[i]);
    }
}

inline void FmlDetector::getAngle(Segment &seg)
{
    seg.angle = (float)(cv::fastAtan2(seg.y2 - seg.y1, seg.x2 - seg.x1) / 180.0f * (float)CV_PI);
}

// 线段方向统一：沿线取 10 个采样点，比较法向 ±1 像素两侧的灰度和，
// 右侧更亮则交换端点
void FmlDetector::additionalOperationsOnSegment(const cv::Mat &src, Segment &seg)
{
    if (seg.x1 == 0.0f && seg.x2 == 0.0f && seg.y1 == 0.0f && seg.y2 == 0.0f)
        return;

    getAngle(seg);

    cv::Point2f start(seg.x1, seg.y1);
    cv::Point2f end(seg.x2, seg.y2);

    double dx = (double)end.x - (double)start.x;
    double dy = (double)end.y - (double)start.y;

    // 法向 (cosv,sinv) = (cos(π/2+ang), sin(π/2+ang))，由段向量直接给出
    //（cos(π/2+ang) = −dy/len，sin(π/2+ang) = dx/len）
    const double dlen = std::sqrt(dx * dx + dy * dy);
    const double cosv = dlen > 0.0 ? -dy / dlen : 0.0;
    const double sinv = dlen > 0.0 ? dx / dlen : 0.0;

    const int num_points = 10;
    cv::Point2f points[num_points];
    points[0] = start;
    points[num_points - 1] = end;
    for (int i = 0; i < num_points; ++i)
    {
        if (i == 0 || i == num_points - 1)
            continue;
        points[i].x = points[0].x + (float)((float)dx / (float)(num_points - 1) * (float)i);
        points[i].y = points[0].y + (float)((float)dy / (float)(num_points - 1) * (float)i);
    }

    cv::Point2i points_right[num_points], points_left[num_points];
    double gap = 1.0;
    for (int i = 0; i < num_points; ++i)
    {
        points_right[i].x = cvRound(points[i].x + (float)(gap * cosv));
        points_right[i].y = cvRound(points[i].y + (float)(gap * sinv));
        points_left[i].x = cvRound(points[i].x - (float)(gap * cosv));
        points_left[i].y = cvRound(points[i].y - (float)(gap * sinv));
        pointInboardTest(src.size(), points_right[i]);
        pointInboardTest(src.size(), points_left[i]);
    }

    // 行指针采样源灰度值
    const size_t stepv = src.step;
    const unsigned char *base = src.ptr(0);
    int iR = 0, iL = 0;
    for (int i = 0; i < num_points; ++i)
    {
        iR += base[points_right[i].y * stepv + points_right[i].x];
        iL += base[points_left[i].y * stepv + points_left[i].x];
    }

    if (iR > iL)
    {
        std::swap(seg.x1, seg.x2);
        std::swap(seg.y1, seg.y2);
        getAngle(seg);
    }
}
} // namespace fml

// 非 AVX2 构建提示：标量回退功能完整，仅性能较低。
// 进程内只打印一次（stderr）
static inline void warnNoAvx2Once()
{
#if !FML_AVX2
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        fprintf(stderr, "[fml] 当前环境不支持 AVX2，已使用标量回退实现"
                        "（性能较低；如需加速请使用支持 AVX2 的构建/CPU）\n");
    }
#endif
}

int fml_detect(const cv::Mat &gray, const fml_params &p,
               std::vector<cv::Vec4f> &lines,
               std::vector<float> *confidences)
{
    warnNoAvx2Once();
    fmlMallocTune();
    if (gray.empty() || gray.type() != CV_8UC1)
        return -1;

    // 参数未变化时复用实例，跨调用共享内部缓冲
    static thread_local std::unique_ptr<FmlDetector> det;
    static thread_local fml_params cur;
    static thread_local bool init = false;
    bool changed = !init ||
                   cur.length_threshold != p.length_threshold ||
                   cur.distance_threshold != p.distance_threshold ||
                   cur.canny_th1 != p.canny_th1 || cur.canny_th2 != p.canny_th2 ||
                   cur.grad_sigma != p.grad_sigma ||
                   cur.do_merge != p.do_merge ||
                   cur.nfa_eps != p.nfa_eps ||
                   cur.nfa_ang_tol_deg != p.nfa_ang_tol_deg ||
                   cur.shift_px != p.shift_px ||
                   cur.threads != p.threads;
    if (changed)
    {
        det.reset(new FmlDetector(p.length_threshold, p.distance_threshold,
                                  p.canny_th1, p.canny_th2,
                                  p.grad_sigma, p.do_merge != 0,
                                  p.nfa_eps, p.nfa_ang_tol_deg,
                                  p.shift_px, false, ElsedCfg{}, p.threads));
        cur = p;
        init = true;
    }

    // 出参缓冲跨调用复用（thread_local，clear 保留容量）：稳态零分配
    static thread_local std::vector<cv::Vec4f> out;
    static thread_local std::vector<float> confs;
    det->detect(gray, out, confidences ? &confs : nullptr);
    lines.swap(out);
    if (confidences)
        confidences->swap(confs);
    return (int)lines.size();
}

int fml_detect_elsed(const cv::Mat &gray, const fml_elsed_params &p,
                     std::vector<cv::Vec4f> &lines,
                     std::vector<float> *confidences)
{
    warnNoAvx2Once();
    if (gray.empty() || gray.type() != CV_8UC1)
        return -1;

    // 与 fml_detect 相互独立：独立实例缓存与出参缓冲
    static thread_local std::unique_ptr<FmlDetector> det;
    static thread_local fml_elsed_params cur;
    static thread_local bool init = false;
    bool changed = !init ||
                   cur.length_threshold != p.length_threshold ||
                   cur.distance_threshold != p.distance_threshold ||
                   cur.do_merge != p.do_merge ||
                   cur.shift_px != p.shift_px ||
                   cur.grad_th != p.grad_th ||
                   cur.anchor_th != p.anchor_th ||
                   cur.scan_intv != p.scan_intv ||
                   cur.min_len != p.min_len ||
                   cur.fit_err != p.fit_err ||
                   cur.px_dist != p.px_dist ||
                   cur.validate_th != p.validate_th ||
                   cur.treat_junc != p.treat_junc ||
                   cur.sigma != p.sigma ||
                   cur.threads != p.threads;
    if (changed)
    {
        const ElsedCfg ec = ElsedCfg::sanitized(
            p.grad_th, p.anchor_th, p.scan_intv, p.min_len,
            p.fit_err, p.px_dist, p.validate_th,
            p.treat_junc != 0, p.sigma);
        // NFA 角容差取传统管线默认 22.5°：ELSED 的置信度输出仍为
        // 有意义的 -log10(NFA)（参考值）；传 0 会使 logNfa 中 log(p)
        // 为 -inf → 置信度 NaN，下游按置信度排序会触发未定义行为
        det.reset(new FmlDetector(p.length_threshold, p.distance_threshold,
                                  0.0, 0.0, 0.0f, p.do_merge != 0,
                                  0.0f, 22.5f, p.shift_px, true, ec,
                                  p.threads));
        cur = p;
        init = true;
    }

    static thread_local std::vector<cv::Vec4f> out;
    static thread_local std::vector<float> confs;
    det->detect(gray, out, confidences ? &confs : nullptr);
    lines.swap(out);
    if (confidences)
        confidences->swap(confs);
    return (int)lines.size();
}

// 结果绘制

namespace
{

// 置信度 → BGR 颜色查表（256 级，首次调用构建一次）：
// t=0 浅绿 → t=1 深绿，逐通道线性插值后经伽马重映射
constexpr float kColLo[3] = {144.0f, 238.0f, 144.0f}; // 浅绿（低置信度端）
constexpr float kColHi[3] = {0.0f, 100.0f, 0.0f};     // 深绿（高置信度端）
constexpr float kGamma = 0.45f;                       // <1 使颜色分布偏向深绿

cv::Vec3b drawColorLUT(float t)
{
    static const std::array<cv::Vec3b, 256> lut = []
    {
        std::array<cv::Vec3b, 256> tab;
        for (int i = 0; i < 256; ++i)
        {
            const float u = std::pow(i / 255.0f, kGamma);
            cv::Vec3b v;
            for (int c = 0; c < 3; ++c)
            {
                const float x = kColLo[c] + (kColHi[c] - kColLo[c]) * u;
                v[c] = (unsigned char)(x < 0.0f     ? 0.0f
                                       : x > 255.0f ? 255.0f
                                                    : x + 0.5f);
            }
            tab[i] = v;
        }
        return tab;
    }();
    int idx = (int)(t * 255.0f + 0.5f);
    idx = idx < 0 ? 0 : idx > 255 ? 255
                                  : idx;
    return lut[idx];
}

} // namespace

void fml_draw_lines(cv::Mat &img, const std::vector<cv::Vec4f> &lines,
                    const std::vector<float> &confs,
                    double scale_x, double scale_y, int thickness,
                    float cmin, float cmax)
{
    if (img.empty() || img.type() != CV_8UC3)
        return;
    const float range = cmax - cmin;
    auto key = [&](size_t i)
    {
        return i < confs.size() ? confs[i] : cmin;
    };
    // 按置信度升序绘制：高置信度线段后画，位于顶层
    std::vector<size_t> order(lines.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b)
                     {
                         return key(a) < key(b);
                     });
    for (size_t i : order)
    {
        const cv::Vec4f &l = lines[i];
        const float t = (range > 0.0f) ? (key(i) - cmin) / range : 0.5f;
        cv::line(img,
                 cv::Point(cvRound(l[0] * scale_x), cvRound(l[1] * scale_y)),
                 cv::Point(cvRound(l[2] * scale_x), cvRound(l[3] * scale_y)),
                 drawColorLUT(t), thickness, cv::LINE_AA);
    }
}
