// 项目名: fml_for_x86_64
// 文件名: detector.hpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// 内部核心枢纽头文件：完整定义 FmlDetector 类（成员变量与全部方法声明）
// 及两前端共享的数据结构（ElsedCfg / ElsedSegment / ElsedBranch）。
// fld.cpp（传统 Canny/FLT 前端）与 elsed.cpp（ELSED 前端）均包含此文件
// 共享同一个类定义；方法的实现分布：
//   detector.cpp  构造/顶层调度/两端共享的后处理流水线（后端公共仓库）
//   fld.cpp       传统管线前端（梯度/NMS/滞后连通/链游走/最小二乘拟合）
//   elsed.cpp     ELSED 前端（锚点扫描/梯度路由绘制/交叉点跳越/段验证）
// 不对外公开，用户只需包含 fml.hpp。
//
// 内部类型统一置于具名 namespace fml：类成员定义跨多个编译单元互相
// 调用，匿名 namespace 的符号名跨 TU 不一致，不可用。fml:: 符号仅
// 静态链接内部使用，不进入公开接口。

#pragma once

#include "fml.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace fml
{
// ELSED 可调参数包。
// 由 fml_elsed_params 的字段经 fml_detect_elsed 钳位后传入
struct ElsedCfg
{
    double grad_th = 30.0;     // L1 幅值低于它的像素清零
    int anchor_th = 8;         // 锚点相对邻居的幅值超出量
    int scan_intv = 2;         // 锚点扫描间隔（行/列）
    int min_len = 15;          // 触发拟合的最短像素数
    double fit_err = 0.2;      // 拟合激活 MSE 阈值
    double px_dist = 1.5;      // 内点距离容差
    double validate_th = 0.15; // 验证角误差阈值（弧度）
    bool treat_junc = true;    // 交叉点跳越开关
    double sigma = 1.0;        // 高斯平滑 σ（5x5 核）；<=0 不平滑

    // 入参钳位：扫描间隔 0 会死循环、min_len<2 无法初始化拟合窗口
    static ElsedCfg sanitized(double grad_th, int anchor_th, int scan_intv,
                              int min_len, double fit_err, double px_dist,
                              double validate_th, bool treat_junc, double sigma)
    {
        ElsedCfg c;
        c.grad_th = grad_th < 0.0 ? 0.0 : grad_th;
        c.anchor_th = anchor_th < 0 ? 0 : (anchor_th > 255 ? 255 : anchor_th);
        c.scan_intv = scan_intv < 1 ? 1 : scan_intv;
        c.min_len = min_len < 2 ? 2 : min_len;
        c.fit_err = fit_err < 0.0 ? 0.0 : fit_err;
        c.px_dist = px_dist < 0.0 ? 0.0 : px_dist;
        c.validate_th = validate_th < 0.0 ? 0.0 : validate_th;
        c.treat_junc = treat_junc;
        c.sigma = sigma < 0.0 ? 0.0 : sigma;
        return c;
    }
};

// 线段及其支撑像素链、增量最小二乘状态；pixels 指向全局像素链
//（els_pixels_），firstPxIndex..lastPxIndex 为该段在链中的窗口
struct ElsedSegment
{
    int64_t sum_x_i = 0, sum_y_i = 0, sum_x_i_y_i = 0, sum_x_i_2 = 0;
    uint32_t N = 0;
    int dx = 0, dy = 0;
    bool isHorizontal = false; // 以 x 为自变量（否则以 y，拟合方向翻转）
    cv::Vec4f endpoints;       // 投影到拟合线上的端点
    cv::Vec3f equation;        // ax+by+c=0（已归一化，符号沿 (dx,dy)）
    cv::Point firstPx, prevFirstPx, lastPx, prevLastPx;
    int firstPxIndex = -1, lastPxIndex = -1;
    bool arePixelsSorted = true;
    bool firstEndpointExtended = false;
    bool secondEndpointExtended = false;
    const std::vector<cv::Point> *pixels = nullptr;

    void init(const std::vector<cv::Point> &pts, int startIdx);
    void skipPositions();
    void addPixel(int x, int y, int pixelIndexInEdge, bool isPixelAtTheEnd);
    void finish();
    void removeLastPx(bool removeFromTheEnd = true);

    // 符号统一（使方程沿 (dx,dy) 方向为正）。仅在消费符号语义时调用
    void ensureSigned();

    inline double getFitError() const
    {
        // 窗口内到直线距离平方均值（float 域累计）
        float dist, fitError = 0;
        for (int i = firstPxIndex; i <= lastPxIndex; ++i)
        {
            dist = equation[0] * (*pixels)[i].x + equation[1] * (*pixels)[i].y + equation[2];
            fitError += dist * dist;
        }
        return fitError / N;
    }
    inline bool isInlier(int x, int y, double lineFitErrThreshold) const
    {
        const float pointToLineDis = equation[0] * x + equation[1] * y + equation[2];
        return std::fabs(pointToLineDis) < (float)lineFitErrThreshold;
    }
    inline bool horizontal() const
    {
        return isHorizontal;
    }
    inline int getNumOfPixels() const
    {
        return (int)N;
    }
    inline const cv::Vec4f &getEndpoints() const
    {
        return endpoints;
    }
    inline const cv::Vec3f &getLineEquation() const
    {
        return equation;
    }
    inline const cv::Point &getFirstPixel() const
    {
        return firstPx;
    }
    inline const cv::Point &getLastPixel() const
    {
        return lastPx;
    }
    inline bool hasSecondSideElements() const
    {
        return !arePixelsSorted;
    }

private:
    void leastSquareLineFit(const std::vector<cv::Point> &pts, int startIdx);
    void leastSquaresLineFitNewPoint(int x, int y);
    void subtractPointFromModel(const cv::Point &p);
    void calculateLineEq();
    void calcSegmentEndpoints();
};

// EdgeDrawer 分支栈节点
struct ElsedBranch
{
    uint8_t direction;             // 本分支起始行进方向
    cv::Point px;                  // 起始像素
    bool addPixelsForTheFirstSide; // true=接到段尾 / false=接到段头
    std::vector<cv::Point> pixels; // 预置像素（第二方向/断缝续走用）
};

// 内部检测器类
class FmlDetector
{
public:
    FmlDetector(int length_threshold, float distance_threshold,
                double canny_th1, double canny_th2,
                float grad_sigma, bool do_merge,
                float nfa_eps, float nfa_ang_tol_deg, int shift_px,
                bool elsed, const ElsedCfg &ec, int threads = 1);

    // 输入必须为 CV_8UC1 灰度图，检测结果写入 lines；
    // confidences 非空时写入与 lines 一一对应的每条线置信度 -log10(NFA)
    void detect(const cv::Mat &gray, std::vector<cv::Vec4f> &lines,
                std::vector<float> *confidences = nullptr);

private:
    struct Segment
    {
        float x1, y1, x2, y2, angle;
        float conf;       // 置信度 -log10(NFA)（merge 按联合支撑点重算）
        int sup_n, sup_m; // 支撑点数 / 其中梯度对齐数（merge 时求和）
    };

    int imagewidth, imageheight, threshold_length;
    float threshold_dist;
    double canny_th1, canny_th2;
    float grad_sigma; // 梯度平滑尺度（σ）；0 = 无平滑
    bool do_merge;
    float nfa_eps;
    float nfa_ang_tol;
    float tan_tol_;                 // tan(nfa_ang_tol)
    int shift_px_ = -1;             // 边缘对先验平移（像素），见 fml_params::shift_px
    double log10_nfa_eps_ = 0;      // log10(nfa_eps)，显著性判定阈值（构造期预计算）
    float nfa_ang_tol_deg_ = 22.5f; // 构造角度容差留存（工作线程子检测器重建用）
    bool elsed_;                    // ELSED 管线开关
    ElsedCfg ec_;                   // ELSED 可调参数包（elsed=1 时生效）

    cv::Mat dx_, dy_;                                    // Sobel 梯度（CV_16S，canny 与梯度角共用一份）
    cv::Mat hbuf_, vbuf_;                                // 3x3 路径中间缓冲（横向/纵向差分行，含哨兵）
    cv::Mat pa_, pb_;                                    // 5 点路径工作平面（CV_16S）
    float ks_[5];                                        // 平滑核（按 σ 采样归一化，Σ=1）
    float kd_[5];                                        // 差分核（高斯一阶导采样，Σu·kd=1）
    int ks_q_[5];                                        // 平滑核 Q8.8 定点（AVX2 路径）
    int kd_q_[5];                                        // 差分核 Q8.8 定点（已含符号，中心=0）
    cv::Mat nms_map_;                                    // canny 边缘图 (rows+2)x(cols+2)，0/255 语义
    std::vector<short> mag_buf_;                         // 3 行幅值滚动缓冲（i16，单线程路径）
    std::vector<short> mag_full_;                        // 全图幅值缓冲（i16，双线程路径，复用免分配）
    std::vector<unsigned char *> nms_stack_;             // 滞后连通栈
    std::vector<unsigned char *> nms_stack_wk_;          // 滞后连通栈（工作线程半区）
    int nthr_ = 1;                                       // 工作线程数（1/2，fml_params::threads）
    std::vector<cv::Point2i> l_points_buf;               // 拟合支撑点缓冲（复用免分配）
    std::vector<cv::Point2i> ld_points_;                 // 链游走点缓冲（每帧复用）
    std::vector<Segment> ld_segments_, ld_segments_tmp_; // 线段缓冲
    std::vector<char> alive_buf_;                        // merge 存活标记（复用免分配）
    std::vector<int> left_buf_;                          // merge 左邻索引链（复用免分配）

    // ELSED 模式（elsed=1）工作区
    cv::Mat els_dx_, els_dy_;                      // Sobel3 梯度（i16）
    cv::Mat els_g_;                                // 阈值化 L1 幅值（i16，<30 清零）
    cv::Mat els_dir_;                              // 2 类方向图（u8，0/255）
    cv::Mat els_edge_;                             // 边缘状态图（u8）
    std::vector<cv::Point> els_pixels_;            // 全局像素链（跨锚点累加）
    std::vector<ElsedSegment> els_segments_;       // 段池（运行时末尾为工作快照）
    std::vector<ElsedBranch> els_stack_;           // 绘制分支栈
    std::vector<cv::Point> els_anchor_px_;         // 复用缓冲：锚点尾像素留存
    std::vector<cv::Point> els_ext_px_;            // 复用缓冲：交叉点延伸像素
    std::vector<cv::Point> els_out_px_;            // 复用缓冲：外点列表
    std::vector<cv::Point> els_anchors_;           // 复用缓冲：锚点表
    std::vector<cv::Point> els_anchors_wk_;        // 锚点列拆分工作线程复用缓冲
    std::vector<Segment> els_validated_;           // 复用缓冲：验证通过的段
    std::vector<Segment> els_validated_wk_;        // 段验证拆分工作线程复用缓冲
    FmlDetector *wk_det_ = nullptr;                // threads=2 空间拆分：工作线程子检测器
                                                   // （同参数、内部串行；懒创建常驻不销毁）
    std::vector<Segment> dual_seg_l_, dual_seg_r_; // 左右半检测结果复用缓冲
    std::vector<Segment> els_kept_;                // 复用缓冲：过滤后的段
    const short *els_gImg_ = nullptr;              // 行指针缓存（init 后固定）
    const unsigned char *els_dirImg_ = nullptr;
    unsigned char *els_edgeImg_ = nullptr;
    const short *els_pDxImg_ = nullptr; // ELSED 梯度行指针（验证/统计用）
    const short *els_pDyImg_ = nullptr;
    int els_w_ = 0, els_h_ = 0;

    void computeGradientAngle(const cv::Mat &gray); // 按 grad_sigma 路由
    void sobelSeparable(const cv::Mat &gray);       // 5 点可分离梯度（权重按 σ 预生成）
    void cannyFromGradient(int low, int high);      // NMS+滞后连通，结果在 nms_map_
    double logNfa(int n, int m) const;              // log10(NFA)，NFA<1 即 logNfa<0
    // 支撑点梯度方向是否与线垂直对齐（点值由调用方读取传入，统计/判定共用一次访存）
    // 逐像素热路径小函数：类内实现保证各前端 TU 内联（跨 TU 拆分会丢失内联机会）
    bool isAligned(float gx, float gy, float ex, float ey) const
    {
        // 取 |dot|：模 π 等价（θg 与 θg+π 同判）；退化 dot=0 → |tan|=∞ 必不对齐
        const float dot = std::fabs(gx * ex + gy * ey);
        if (dot <= 0.0f)
            return false;
        // 夹角正切 = cross/|dot| ≤ tan(tol)
        return std::fabs(gx * ey - gy * ex) <= dot * tan_tol_;
    }

    void incidentPoint(const double l[3], cv::Point2f &pt);
    void mergeLines(const Segment &seg1, const Segment &seg2, Segment &seg_merged);
    bool mergeSegments(const Segment &seg1, const Segment &seg2, Segment &seg_merged);
    void extractSegments(const std::vector<cv::Point2i> &points, std::vector<Segment> &segments);
    bool getPointChainCached(const unsigned char *const rows[3], int x, int y,
                             cv::Point &chained_pt, float &direction, int step) const;
    void lineDetection(const cv::Mat &src, std::vector<Segment> &segments_all,
                       cv::Mat canny);
    // 单图完整管线（梯度→canny/绘制→链提取→合并），供双线程左右半各自调用
    void detectSegments(const cv::Mat &img, std::vector<Segment> &segments);
    // threads=2 空间拆分：左右两半并行各自完整检测，拼接 + 接缝共线缝合
    void detectSegmentsDual(const cv::Mat &img, std::vector<Segment> &segments);
    void mergeAndCollect(const cv::Mat &src, std::vector<Segment> &segments_tmp,
                         std::vector<Segment> &segments_all); // 两模式共用的合并阶段
    void pointInboardTest(const cv::Size srcSize, cv::Point2i &pt);
    inline void getAngle(Segment &seg);
    void additionalOperationsOnSegment(const cv::Mat &src, Segment &seg);

    // ELSED 管线
    void detectElsed(const cv::Mat &src, std::vector<Segment> &segments_all);
    void elsedComputeGradients(const cv::Mat &gray); // Gauss5x5(σ=1)+Sobel3 → g/dir
    void elsedComputeAnchorPoints(std::vector<cv::Point> &anchors, int anchorThresh,
                                  int w0 = 1, int w1 = -1) const;
    void elsedDrawAnchorPoints(const std::vector<cv::Point> &anchors);
    void elsedDrawEdgeInBothDirections(uint8_t direction, cv::Point anchor);
    void elsedDrawEdgeTreeStack(cv::Point anchor,
                                std::vector<cv::Point> &initialPixels,
                                bool firstAnchorDirection);
    bool elsedFindNextPxWithGradient(uint8_t pxGradDirection,
                                     cv::Point &px, cv::Point &lastPx) const;
    bool elsedCanSegmentBeExtended(ElsedSegment &segment, bool extendByTheEnd,
                                   std::vector<cv::Point> &pixelsInTheExtension);
    void elsedAddJunctionPixelsToSegment(const std::vector<cv::Point> &junctionPixels,
                                         ElsedSegment &segment,
                                         bool addPixelsForTheFirstSide);
};
} // namespace fml
