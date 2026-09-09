// 项目名: fml_for_x86_64
// 文件名: fml.hpp
// Copyright (c) 2026 player404.
// SPDX-License-Identifier: MIT
//
// 直线段检测器，双管线：
//   fml_detect        Sobel 梯度 → Canny 边缘 → 边缘链提取 → 最小二乘拟合
//                     → NFA 显著性验证 → 先验平移 → 共线合并
//   fml_detect_elsed  梯度锚点扫描 → 边缘绘制游走 → 段角度验证 →
//                     接入同一后段（长度/贴边过滤、先验平移、方向统一、共线合并）
//
// 编译器需求：Linux 端 GCC；Windows 端 MinGW。MSVC 部分指令集不兼容，可根据需求重写
// 架构说明：ARM（树莓派、rk3588 等）使用需改用 NEON 指令集重写
// 依赖：opencv4.x+
//
// 用法（ELSED 管线）：
//   cv::Mat gray = ...;                    // 8 位单通道灰度图
//   fml_elsed_params p;                    // 默认值即推荐值
//   std::vector<cv::Vec4f> lines;
//   std::vector<float> confs;              // 可选：接收每条线的置信度
//   int n = fml_detect_elsed(gray, p, lines, &confs);
//
// 用法（默认管线）：
//   fml_params p;                          // 默认参数，可按需修改字段
//   int n = fml_detect(gray, p, lines, &confs);
//
// lines[i] = (x1, y1, x2, y2)，图像像素坐标；
// confs[i] = 置信度 -log10(NFA)：fml_detect 恒正；
//            fml_detect_elsed 仅参考，可为负（验证用角度判据）
//
// 推荐输入分辨率 320*240（大多数情况下已经足够，图片像素数和处理速度基本成正比）
// 低分辨率也有平滑的作用，建议自行根据需求选择

// v2.0更新说明
// 0 接口改名为fml，引入更多功能；
// 1 引入ELSED检测管线，速度更快但检测行为与经典管线略有不同，通过ELSED接口调用；引入标准的绘制程序；
//   *ELSED与传统管线质量无优劣之分
// 2 引入几何先验：（shift参数）可合并同一车道线的两个边缘，可按需调整、开启或关闭。
// 3 接口返回变化：同步返回NFA计算的线段置信度，可通过库中的绘制程序按颜色深浅绘制（绘制程序未SIMD优化）；
// 4 串行性能优化：更精细的指令集展开和更高效的混合精度运算；
// 5 线程撕裂优化：引入多线程机制，在多工作线程配置下最多可提升约30%的性能，但不再确保线程安全；
//   *两个管线在多线程下的行为都会略微变化；单线程配置下依然线程安全；
// 6 非SIMD回退修复；但仍不推荐在非AVX2平台使用，性能损失非常大；
// 7 兼容性优化：从兼容CPP20以上改为兼容CPP17以上标准；但依然在MSVC编译器下有问题；
// 8 其他优化。

// 在开启ELSED管线与线程撕裂的情况下，耗时约减少50%，同时检测质量维持不变或更好。
// 总体耗时约为调用opencv接口实现的高斯模糊+canny+houghP的1/3到1/5.

// 参考文献：
//   [1] Lee, J. H., Lee, S., Zhang, G., Lim, J., Chung, W. K., & Suh, I. H.
//       (2014). Outdoor place recognition in urban environments using
//       straight lines. In IEEE International Conference on Robotics and
//       Automation (ICRA) (pp. 5550-5557). Hong Kong, China.
//   [2] von Gioi, R. G., Jakubowicz, J., Morel, J. M., & Randall, G. (2012).
//       LSD: A Line Segment Detector. Image Processing On Line, 2, 35-55.
//   [3] Suárez, I., Buenaposada, J. M., & Baumela, L. (2022).
//       ELSED: Enhanced Line SEgment Drawing. Pattern Recognition,
//       127, 108619.
//   [4] Akinlar, C., & Topal, C. (2011). EDLines: A real-time line segment
//       detector with a false detection control. Pattern Recognition
//       Letters, 32(13), 1633-1642.

#pragma once

#include <opencv2/core.hpp>
#include <vector>

/// @brief 默认管线参数（全部字段为默认值，可直接使用后按需覆盖）
struct fml_params 
{
    int length_threshold = 10;                ///< 最短支撑点数：线段至少包含的连续边缘点数
    float distance_threshold = 1.414213562f;  ///< 点到线距离容差（像素）：支撑点判定与共线合并的距离阈值
    double canny_th1 = 380.0;                 ///< Canny 低阈值（滞后连通种子下限）
    double canny_th2 = 700.0;                 ///< Canny 高阈值（强边缘判定下限），需保证高阈值 >= 低阈值
    float grad_sigma = 2.0f;                  ///< 梯度平滑尺度 σ；0 = 无平滑（3x3 Sobel 路径），默认 2.0
    int do_merge = 2;                         ///< 是否合并共线线段；非 0 开启；可根据后处理算法取舍
    float nfa_eps = 1.0f;                     ///< NFA 显著性阈值；NFA < nfa_eps 的线段被保留
    float nfa_ang_tol_deg = 22.5f;            ///< 支撑点梯度方向对齐角度容差（度）
    int shift_px = -1;                        ///< 边缘对几何先验平移（像素）：NFA 通过后按支撑点
                                              ///< 主导梯度方向（Σdx 符号）分类，向左（Σdx<0）的
                                              ///< 线右移 n，向右（Σdx>0）的左移 n，Σdx==0 不动；
                                              ///< 0 = 关闭；可负（平移方向反转）。亮条带两边缘
                                              ///< 随 +n 外扩、暗条带内收。后续长度/贴边过滤、
                                              ///< 方向统一、merge 均作用于平移后坐标
    int threads = 1;                          ///< 工作线程：1 = 单线程 2 = 双线程 注意：开启超线程后线程不再安全。
};

/// @brief ELSED 管线参数
struct fml_elsed_params 
{
    int length_threshold = 10;                ///< 最短线段长度（像素）：输出过滤用
    float distance_threshold = 1.414213562f;  ///< 共线合并的点线距离容差（像素）：
                                              ///< 合并判据为 2*该值
    int do_merge = 2;                         ///< 共线合并开关：非 0 开启（ELSED 段本身
                                              ///< 较长较整，合并作用比默认管线小）
    int shift_px = -1;                        ///< 边缘对几何先验平移（像素）：按支撑点
                                              ///< Σdx 符号左右平移，语义与 fml_detect 一致；
                                              ///< 0 = 关闭；可负（方向反转）

    // ELSED 算法参数
    double grad_th = 30.0;                    ///< 梯度阈值：L1 幅值低于它的像素不算边缘候选。
                                              ///< 调小收弱边缘/噪点多，调大抗噪
    int anchor_th = 8;                        ///< 锚点阈值：锚点幅值须高出垂直
                                              ///< 方向两邻居这么多；找不到锚点内部自动减半
    int scan_intv = 2;                        ///< 锚点扫描间隔（内部钳位 >=1）
    int min_len = 15;                         ///< 最短拟合长度（钳位 >=2）。
                                              ///< 320x240 下调小（8~12）可检出更短虚线
    double fit_err = 0.2;                     ///< 拟合激活 MSE：窗口均方距离
                                              ///< 误差小于它才认定为直线；调大允许更弯
    double px_dist = 1.2;                     ///< 内点距离容差：已激活段判定
                                              ///< 新像素是否内点；调严会断段（线数变多）
    double validate_th = 0.25;                ///< 段验证角误差（弧度）：重投影处
                                              ///< 插值梯度角与段法向之差大于它记外点，
                                              ///< 内点须多于外点段才保留。最强的质量闸：
                                              ///< 调大出线多杂线多，调小近乎全灭
    int treat_junc = 0;                       ///< 交叉点跳越开关：段端部允许沿
                                              ///< 直线方向跳过 ≤9px 的路口续接（内部判据
                                              ///< 固定：延伸区梯度自相关特征值比>10 且
                                              ///< 特征向量角差<10°）。不希望线跨过横穿
                                              ///< 边时可关闭
    double sigma = 0.9;                       ///< 高斯平滑 σ（5x5 核；<=0 不平滑）。
                                              ///< 与 grad_th 联动：sigma 大则梯度整体变小
    int threads = 1;                          ///< 工作线程：1 = 单线程（默认）；2 = 双线程 注意：开启超线程后线程不再安全
};

/// @brief 在灰度图中检测直线段（默认管线）
/// @param gray   输入图像，必须为 CV_8UC1 单通道灰度图
/// @param p      检测参数
/// @param lines  [out] 检测结果，每条线段为 (x1, y1, x2, y2) 图像像素坐标；
///               覆盖原有内容
/// @param confidences [out] 可选（默认空指针跳过）。非空时清空并写入与 lines
///               一一对应的线段置信度 -log10(NFA)，恒正（检出线段 NFA <
///               nfa_eps）。merge 生成的线段按联合支撑点数重算 NFA，
///               不是两条子线置信度的简单相加
/// @return 检测到的线段数量；输入非法（空图或类型不符）返回 -1
int fml_detect(const cv::Mat& gray, const fml_params& p,
               std::vector<cv::Vec4f>& lines,
               std::vector<float>* confidences = nullptr);

/// @brief 在灰度图中检测直线段（ELSED 管线）
/// @param gray   输入图像，必须为 CV_8UC1 单通道灰度图
/// @param p      检测参数
/// @param lines  [out] 同 fml_detect
/// @param confidences [out] 可选。-log10(NFA)，仅作参考排序用，可为负
///               （该管线的段验证用角度判据，NFA 不做门限）
/// @return 检测到的线段数量；输入非法（空图或类型不符）返回 -1
int fml_detect_elsed(const cv::Mat& gray, const fml_elsed_params& p,
                     std::vector<cv::Vec4f>& lines,
                     std::vector<float>* confidences = nullptr);

/// @brief 将线段绘制到 BGR 图上（就地修改）
///
/// 颜色按置信度从浅绿（低）渐变到深绿（高），按置信度升序绘制，
/// 高置信度线段位于顶层。抗锯齿线段绘制。仅执行绘制，不含任何
/// 窗口显示逻辑。
///
/// @param img        目标 BGR 图（CV_8UC3，就地修改）
/// @param lines      线段集合，(x1,y1,x2,y2) 按检测坐标给出
/// @param confs      置信度，与 lines 一一对应；空向量时全部使用中间色
/// @param scale_x    水平缩放系数：img 像素 x = 线段 x * scale_x
/// @param scale_y    垂直缩放系数：img 像素 y = 线段 y * scale_y
/// @param thickness  线宽（像素）
/// @param cmin       置信度归一化下界（映射为浅绿）
/// @param cmax       置信度归一化上界（映射为深绿）；cmax <= cmin 时
///                   全部使用中间色
void fml_draw_lines(cv::Mat& img, const std::vector<cv::Vec4f>& lines,
                    const std::vector<float>& confs,
                    double scale_x, double scale_y, int thickness,
                    float cmin, float cmax);
