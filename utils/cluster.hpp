// cluster.hpp —— fml 线段结果的车道线聚类接口
//
// 与 fml.hpp 配套使用：fml_detect/fml_detect_elsed 输出的线段按车道线
// 归属聚类为"中心点链 + 抛物线拟合"：
//   fml_cluster_lines  聚类主入口（传入线段 + 图像宽高）
//   fml_draw_clusters  聚类结果绘制（有效簇彩色链，无效簇灰色）
//
// 算法：节点化两段式——先局部成节点，再全局提线
//   节点层（图像纵向等分 bands 条扫描带，逐窗独立重做）：窗口内
//     线段按 x(yc) 排序后多开放组贪心分组为小簇（=节点），方向与
//     位置双约束（含成对发散检验防交叉混组）；节点携带窗口中心
//     测量 (x, yc)、平均方向单位向量、权重 w = Σ len·(0.5+饱和conf/3)
//     与组内平均置信度
//   提取层（全帧节点统一处理，全局视野）：方向加权 MSAC 逐线提取
//     ——随机 3 点采样（fit_quad=0 时 2 点直线）解出模型，双约束内点
//     判定（横向残差 <= x_tol 且节点方向与模型切向夹角达标），按
//     加权 MSAC 分数择优；内点加权最小二乘精修后按 miss_max×窗口高
//     断线切分取最长连续子段，内点硬删除后继续提取下一条
//   输出：内点节点序列（按 y 降序）即车道线中心点链 centers；评分
//     = 0.28·链长 + 0.22·端到端延展 + 0.50·log10(1+平均置信度)；未达
//     min_length 的链 valid=false 且得分打折；线底端播种在顶部
//     top_noise_ratio 噪声带内（整线居顶部噪声带）→ top_noise
//
// 链拟合（描述性，fit_quad 开关）：MSAC（抛物线 3 点 / 直线 2 点
// 采样）剔除外点 + 内点迭代最小一乘（L1/IRLS）精修，模型
// x = a2·y² + a1·y + a0，直车道线的 a2 自然收敛为 ≈0；提取门限
// （x_tol，宽）与链内精修门限（msac_thresh，严）两级分离：
// 提取找全，精修剔噪。
//
// 预处理剔除：与竖直夹角超过 max_psi 的"太水平"线段（横向箭头/
// 斑马线边缘等）在一切聚类之前直接丢弃。置信度为 fml 输出的原始
// 绝对值（-log10 NFA）。
//
// 性能设计：热路径零三角函数（方向一律单位向量，角度门限化为
// 点积门限）、零堆分配（全栈数组 + 复用缓冲 + 输出原位覆盖）、
// 提取层 AVX2 8 宽 FMA、帧级固定种子 LCG 保证同输入同输出。
//
// 曲线参数（a2/a1/a0 与 rms）为中心点链的描述性拟合，不参与任何
// 判据。提取与拟合均为 x(y) 参数化（车道线 x(y) 单值的几何先验），
// 近竖直车道线效果最佳。

#pragma once

#include <opencv2/core.hpp>
#include <vector>

/// @brief 车道线：中心点链及其描述性拟合与评分
struct FmlSegCluster
{
    std::vector<int> indices;           ///< 途经小簇的成员线段下标（去重
                                        ///< 升序，仅供调试/回退渲染，无判据
                                        ///< 意义）
    std::vector<cv::Point2f> centers;   ///< 车道线中心点链（每内点节点
                                        ///< 一个点，自底向上）——即检测结果
    float score = 0.f;                  ///< 簇置信度（降序输出，无上界）
    float a2 = 0.f, a1 = 0.f, a0 = 0.f; ///< x = a2·y² + a1·y + a0
                                        ///< （fit_quad=0 时 a2 恒为 0）
    float t_min = 0.f, t_max = 0.f;     ///< 参数范围（y 区间），按 MSAC
                                        ///< 内点集计算——被剔除的点不计
                                        ///< 入边界，曲线不外推到无支撑区
    float length = 0.f;                 ///< 中心链长度（内点节点数×窗口高，
                                        ///< 像素）
    float rms = 0.f;                    ///< 中心点链拟合的 RMS 残差（像素，
                                        ///< 仅参考）——横向（x 向）残差
    float conf = 0.f;                   ///< 途经节点合并置信度的均值
                                        ///< （原始 -log10 NFA，仅参考）
    bool valid = false;                 ///< 链长达标且非顶部噪声带
    bool top_noise = false;             ///< 线底端播种（线段下端点）在
                                        ///< 顶部 top_noise_ratio 噪声带内
                                        ///< （整线居顶部噪声带），
                                        ///< valid 强制 false
    float theta = 0.f;                  ///< 旋转主轴角（度；0 = 竖直主轴
                                        ///< 即 x(y) 参考系）。fit_rot=1 时
                                        ///< 输出；恒 0° 表示自适应无需旋转
    float ra = 0.f, rb = 0.f, rc = 0.f; ///< 局部抛物线
                                        ///< u = ra·v² + rb·v + rc，
                                        ///< u = x·cosθ + y·sinθ，
                                        ///< v = -x·sinθ + y·cosθ
                                        ///< （θ=0 时与 a2/a1/a0 一致）
};

/// @brief 聚类参数（缺省值按 320x240 工作分辨率标定；其他分辨率按
///        比例设置，图像尺寸经 fml_cluster_lines 的 width/height 传入）
struct fml_cluster_params
{
    int bands = 20;               ///< 扫描带数：图像纵向等分的窗口数，
                                  ///< 窗口高 = 图像高度 / bands（默认 20 带，
                                  ///< 240 高即 12px 一窗）；值越大带越窄
    int miss_max = 3;             ///< 最大容许断线距离（窗口数）：候选线
                                  ///< 内点按 y 升序扫描相邻纵向间隙，超过
                                  ///< miss_max×窗口高 处切断，仅节点数最多
                                  ///< 的连续子段作为该线输出（其余子段节
                                  ///< 点放回剩余池参与后续提取）——决定可
                                  ///< 跨越的最大虚线间隙，防止把跨大间隙
                                  ///< 的虚假长链拟合为一条曲线
    float x_tol = 12.f;           ///< 全局提取的内点横向容差（像素）：
                                  ///< 节点 x 与候选线模型预测 x 的 |Δx|
                                  ///< 上限。设计约定：匹配宜宽于聚类
                                  ///< （x_tol ≥ match_x_tol）——聚类从严
                                  ///< 防混线，提取从宽容忍测量抖动
    float match_x_tol = 20.f;     ///< 窗口内聚类的组内横向容差（像素）：
                                  ///< 同窗线段归入同小簇的间隙门限（含成
                                  ///< 对发散检验 2·match_x_tol）。
                                  ///< 设计约定：从严（≤ x_tol），聚错簇
                                  ///< 的代价远高于丢一个节点
    float ang_tol = 15.f;         ///< 方向夹角容差（度）：窗口内聚类与
                                  ///< 全局 MSAC 内点判定（节点方向与
                                  ///< 模型切向夹角）两处共用
    float max_psi = 60.f;         ///< 太水平剔除阈（度，与竖直夹角）：
                                  ///< |ψ| 超过它的线段在一切聚类之前直接
                                  ///< 丢弃（横向箭头/斑马线边缘等）；
                                  ///< 90 = 不剔除
    float min_length = 50.f;      ///< 有效中心链最短长度（像素，内点
                                  ///< 节点数×窗口高）
    float top_noise_ratio = 1.f / 3.f; ///< 顶部噪声带：线底端播种位置
                                  ///< （线段下端点 y）在画面上方该比例
                                  ///< 高度以内（整线居顶部噪声带）视为
                                  ///< 噪声，仅判性；0 = 关闭
    float msac_thresh = 5.0f;     ///< 链内精修 MSAC 的内点横向残差
                                  ///< 阈值（像素；严于提取门限 x_tol，
                                  ///< 两级分离：提取找全，精修剔噪）
    int msac_iters = 16;          ///< 链内精修 MSAC 采样次数（提取后
                                  ///< 的链内点率高，16 次已充分，一般
                                  ///< 无需调整；全局提取阶段用内部
                                  ///< 常量，不经此参数）
    int fit_quad = 1;             ///< 拟合模型开关：非 0 = 二次曲线 MSAC
                                  ///< + 曲线最小二乘（直线模型见 fit_quad=0）
                                  ///< ；0 = 直线 x = a1·y + a0（输出 a2 恒 0）
    int fit_rot = 1;              ///< 旋转主轴拟合：非 0 = 在竖直主轴 a2/a1/a0
                                  ///< 基础上搜索最优旋转角 θ∈[-60°,60°]，
                                  ///< 在旋转局部系中做抛物线 L1/IRLS 精修，
                                  ///< 输出 theta/ra/rb/rc（贴合急弯/斜向车道线）；
                                  ///< 0 = 仅竖直主轴
    int lane_scorer = 1;          ///< 车道线打分模式（数据训练的可替换加权打分，
                                  ///< 在输出前对 score 重排；默认 1 = M5 乘法核）：
                                  ///<   0 = 保留原始加法 score（传统）
                                  ///<   1 = M5 乘法核（log10(1+Σc)^a·min(lenw,cap)^b，
                                  ///<       可解释/轻量，默认）
                                  ///<   2 = 白盒线性 LR（3 特征：npts/maxYgap/cx_std）
                                  ///<   3 = RBF-SVM（24 特征，效果最强）
                                  ///< 权重由 fml_lane_scorer_init/load 提供；默认用内置
    const struct fml_lane_scorer_weights *lane_scorer_w = nullptr;
                                  ///< 可选：自定义权重（nullptr=用内置默认权重）
                                  ///< 用法：fml_lane_scorer_init(&w, mode, m5) 或
                                  ///< fml_lane_scorer_load(&w, path); 再赋给此字段
};

// 三档打分特征维数（与训练一致的固定布局）
#define FML_LANE_SCORER_NFEAT 24

/// @brief M5 乘法核参数（参数少，直接传；全部带默认值）
/// 公式：score = log10(1+Σc)^a · min(lenw,cap)^b − rank_w·max(rankNorm−1/3, 0)
///       lenw = length / (img_h/2)
struct fml_m5_scorer_params
{
    float a = 0.7f;      ///< 证据指数（Σc = conf·npts 的证据累积，log10 压缩）
    float b = 1.3f;      ///< 长度指数（lenw 无饱和线性映射 H/2→1、H→2）
    float cap = 0.7f;    ///< 长度封顶（护栏/高架多的场景调小到 0.5，开阔高速放宽到 1.0）
    float rank_w = 0.3f; ///< rank 软罚系数（每帧第 4 条起的弱候选减分）
};

/// @brief 车道线打分器权重（可外部加载覆盖；三种模式共用此容器）
///
/// 特征布局（24 维，与训练集 /tmp 导出 v2 一致，索引固定）：
///  0 npts, 1 length, 2 score(旧), 3 conf, 4 sumc=conf·npts, 5 theta,
///  6 |a1|, 7 a2, 8 |a2|·ySpan, 9 |a0-160|, 10 ySpan, 11 cx_mean,
///  12 cx_std, 13 cx_range, 14 xTop, 15 xBot, 16 tilt_dxdy, 17 maxYgap,
///  18 nbGap, 19 ptDensity, 20 relScore, 21 relSumc, 22 rankNorm, 23 tiltAbs
/// 权重文件的用法（顺序随意，缺省保留原值）：
///   mode N / img_h H / b V / gamma G / n_sv K /
///   mu 24 个 / istd 24 个 / w 24 个 / sv 每行 24 个共 K 行 / sc 每行 1 个共 K 个
struct fml_lane_scorer_weights
{
    int mode = 1;                       ///< 对应 lane_scorer 档位（1/2/3）
    float img_h = 240.f;                ///< 检测高（lenw 参考：lenw=length/(img_h/2)）
    float mu[FML_LANE_SCORER_NFEAT] = {0.f};   ///< 特征均值
    float istd[FML_LANE_SCORER_NFEAT] = {0.f}; ///< 特征标准化 1/σ
    float w[FML_LANE_SCORER_NFEAT] = {0.f};    ///< 线性权重
    float b = 0.f;                      ///< 线性/RBF 偏置
    float gamma = 0.f;                  ///< RBF 核参数 γ
    int n_sv = 0;                       ///< RBF 支持向量数
    std::vector<float> sv;              ///< 支持向量（n_sv × NFEAT，行主序）
    std::vector<float> sv_coef;         ///< 支持向量系数（n_sv）
    fml_m5_scorer_params m5;            ///< M5 乘法核参数（mode=1 时生效）
};

/// @brief 用内置权重（当前标注数据集训练）初始化打分器
/// @param w       输出权重容器（覆盖内容）
/// @param mode    打分模式（1=M5, 2=LR, 3=RBF；默认 1）
/// @param m5      M5 参数（mode=1 时生效；全部有默认值，通常不传即可）
/// 用法：
///   fml_lane_scorer_init(w);                              // 默认 M5
///   fml_lane_scorer_init(w, 2);                           // LR
///   fml_lane_scorer_init(w, 1, {0.6f, 1.4f, 0.8f, 0.3f}); // 自定义 M5
void fml_lane_scorer_init(fml_lane_scorer_weights &w, int mode = 1,
                          const fml_m5_scorer_params &m5 = fml_m5_scorer_params{});

/// @brief 从文本权重文件加载打分器（覆盖 w；未命中项保留原值）
/// @param w    输出权重容器
/// @param path 权重文件路径（格式见 fml_lane_scorer_weights 注释）
/// @return 0 成功；非 0 失败（无法打开/解析）
int fml_lane_scorer_load(fml_lane_scorer_weights &w, const char *path);

/// @brief 读取内置默认权重（外部可读取/拷贝/转发）
/// @param mode 打分模式（1=M5, 2=LR, 3=RBF；默认 1）
/// @return 内置默认权重容器的 const 引用（生命周期为进程全程，勿修改）
const fml_lane_scorer_weights &fml_lane_scorer_default(int mode = 1);

/// @brief 窗口切分调试信息（fml_cluster_lines 可选输出，供"簇切分"
/// 调试视图绘制：窗口划分线 + 节点中心点，不连线）
struct fml_cluster_debug
{
    std::vector<float> y_lo;          ///< 各窗口上边界 y（与窗口生成顺序
                                      ///< 一致，自底向上）
    std::vector<float> y_hi;          ///< 各窗口下边界 y
    std::vector<cv::Point2f> centers; ///< 所有节点中心 (x, 窗口中心 y)
    std::vector<char> matched;        ///< 与 centers 一一对应：
                                      ///< 1 = 已归属某候选线，0 = 未成线
};

/// @brief 将 fml 检出的线段按车道线归属聚类
/// @param lines     fml 输出线段 (x1,y1,x2,y2)，图像像素坐标
/// @param confs     fml 输出置信度（与 lines 一一对应；可含负值）
/// @param width     检测图像宽（像素；lines 坐标所在的坐标系）
/// @param height    检测图像高（像素；bands/top_noise_ratio 等按此
///                  高度换算，与 fml 检测分辨率一致）
/// @param clusters  [out] 车道线结果，按 score 降序；centers 为中心点链
/// @param p         聚类参数
/// @param dbg       [out] 可选（默认空指针跳过），窗口切分调试信息
/// @return 有效车道线数量（valid=true 的簇数）
int fml_cluster_lines(const std::vector<cv::Vec4f> &lines,
                      const std::vector<float> &confs,
                      int width, int height,
                      std::vector<FmlSegCluster> &clusters,
                      const fml_cluster_params &p = fml_cluster_params{},
                      fml_cluster_debug *dbg = nullptr);

/// @brief fml_draw_clusters 绘制模式宏（传入 mode 参数选择绘制内容）
#define FML_CLUSTER_DRAW_CHAINS 0 ///< 聚类视图：中心点链（有效绿/无效灰）
                                  ///< + 蓝色 R 标签（排名 + 得分）
#define FML_CLUSTER_DRAW_FIT 1    ///< 拟合视图：拟合曲线（fit_rot 开启时
                                  ///< 为旋转主轴圆弧，否则竖直主轴抛物线/
                                  ///< 直线）+ 中心点 + 蓝色 R 标签
#define FML_CLUSTER_DRAW_SPLIT 2  ///< 簇切分视图：窗口划分细线 + 节点
                                  ///< 中心点（黄=已归属候选线，红=未
                                  ///< 成线）；需要 dbg，基底（fml 线段
                                  ///< 置信度图）由调用方先行绘制

/// @brief 将聚类结果按指定模式绘制到 BGR 图上（就地修改）
///
/// 绘制内容由 mode 宏（FML_CLUSTER_DRAW_*）选择，全部视图的绘制
/// 流程收拢于此（与 fml_draw_lines 同风格：只画不弹窗）：
///   FML_CLUSTER_DRAW_CHAINS  有效簇纯绿中心点链（粗线）+ 蓝色
///                            R 标签（排名 + 得分）；无效簇灰色细链
///   FML_CLUSTER_DRAW_FIT     有效簇拟合曲线（粗线）+ 中心点小点 +
///                            蓝色 R 标签；无效簇灰色
///   FML_CLUSTER_DRAW_SPLIT   窗口划分灰色细线 + 节点中心点
///                            （黄=已归属，红=未成线；需要 dbg）
///
/// @param img        目标 BGR 图（CV_8UC3，就地修改）
/// @param clusters   fml_cluster_lines 的输出
/// @param scale_x    水平缩放系数：img 像素 x = 坐标 x * scale_x
/// @param scale_y    垂直缩放系数：img 像素 y = 坐标 y * scale_y
/// @param thickness  有效簇线宽（像素）；无效簇自动减半
/// @param mode       绘制模式（FML_CLUSTER_DRAW_*）
/// @param dbg        SPLIT 模式必需（其余模式可传空指针）
void fml_draw_clusters(cv::Mat &img,
                       const std::vector<FmlSegCluster> &clusters,
                       double scale_x, double scale_y, int thickness,
                       int mode, const fml_cluster_debug *dbg = nullptr);
