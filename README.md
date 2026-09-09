# FML3.1_FOR_X86_64_LTS

```
FML3.1_FOR_X86_64_LTS
├─code 源代码
│  ├─include
│  └─source
├─img
├─release 二进制预编译包，推荐使用此包
│  ├─include
│  └─lib
└─utils 训练打分器用的工具
```

## v3.1 LTS 发布说明

### 更新说明
- 第一个 LTS 版本，接口稳定，后期可能发布新的 libfml.a，但不会大范围更改功能和接口。
- cluster 模块变为：网格聚类划分 + 二次 MSAC 算法提取 + 打分器 + 迭代最小一乘角度恢复。具体内容见头文件说明。

### 注意事项
1. cluster 模块为附属模块，通过 `cluster.hpp` 引入，使用（尤其是打分器部分）较为复杂，打分器推荐使用默认配置，其他参数按自己的输入分辨率调试。
2. 推荐使用预编译包。接口说明一定参照头文件，可参考 `example.cpp` 的配置（release 和 code 里的一样）。
3. 个人作者文档编撰可能不清晰，算法较复杂，版本之间文档管理混乱，请见谅。源码已全量开源，欢迎改进。如有问题 QQ 联系 439887968。

### 发布说明
libfml 的两个组成部分及其配套工具链：
1. **fml** —— 直线段检测器（双管线）
2. **cluster** —— 线段按车道线归属聚类 + 数据训练打分器
3. **utils** —— 数据标注/训练工具链（extract_json / annotate.py / train.py）

---

## 0. 总体结构

| 文件 | 说明 |
|---|---|
| `code/include/fml.hpp` | 检测器接口（fml_detect / fml_detect_elsed / fml_draw_lines） |
| `code/include/cluster.hpp` | 聚类接口（fml_cluster_lines / fml_draw_clusters / 打分器） |
| `code/source/fml.cpp` | 检测器实现（C++17，AVX2 加速，有标量回退） |
| `code/source/cluster.cpp` | 聚类实现（节点化两段式 MSAC；打分器含 AVX2 分支） |
| `code/source/scorer_weights_data.h` | 内置三档打分权重（自动生成） |
| `dist/libfml.a` | 静态库交付物（fml.o + cluster.o） |
| `dist/fml.hpp` / `cluster.hpp` | 与 include 同步的头文件 |

**依赖**：OpenCV（core/imgproc/imgcodecs；GUI 工具另需 highgui）。

推荐编译标志：
```
-O3 -march=haswell -mtune=haswell -ffast-math -std=c++17
```
非 AVX2 平台可编译（代码含标量回退），但性能损失大，不推荐。

---

## 1. 直线段检测（fml.hpp）

两个管线，选其一：

```cpp
int fml_detect(const cv::Mat& gray, const fml_params& p,
               std::vector<cv::Vec4f>& lines,
               std::vector<float>* confidences = nullptr);

int fml_detect_elsed(const cv::Mat& gray, const fml_elsed_params& p,
                     std::vector<cv::Vec4f>& lines,
                     std::vector<float>* confidences = nullptr);
```

- 输入 `gray` 必须为 `CV_8UC1`。`lines` 输出 `(x1,y1,x2,y2)` 像素坐标。
- 返回值 = 线段数；空图/类型不符返回 -1。

**管线对比：**

| 管线 | 流程 | 置信度 |
|---|---|---|
| `fml_detect` | Sobel 梯度 → Canny → 边缘链 → 最小二乘拟合 → NFA 显著性验证 → 几何先验平移 → 共线合并 | 恒正（-log10 NFA） |
| `fml_detect_elsed` | 梯度锚点扫描 → 边缘绘制游走 → 段角度验证 → 同一后段 | 仅排序参考，可为负（验证用角度判据，NFA 不做门限） |

- 推荐输入分辨率 **320x240**。
- `threads=2` 时提速约 30% 但**不再线程安全**（仅允许单应用线程调用），`threads=1` 为线程安全默认。

**绘制：**
```cpp
void fml_draw_lines(cv::Mat& img, const std::vector<cv::Vec4f>& lines,
                    const std::vector<float>& confs,
                    double scale_x, double scale_y, int thickness,
                    float cmin, float cmax);
```
按置信度从浅绿到深绿渐变绘制（cmax<=cmin 时全用中间色）。

---

## 2. 车道线聚类（cluster.hpp）

```cpp
int fml_cluster_lines(const std::vector<cv::Vec4f>& lines,
                      const std::vector<float>& confs,
                      int width, int height,
                      std::vector<FmlSegCluster>& clusters,
                      const fml_cluster_params& p = fml_cluster_params{},
                      fml_cluster_debug* dbg = nullptr);
```

- 输入为 fml 检测结果（lines + 一一对应 confs）及检测分辨率。
- 输出 `clusters` 按 score 降序；返回有效车道线数（valid=true 的簇数）。

**算法概览（节点化两段式）：**
1. **节点层**：纵向 `bands` 条扫描带内独立重做小簇聚类（=节点），方向与位置双约束（含成对发散检验）。
2. **提取层**：全帧节点方向加权 MSAC 逐线提取（3 点/2 点采样），内点加权最小二乘精修，按 `miss_max` 断线切分取最长连续子段，内点硬删除后继续提取下一条。
3. **输出**：中心点链 centers + 描述性抛物线拟合 `x = a2·y² + a1·y + a0` + 评分 + 有效性标志。

**输出结构 FmlSegCluster 关键字段：**

| 字段 | 说明 |
|---|---|
| `centers` | 中心点链（cv::Point2f，自底向上，即检测结果） |
| `score` | 簇置信度（降序输出，经打分器重排后写回） |
| `a2/a1/a0` | x(y) 抛物线参数（fit_quad=0 时 a2 恒 0） |
| `theta` | 旋转主轴角（度；fit_rot=1 时输出，0°=无需旋转） |
| `ra/rb/rc` | 旋转局部系抛物线 u = ra·v² + rb·v + rc |
| `length` | 中心链长度（节点数×窗口高） |
| `rms` | 拟合 RMS 残差（横向 x 向，仅参考） |
| `conf` | 途经节点合并置信度均值 |
| `valid` | 链长达标且非顶部噪声带 |
| `top_noise` | 线底端播种在顶部噪声带内（valid 强制 false） |
| `t_min/t_max` | 参数范围（y 区间，按 MSAC 内点集计算） |

**fml_cluster_params 关键参数（默认按 320x240 标定）：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `bands` | 20 | 扫描带数（图高/bands=窗口高） |
| `miss_max` | 3 | 最大断线间距（窗口数），超过处切断——决定可跨越的最大虚线间隙 |
| `x_tol` | 12.f | 全局提取内点横向容差（像素），匹配宜宽于聚类 |
| `match_x_tol` | 20.f | 窗口内聚类的组内横向容差（从严，聚错簇代价高） |
| `ang_tol` | 15.f | 方向夹角容差（度） |
| `max_psi` | 60.f | 太水平剔除阈（度；90=不剔除） |
| `min_length` | 50.f | 有效中心链最短长度（像素） |
| `top_noise_ratio` | 1/3 | 顶部噪声带比例（0=关闭） |
| `msac_thresh` | 5.0f | 链内精修 MSAC 内点横向残差阈值（严于 x_tol） |
| `msac_iters` | 16 | 链内精修采样次数 |
| `fit_quad` | 1 | 1=二次曲线拟合，0=直线（a2 恒 0） |
| `fit_rot` | 1 | 1=旋转主轴 L1 精修（theta/ra/rb/rc 有效） |
| `lane_scorer` | 1 | 打分模式：0=原始加法 1=M5 2=LR 3=RBF（见下） |
| `lane_scorer_w` | nullptr | 自定义权重（nullptr=内置默认） |

**调试输出 fml_cluster_debug（dbg 非空时收集）：**

| 字段 | 说明 |
|---|---|
| `y_lo / y_hi` | 各窗口上下边界 |
| `centers` | 所有节点中心 |
| `matched` | 节点是否归属候选线（1/0） |

**绘制：**
```cpp
void fml_draw_clusters(cv::Mat& img, const std::vector<FmlSegCluster>& clusters,
                       double scale_x, double scale_y, int thickness,
                       int mode, const fml_cluster_debug* dbg = nullptr);
```
- `mode`: `FML_CLUSTER_DRAW_CHAINS=0`（中心点链）/ `FML_CLUSTER_DRAW_FIT=1`（拟合曲线）/ `FML_CLUSTER_DRAW_SPLIT=2`（窗口切分，需 dbg）

---

## 3. 车道线打分器（lane_scorer 0/1/2/3）

聚类输出后对 score 重排的加权打分，数据训练、可替换：

| 模式 | 名称 | 说明 |
|---|---|---|
| 0 | 原始加法 | 保留原始加法 score（传统加减分，可解释） |
| 1 | M5 乘法核（默认） | log10(1+Σc)^a · min(lenw,cap)^b − rank_w·max(rankNorm−1/3, 0)，可解释/轻量 |
| 2 | 白盒线性 LR | 3 有效特征（npts / maxYgap / cx_std），稀疏 24 维展开，同样轻量且可解释 |
| 3 | RBF-SVM-24（效果最强） | 92 个支持向量 × 24 维特征，每簇约 320 ns（-ffast-math + AVX2 距离），端到端约 +5µs/帧 |

穿透方式：cluster 提取阶段按固定规则取分，有效簇全簇参与；打分器仅重排 score，不影响 valid 判定以外的聚类行为。

**权重接口：**
```cpp
void fml_lane_scorer_init(fml_lane_scorer_weights& w,
                          int mode = 1,
                          const fml_m5_scorer_params& m5 = {});
    // 用内置权重初始化（mode=1 M5 / 2 LR / 3 RBF）
int  fml_lane_scorer_load(fml_lane_scorer_weights& w, const char* path);
    // 从权重文本文件加载（未命中项保留原值）；0=成功
const fml_lane_scorer_weights& fml_lane_scorer_default(int mode = 1);
    // 读取内置默认权重（进程全程有效，勿修改）
```

**用法：**
```cpp
fml_cluster_params p;
p.lane_scorer = 3;
p.lane_scorer_w = &w;              // w 由 init/load 填充
fml_cluster_lines(lines, confs, W, H, clusters, p);
// lane_scorer_w=nullptr 时自动用 fml_lane_scorer_default(mode)
```

**权重文件格式（文本，键值对，顺序随意）：**
```
mode 1|2|3
img_h 240.0           # lenw 参考尺度
m5_a / m5_b / m5_cap / m5_rank_w   # 仅 mode=1
mu  24 个浮点        # 特征均值（标准化用）
istd 24 个浮点        # 特征 1/σ
w 24 个浮点           # 线性权重（mode=2）
b / gamma            # mode=2: b；mode=3: b + gamma
nsv K                # mode=3 支持向量数
sv 每行 24 个，共 K 行
svc 一行 K 个         # 支持向量系数
```

---

## 4. 24 维特征布局（与训练/标注契约一致，索引固定）

| 索引 | 特征 | 索引 | 特征 |
|---|---|---|---|
| 0 | npts | 12 | cx_std |
| 1 | length | 13 | cx_range |
| 2 | score(旧) | 14 | xTop |
| 3 | conf | 15 | xBot |
| 4 | sumc=conf·npts | 16 | tilt_dxdy |
| 5 | theta | 17 | maxYgap |
| 6 | abs(a1) | 18 | nbGap |
| 7 | a2 | 19 | ptDensity |
| 8 | abs(a2)·ySpan | 20 | relScore |
| 9 | abs(a0-160) | 21 | relSumc |
| 10 | ySpan | 22 | rankNorm |
| 11 | cx_mean | 23 | tiltAbs |

（rank 相关量由当前排序位次/帧内簇数算出，帧内相对特征。）

---

## 5. 数据工具链（dist/utils）

三件套 + 静态库，跨平台（extract_json 为 Linux 预编译，其余 pure Python 跨平台）：

1. **extract_json <图像目录> <输出.json>**
   - C++ 批处理：逐图 fml 检测 + 聚类，每簇导出 centers、24 维 features、score/m5 参考分、valid/suggest、拟合参数。
   - JSON 内嵌全部特征 → 下游纯 Python 标注/训练无需 C++。

2. **annotate.py <图像目录> <检测.json> [输出.json]**
   - 标注 GUI（opencv-python）：三档 绿(1)=是 / 灰(0)=否 / 蓝(2)=难例。
   - a/d 翻图自动保存；s 手动保存；q 退出保存；c 切换显示。
   - 再次打开自动继承旧标注；默认回写输入文件（建议副本标注）。

3. **train.py <标注.json> --mode m5|lr|rbf [--out weights.txt]**
   - 网格搜索（M5）或 sklearn（LR / RBF-SVM）训练，输出权重文本，用 `fml_lane_scorer_load` 加载。样本需 >=20 条。

**典型全流程：**
```bash
./extract_json ./images ./det.json          # 1. Linux 批量导出
python annotate.py ./images ./det.json      # 2. 任意平台标注
python train.py ./det.json --mode rbf       # 3. 训练权重
# 4. 上位机 fml_lane_scorer_load 加载重训权重覆盖内置
```

---

## 6. 构建

```bash
# 静态库（交付物 dist/libfml.a）
bash code/build_lib.sh

# 演示应用（图形界面调参，build/app）
bash code/build.sh

# VM 内基准/工具（utils/bench）
bash utils/bench/run.sh

# 聚类基准
bash utils/cluster_tool/dev/build_bench.sh [主文件.cpp]
```

---

## 7. 已知约束与注意

- `threads=2` 时不保证线程安全，仅单应用线程调用；threads 缺省 1。
- MSVC 部分指令集不兼容，Windows 请用 MinGW。
- ARM 平台（树莓派/rk3588）需 NEON 重写，当前仅 x86_64 AVX2。
- 曲线参数为描述性拟合（x(y) 单值的几何先验），近竖直车道线最佳；车道线 x(y) 多值时 a2 拟合退化，仅作参考不参与判据。
- 打分器默认权重在 `scorer_weights_data.h` 中可由训练工具链重训覆盖。

---

## 历史版本 README

### v3.0 beta 更新说明：引入 cluster 模块
引入与 fml 检测器标配的后处理模块。此版本的后处理模块采用启发式生长思路。

### v2.0 发布说明：4000+ 行源码手动优化的检测器
1. 接口改名为 fml，引入更多功能；
2. 引入 ELSED 检测管线，速度更快但检测行为与经典管线略有不同，通过 ELSED 接口调用；引入标准的绘制程序；
   - *ELSED 与传统管线质量无优劣之分，参数说明见 hpp 注释；example 默认调用 ELSED 无线程撕裂。*
3. 引入几何先验：（shift 参数）可合并同一车道线的两个边缘，可按需调整、开启或关闭。
4. 接口返回变化：同步返回 NFA 计算的线段置信度，可通过库中的绘制程序按颜色深浅绘制（绘制程序未 SIMD 优化）；
5. 串行性能优化：更精细的指令集展开和更高效的混合精度运算；
6. 线程撕裂优化：引入多线程机制，在多工作线程配置下最多可提升约 30% 的性能，但不再确保线程安全；
   - *两个管线在多线程下的行为都会略微变化；单线程配置下依然线程安全；*
7. 非 SIMD 回退修复；但仍不推荐在非 AVX2 平台使用，性能损失非常大；
8. 兼容性优化：从兼容 CPP20 以上改为兼容 CPP17 以上标准；但依然在 MSVC 编译器下有问题；
9. 其他优化。

推荐使用预编译包。编译器的影响还是比较大的。基准测试在 i9-13900HX 单核完成，分辨率 320*240，见 benchmark_i913900HX.png。

### v1.0 beta 发布说明

本算法在电脑上能跑每秒 3000 帧（约 0.33ms 一帧）。速度完全不用担心。仅依赖 opencv（opencv 仅仅为了接口传 cv::Mat 方便，如果有需要可私信我，我可以提供一个不依赖 opencv 的版本）。预编译版本仅支持 linux。cpp 版本在 Windows 端支持 MinGW 编译器，linux 端支持 GCC 以及 Clang 编译器（其他大概率支持，没试过）。MSVC 编译器有指令不支持，可能需要自行更改部分代码。欢迎有人自行优化然后在速度上刷榜。

**程序概述**

针对 AVX2 指令集（i5-4200U）优化的直线检测器，输入输出见 .hpp 文件，编译依赖见 .hpp 文件。这个程序针对 x86/64 芯片以及 AVX2 做过指令集级别的速度优化，速度快不代表质量低（参数合适的情况下）。针对 5G 的同学，很抱歉我没有准备适配树莓派的优化版本，大家可以看我之前开源的代码，走的是无 NMS 的算法，那个程序跑 70 帧还是没有问题的（虽然现在看优化的可能也不是特别到位）。

example.cpp 仅供调用参考。

这个检测器输出的是比较碎的直线段，并不是最终的完整直线，因此使用的时候还需要自行进行后处理，这个就并非通用直线检测器的任务范围了。另外就是这个检测器对光照，噪声等不敏感，仅需要转灰度即可，不是特别需要提前高斯模糊或者光照平衡（检测器内嵌了一个高斯模糊，也就是 σ 参数。这块已经和 sobel 算子做了核融合了，比 opencv 实现性能好的多）。参数说明见最后。这个实现大致是结合了 LSD 的 NFA 思想，EDlines 的短链游走思想，以及 FLD 的从 Canny 提取廉价锚点的优化思路。

性能绝对靠谱（我相信直线质量比市面上能找到的所有非深度学习的直线检测算法都靠谱，速度上爆杀所有开源实现），放心用即可。极客小组内部使用就好了，不要外传。

**参数说明如下**

推荐自行预习相关理论。推荐输入 320*240（可自行根据需要更改），在 hpp 中也有简要说明。

```cpp
struct fld_params {
    int length_threshold = 10;                ///< 最短支撑点数：线段至少包含的连续边缘点数
    float distance_threshold = 1.414213562f;  ///< 点到线距离容差（像素）：支撑点判定与共线合并的距离阈值
    double canny_th1 = 380.0;                 ///< Canny 低阈值（滞后连通种子下限）
    double canny_th2 = 700.0;                 ///< Canny 高阈值（强边缘判定下限）具体canny的调参方法可以自行查阅，保证高阈值>=低阈值即可。
    float grad_sigma = 2.0f;                  ///< 梯度平滑尺度 σ；0 = 无平滑（3x3 Sobel 路径），默认 2.0。这里的高斯模糊最终会融合进sobel的权重中，后者是我手动展开SIMD计算的分离卷积，所以推荐直接调这个继承的高斯权重而非自己额外加。
    int do_merge = 2;                         ///< 是否合并共线线段；非 0 开启；个人建议根据自己的后处理算法取舍（就是你是自己处理原始的检测结果，还是直接合并近似拿到一个比较干净的结果）。
    float nfa_eps = 1.0f;                     ///< NFA 显著性阈值；NFA < nfa_eps 的线段被保留
    float nfa_ang_tol_deg = 22.5f;            ///< 支撑点梯度方向对齐角度容差（度）这两项NFA的参数可调节过滤杂线。一般来说最后调节这两个参数
};
```

**算法流程概述**

函数接受灰度图像后：

1. 通过可分离的梯度算子（支持高斯平滑参数）计算每个像素的梯度幅值和方向，随后 canny 生成二值边缘图并保存每个边缘像素的梯度方向（供最后的 NFA 使用）。
2. 在边缘图上逐行扫描，以边缘像素为种子，沿 8 连通方向进行链式追踪，并施加方向一致性约束以避免回溯，同时边走边将已访问像素清零。每条边缘链被收集为一系列有序点集。
3. 从链起点开始，用最小二乘法拟合直线，逐步扩展支撑点，检查点到直线距离是否超过阈值；若超差则回退并终止扩展。
4. 对拟合出的候选线段需通过 NFA 检验：统计支撑点中梯度方向与线段法向对齐的像素数，计算其尾部概率，只有当 NFA 小于设定阈值（即显著性足够高）时才保留该线段。
5. 提取出的线段进一步进行方向归一化——沿线采样像素并比较法向两侧灰度累计和，使线段统一为"右侧更亮"的朝向。若启用合并模式，则对线段集合进行两两合并，合并条件包括中点距离、角度差和端点到对方直线的距离，合并后重新拟合线段并再次归一化方向。

最终输出为浮点型端点坐标的线段列表。

---

作者 player404. QQ 439887968
