// main.cpp
// fml 直线检测 + 车道线聚类调用示例（ELSED 管线）。
//
// 流程：cv::imread 读图（灰度）→ 缩放 320x240 → fml_detect_elsed 检测
//       → fml_cluster_lines 将线段按车道线聚类
//       → 车道线以中心点链折线显示（每匹配窗口一个中心点相连；
//         有效簇按置信度排名取醒目颜色，噪声灰色细线）
//       → imshow 显示。
//
// 显示模式按 c 键轮换：聚类视图（中心点链，默认）/ 置信度视图
// （fml_draw_lines，浅绿=低 → 深绿=高）/ 簇切分视图（置信度基底 +
// 窗口划分细线 + 小簇中心点：黄=已关联，红=升格）/ 拟合视图（原图
// 基底 + 每条中心链的拟合曲线：有效簇纯绿粗线、无效簇灰色，中心
// 点按排名取色）。窗口标题显示线段数与有效簇数。
//
// 进度条（trackbar）实时调参：默认值与 fml.hpp 中 fml_elsed_params 一致；
// 拖动即用新参数对当前图重新检测。默认管线接口（fml_detect +
// fml_params）见 fml.hpp，本示例默认使用 ELSED 管线。
//
// 按键：d / → 下一张   a / ← 上一张   c 切换显示模式   ESC / q 退出
//
// 独立聚类参数窗口 "Cluster Params"，实时调节聚类阈值与滤波系数

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "cluster.hpp"
#include "fml.hpp"

namespace
{

const int DS_W = 320, DS_H = 240;  // 检测分辨率
const int LINE_WIDTH = 12;         // 显示线宽
const char *WIN = "fml";           // 结果显示窗口
const char *WIN_TB = "fml-params"; // 调参窗口（检测参数）
const char *WIN_CLU = "Cluster Params"; // 聚类参数窗口
enum
{
    KeyLeft = 81,
    KeyRight = 83
}; // 方向键键值（waitKey 返回）

// ---- 检测参数轨道 ----
int tb_length = 10;    // length_threshold        1..100
int tb_dist10 = 14;    // distance_threshold x10  0..50
int tb_merge = 1;      // do_merge                0..1
int tb_shift20 = 19;   // shift_px + 20（轨道 0..40，实际 -20..+20，默认19）
int tb_downsample = 1; // 降采样比例 1..4（1=不降采样，2=160x120，以此类推）
// ELSED 参数（double 除法保精度，如 25/100.0 == 0.25，默认位与结构体一致）
int tb_e_grad = 30;    // grad_th      0..300
int tb_e_anchor = 8;   // anchor_th    0..64
int tb_e_scan = 2;     // scan_intv    1..4
int tb_e_minlen = 15;  // min_len      2..60
int tb_e_fit100 = 20;  // fit_err x100 0..100（默认 20 → 0.2）
int tb_e_pdist10 = 12; // px_dist x10  0..50（默认 12 → 1.2）
int tb_e_val100 = 25;  // validate_th x100 0..100（默认 25 → 0.25）
int tb_e_junc = 0;     // treat_junc   0/1
int tb_e_sigma10 = 9;  // sigma x10    0..30（默认 9 → 0.9）
int tb_threads = 0;    // threads 开关 0/1（1=双线程帧内加速，同物理核超线程）

// ---- 聚类参数轨道 ----
int tb_bands = 20;         // bands          1~40（扫描带数，图高/bands
                           // = 窗口高；默认 20 带，值越大带越窄）
int tb_x_tol = 20;         // x_tol         1~80 (像素，全局提取容差)
int tb_match_x = 20;       // match_x       1~80 (像素，同窗线段归组间隙)
int tb_ang_tol = 15;       // ang_tol       1~45 (度)
int tb_miss_max = 3;       // miss_max      0~10（最大容许断线距离，窗口数）
int tb_min_length = 30;    // min_length    0~200
int tb_top_noise = 33;     // top_noise     0~100（%画面高，0=关闭；
                           // 默认 33 = 上 1/3 播种的簇视为噪声）
int tb_max_psi = 60;       // max_psi       0~90（度，太水平剔除阈，
                           // 90 = 不剔除）
int tb_msac_th = 80;       // msacth x10    0~100（MSAC 内点阈值，
                           // 80 = 8.0px）
int tb_fit_quad = 1;       // fit_quad      0/1（1=二次曲线拟合，0=直线）
int tb_fit_rot = 1;       // fit_rot       0/1（1=旋转主轴 L1 拟合）
int tb_lane_scorer = 1;   // lane_scorer   0=原score 1=M5(默认) 2=LR 3=RBF
                          //               （数据训练打分器；FML_PREVIEW
                          //               默认 M5 核）

bool g_dirty = true;   // 参数或图片变化，需要重新检测
int g_view = 0;        // 显示模式：0 = 聚类视图，1 = 置信度视图

// 聚类显示调色板（按簇置信度排名取色，20 色循环）
const cv::Scalar CLUSTER_COLORS[20] = {
    {0, 0, 255},
    {0, 255, 0},
    {255, 0, 0},
    {0, 255, 255},
    {255, 255, 0},
    {255, 0, 255},
    {0, 165, 255},
    {255, 128, 0},
    {80, 220, 255},
    {200, 0, 200},
    {0, 90, 255},
    {180, 255, 255},
    {60, 60, 255},
    {255, 60, 60},
    {60, 255, 60},
    {90, 200, 130},
    {130, 90, 30},
    {30, 130, 200},
    {200, 130, 200},
    {240, 180, 120},
};
const cv::Scalar NOISE_COLOR = {110, 110, 110};

void onTrackbar(int, void *)
{
    g_dirty = true;
}

// 从进度条当前值组装检测参数（默认值即 fml_elsed_params 默认值）
fml_elsed_params paramsFromTrackbars()
{
    fml_elsed_params p;
    p.length_threshold = std::max(1, tb_length);
    p.distance_threshold = tb_dist10 / 10.0f;
    p.do_merge = tb_merge;
    p.shift_px = tb_shift20 - 20;
    // ELSED 参数：double 除法保精度（如 25/100.0 == 0.25）
    p.grad_th = (double)std::max(tb_e_grad, 0);
    p.anchor_th = std::max(tb_e_anchor, 0);
    p.scan_intv = std::max(tb_e_scan, 1);
    p.min_len = std::max(tb_e_minlen, 2);
    p.fit_err = std::max(tb_e_fit100, 0) / 100.0;
    p.px_dist = std::max(tb_e_pdist10, 0) / 10.0;
    p.validate_th = std::max(tb_e_val100, 0) / 100.0;
    p.treat_junc = std::min(std::max(tb_e_junc, 0), 1);
    p.sigma = std::max(tb_e_sigma10, 0) / 10.0;
    return p;
}

// 从进度条组装聚类参数
fml_cluster_params clusterParamsFromTrackbars()
{
    fml_cluster_params p;
    p.bands = std::max(1, tb_bands);
    p.x_tol = std::max(1, tb_x_tol);
    p.match_x_tol = std::max(1, tb_match_x);
    p.ang_tol = std::max(1, tb_ang_tol);
    p.miss_max = std::max(0, tb_miss_max);
    p.min_length = (float)tb_min_length;
    p.max_psi = (float)std::max(0, std::min(90, tb_max_psi));
    p.msac_thresh = std::max(0, std::min(100, tb_msac_th)) / 10.0f;
    p.fit_quad = tb_fit_quad ? 1 : 0;
    p.fit_rot = tb_fit_rot ? 1 : 0;
    p.lane_scorer = tb_lane_scorer; // 0=原score 1=M5 2=LR 3=RBF（数据训练打分器）
    p.top_noise_ratio = std::max(0, std::min(100, tb_top_noise)) / 100.0f;
    return p;
}

// 自然排序：把数字段按数值比较，保证 2.jpg 排在 10.jpg 前。
bool naturalLess(const std::string &a, const std::string &b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
        if (std::isdigit(a[i]) && std::isdigit(b[j]))
        {
            size_t si = i, sj = j;
            while (i < a.size() && std::isdigit(a[i]))
                ++i;
            while (j < b.size() && std::isdigit(b[j]))
                ++j;
            std::string na = a.substr(si, i - si);
            std::string nb = b.substr(sj, j - sj);
            na.erase(0, na.find_first_not_of('0'));
            nb.erase(0, nb.find_first_not_of('0'));
            if (na.size() != nb.size())
                return na.size() < nb.size();
            if (na != nb)
                return na < nb;
        }
        else
        {
            if (a[i] != b[j])
                return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return a.size() < b.size();
}

bool isImageExt(const std::string &name)
{
    const char *exts[] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    std::string lower = name;
    for (auto &ch : lower)
        ch = (char)std::tolower(ch);
    for (const char *e : exts)
        if (lower.size() >= strlen(e) && lower.compare(lower.size() - strlen(e), strlen(e), e) == 0)
            return true;
    return false;
}

// 读第 idx 张图 → 检测 → 聚类 → 按 view 绘制，返回标注图（不弹窗）
cv::Mat annotate(const std::vector<std::string> &files, int idx,
                 const fml_elsed_params &p, const fml_cluster_params &cp,
                 std::vector<cv::Vec4f> &lines, std::vector<float> &confs,
                 int view, int *out_nl = nullptr, int *out_nvalid = nullptr,
                 double *out_ms = nullptr, int *out_w = nullptr,
                 int *out_h = nullptr)
{
    // 读图：彩色原图用于显示；灰度缩放到检测分辨率
    cv::Mat img = cv::imread(files[idx], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::fprintf(stderr, "[fml] 读图失败: %s\n", files[idx].c_str());
        return {};
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 根据降采样比例计算实际检测分辨率（默认 1 = 320x240）
    int ds = std::max(1, tb_downsample);
    int det_w = DS_W / ds;
    int det_h = DS_H / ds;
    cv::resize(gray, gray, cv::Size(det_w, det_h), 0, 0, cv::INTER_AREA);

    // 处理总耗时（检测 + 聚类 + 绘制）
    auto t0 = std::chrono::steady_clock::now();
    int nl = 0;
    try
    {
        nl = fml_detect_elsed(gray, p, lines, &confs);
    }
    catch (const cv::Exception &e)
    {
        std::fprintf(stderr, "[fml] 检测阶段 OpenCV 异常: %s\n", e.what());
        lines.clear();
        confs.clear();
        return {};
    }
    // 帧内置信度范围（着色归一化用）
    float cmin = 0.0f, cmax = 0.0f;
    if (!confs.empty())
    {
        cmin = cmax = confs[0];
        for (float g : confs)
        {
            if (g < cmin)
                cmin = g;
            if (g > cmax)
                cmax = g;
        }
    }

    // 在原始分辨率彩色图上画线（线段坐标 × 缩放系数）
    double sx = (double)img.cols / det_w;
    double sy = (double)img.rows / det_h;

    // 车道线聚类（传入聚类参数）；dbg 收集窗口划分与小簇中心供切分视图
    std::vector<FmlSegCluster> clusters;
    fml_cluster_debug dbg;
    int nvalid = 0;
    try
    {
        nvalid = fml_cluster_lines(lines, confs, det_w, det_h,
                                       clusters, cp, &dbg);
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "[fml] 聚类阶段异常: %s\n", e.what());
    }
    // 纯计算耗时（检测 + 聚类；绘制不计入）
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();

    if (view == 0)
    {
        // 聚类视图：车道线中心点链折线（有效簇纯绿粗线 + 蓝色 R 标签，
        // 噪声/无效簇灰色细线）
        fml_draw_clusters(img, clusters, sx, sy, LINE_WIDTH,
                          FML_CLUSTER_DRAW_CHAINS);
    }
    else if (view == 3)
    {
        // 拟合视图：拟合曲线（粗线）+ 中心点小点 + 蓝色 R 标签，
        // 原图为基底
        fml_draw_clusters(img, clusters, sx, sy, LINE_WIDTH,
                          FML_CLUSTER_DRAW_FIT);
    }
    else
    {
        // 置信度视图 + 簇切分视图：fml 线段基底（浅绿=低 → 深绿=高）
        fml_draw_lines(img, lines, confs, sx, sy, LINE_WIDTH, cmin, cmax);
        if (view == 2)
        {
            // 簇切分视图：窗口划分细线 + 小簇中心点（黄=已关联，
            // 红=升格），不连线
            fml_draw_clusters(img, clusters, sx, sy, LINE_WIDTH,
                              FML_CLUSTER_DRAW_SPLIT, &dbg);
        }
    }
    if (out_nl)
        *out_nl = nl;
    if (out_nvalid)
        *out_nvalid = nvalid;
    if (out_ms)
        *out_ms = ms;
    if (out_w)
        *out_w = det_w;
    if (out_h)
        *out_h = det_h;
    return img;
}

// GUI 渲染：annotate + 窗口标题 + imshow
void render(const std::vector<std::string> &files, int idx,
            const fml_elsed_params &p, const fml_cluster_params &cp,
            std::vector<cv::Vec4f> &lines, std::vector<float> &confs)
{
    int nl = 0, nvalid = 0, det_w = DS_W, det_h = DS_H;
    double ms = 0;
    const cv::Mat img = annotate(files, idx, p, cp, lines, confs, g_view,
                                 &nl, &nvalid, &ms, &det_w, &det_h);
    if (img.empty())
        return;
    size_t pos = files[idx].find_last_of('/');
    std::string name = pos == std::string::npos ? files[idx] : files[idx].substr(pos + 1);
    char title[224];
    const char *vn = g_view == 0   ? "cluster"
                     : g_view == 1 ? "conf"
                     : g_view == 2 ? "split"
                                   : "fit";
    std::snprintf(title, sizeof(title),
                  "%s  %d lines  %d valid clusters  [%d/%zu]  %.2f ms  "
                  "%dx%d  shift %+d  [%s]  (a/d switch, c view, ESC quit)",
                  name.c_str(), nl, nvalid, idx + 1, files.size(), ms, det_w,
                  det_h, p.shift_px, vn);
    cv::imshow(WIN, img);
    cv::setWindowTitle(WIN, title);
}

} // namespace

int main()
{
    // 图片目录：从可执行文件位置向上找 img/（本工程在项目根下）
    char exePath[4096];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n <= 0)
        return 1;
    exePath[n] = '\0';
    std::string dir = exePath;
    std::string base = dir.substr(0, dir.find_last_of('/'));
    dir.clear();
    for (int up = 0; up <= 3 && dir.empty(); ++up)
    {
        std::string probe = base + "/img";
        struct stat st;
        if (stat(probe.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            dir = probe;
        size_t pos = base.find_last_of('/');
        if (pos == std::string::npos || pos == 0)
            break;
        base = base.substr(0, pos);
    }
    if (dir.empty())
        dir = "img";

    // 收集图片（自然排序）
    std::vector<std::string> files;
    {
        std::vector<std::string> all;
        cv::glob(dir + "/*", all, false);
        for (const auto &f : all)
        {
            size_t pos = f.find_last_of('/');
            std::string name = pos == std::string::npos ? f : f.substr(pos + 1);
            if (isImageExt(name))
                files.push_back(f);
        }
        std::sort(files.begin(), files.end(), naturalLess);
    }
    if (files.empty())
    {
        printf("no images found in %s\n", dir.c_str());
        return 1;
    }

    // ---- 批量预览模式（FML_PREVIEW 非空：随机抽 20 张保存拟合视图）----
    if (getenv("FML_PREVIEW") != nullptr)
    {
        const std::string outDir = dir + "/preview";
        mkdir(outDir.c_str(), 0755);
        const fml_elsed_params ep = paramsFromTrackbars();
        fml_cluster_params cp = clusterParamsFromTrackbars();
        // 预览固定用 M5 乘法核打分器（可解释/轻量档）
        cp.lane_scorer = 1;

        // 清空上一轮预览（确保本次目录即随机 20 张，不混新旧）
        {
            std::vector<cv::String> old;
            cv::glob(outDir + "/*", old, false);
            for (const cv::String &f : old)
                std::remove(f.c_str());
        }
        // 随机抽取 20 张不同图片：整目录索引打乱后取前 20 个（去重且
        // 顺序随机）；读图失败自动补抽后续索引
        std::vector<size_t> order(files.size());
        for (size_t i = 0; i < files.size(); ++i)
            order[i] = i;
        std::mt19937 rng(static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        std::shuffle(order.begin(), order.end(), rng);

        int saved = 0;
        for (size_t oi = 0; oi < order.size() && saved < 20; ++oi)
        {
            const size_t i = order[oi];
            cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
            if (img.empty())
                continue;
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, gray, cv::Size(DS_W, DS_H), 0, 0,
                       cv::INTER_AREA);
            std::vector<cv::Vec4f> lines;
            std::vector<float> confs;
            fml_detect_elsed(gray, ep, lines, &confs);
            std::vector<FmlSegCluster> c1;
            fml_cluster_lines(lines, confs, DS_W, DS_H, c1, cp);
            const double sx = img.cols / (double)DS_W;
            const double sy = img.rows / (double)DS_H;
            fml_draw_clusters(img, c1, sx, sy, LINE_WIDTH,
                              FML_CLUSTER_DRAW_FIT);
            const size_t pos = files[i].find_last_of('/');
            const std::string name = pos == std::string::npos
                                         ? files[i]
                                         : files[i].substr(pos + 1);
            const std::string outPath = outDir + "/preview_" + name;
            cv::imwrite(outPath, img);
            ++saved;
            std::printf("[preview] %s\n", outPath.c_str());
        }
        std::printf("[preview] 共保存 %d 帧 -> %s\n", saved, outDir.c_str());
        return 0;
    }

    std::vector<cv::Vec4f> lines; // 检测结果缓冲
    std::vector<float> confs;     // 每条线的置信度（与 lines 对应）
    int idx = 0;

    // 先显示一帧再建进度条（窗口需已存在）
    try
    {
        render(files, idx, paramsFromTrackbars(), clusterParamsFromTrackbars(), lines, confs);
    }
    catch (const cv::Exception &e)
    {
        std::fprintf(stderr, "[fml] 窗口/显示初始化 OpenCV 异常:\n%s\n"
                             "提示: 检查 DISPLAY 与 WSLg 状态（Windows 侧执行"
                             " wsl --shutdown 后重进可复位 WSLg）\n",
                     e.what());
        return 1;
    }

    // ---- 调参窗口1：检测参数 ----
    cv::namedWindow(WIN_TB, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("len_min", WIN_TB, &tb_length, 100, onTrackbar);
    cv::createTrackbar("dist_x0.1", WIN_TB, &tb_dist10, 50, onTrackbar);
    cv::createTrackbar("merge", WIN_TB, &tb_merge, 1, onTrackbar);
    cv::createTrackbar("shift_px", WIN_TB, &tb_shift20, 40, onTrackbar);
    cv::createTrackbar("downsample", WIN_TB, &tb_downsample, 4, onTrackbar);
    cv::createTrackbar("threads", WIN_TB, &tb_threads, 1, onTrackbar);
    // ELSED 参数
    cv::createTrackbar("e_grad_th", WIN_TB, &tb_e_grad, 300, onTrackbar);
    cv::createTrackbar("e_anchor", WIN_TB, &tb_e_anchor, 64, onTrackbar);
    cv::createTrackbar("e_scan", WIN_TB, &tb_e_scan, 4, onTrackbar);
    cv::createTrackbar("e_min_len", WIN_TB, &tb_e_minlen, 60, onTrackbar);
    cv::createTrackbar("e_fitx100", WIN_TB, &tb_e_fit100, 100, onTrackbar);
    cv::createTrackbar("e_pdist10", WIN_TB, &tb_e_pdist10, 50, onTrackbar);
    cv::createTrackbar("e_valx100", WIN_TB, &tb_e_val100, 100, onTrackbar);
    cv::createTrackbar("e_junc", WIN_TB, &tb_e_junc, 1, onTrackbar);
    cv::createTrackbar("e_sigma10", WIN_TB, &tb_e_sigma10, 30, onTrackbar);

    // ---- 调参窗口2：聚类参数 ----
    cv::namedWindow(WIN_CLU, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(WIN_CLU, 400, 0); // 放在右侧，避免覆盖
    cv::createTrackbar("bands", WIN_CLU, &tb_bands, 40, onTrackbar);
    cv::createTrackbar("x_tol", WIN_CLU, &tb_x_tol, 80, onTrackbar);
    cv::createTrackbar("match_x", WIN_CLU, &tb_match_x, 80, onTrackbar);
    cv::createTrackbar("ang_tol", WIN_CLU, &tb_ang_tol, 45, onTrackbar);
    cv::createTrackbar("miss_max", WIN_CLU, &tb_miss_max, 10, onTrackbar);
    cv::createTrackbar("min_length", WIN_CLU, &tb_min_length, 200, onTrackbar);
    cv::createTrackbar("top_noise", WIN_CLU, &tb_top_noise, 100, onTrackbar);
    cv::createTrackbar("max_psi", WIN_CLU, &tb_max_psi, 90, onTrackbar);
    cv::createTrackbar("msacth_x10", WIN_CLU, &tb_msac_th, 100, onTrackbar);
    cv::createTrackbar("fit_quad", WIN_CLU, &tb_fit_quad, 1, onTrackbar);
    cv::createTrackbar("fit_rot", WIN_CLU, &tb_fit_rot, 1, onTrackbar);
    cv::createTrackbar("lane_score 0原1M5 2LR 3RBF", WIN_CLU,
                       &tb_lane_scorer, 3, onTrackbar);

    while (true)
    {
        // 参数被拖动 → 当前图重新检测（30ms 轮询保持窗口响应）
        int key = cv::waitKey(30);
        bool nav = false;
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        if (key == 'a' || key == KeyLeft)
        {
            idx = (idx + (int)files.size() - 1) % (int)files.size();
            nav = true;
        }
        if (key == 'd' || key == KeyRight || key == ' ')
        {
            idx = (idx + 1) % (int)files.size();
            nav = true;
        }
        // c：切换 聚类 / 置信度 / 簇切分 / 拟合 视图
        if (key == 'c' || key == 'C')
        {
            g_view = (g_view + 1) % 4;
            nav = true;
        }

        if (g_dirty || nav)
        {
            g_dirty = false;
            try
            {
                render(files, idx,
                       paramsFromTrackbars(),
                       clusterParamsFromTrackbars(),
                       lines, confs);
            }
            catch (const cv::Exception &e)
            {
                std::fprintf(stderr, "[fml] 渲染阶段 OpenCV 异常: %s\n", e.what());
            }
        }
    }
    return 0;
}