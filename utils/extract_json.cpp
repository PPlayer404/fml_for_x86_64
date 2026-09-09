// extract_json.cpp —— 车道线检测信息批量导出（标注/训练工具链第一步）
//
// C++ 批处理（无 GUI、两参数）：
//   extract_json <图像目录> <输出.json>
//
// 逐图跑检测+聚类，每簇输出：
//   - 中心点链 centers（显示用）
//   - 24 维打分/训练特征 features（与 cluster.hpp FML_LANE_SCORER_NFEAT 契约一致）
//   - 原始加法 score 与 M5 乘法核分数 m5（标注参考）
//   - 默认标注建议 suggest（= valid ? 1 : 0）
//   - 描述性拟合 a2/a1/a0（可选画拟合线）
//
// JSON 内嵌全部特征 → 下游 pure Python 标注/训练无需 C++ 参与。
// 依赖：dist/libfml.a + OpenCV（仅 core/imgcodecs/imgproc，无 GUI）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "fml.hpp"
#include "cluster.hpp"

namespace
{
const int DS_W = 320, DS_H = 240;

fml_elsed_params detParams()
{
    fml_elsed_params p;
    p.length_threshold = 10;
    p.distance_threshold = 1.4f;
    p.do_merge = 1;
    p.shift_px = -1;
    p.grad_th = 30;
    p.anchor_th = 8;
    p.scan_intv = 2;
    p.min_len = 15;
    p.fit_err = 0.2;
    p.px_dist = 1.2;
    p.validate_th = 0.25;
    p.treat_junc = 0;
    p.sigma = 0.9;
    p.threads = 0;
    return p;
}
fml_cluster_params cluParams()
{
    fml_cluster_params p;
    p.bands = 20;
    p.x_tol = 20.f;
    p.match_x_tol = 20.f;
    p.ang_tol = 15.f;
    p.miss_max = 3;
    p.min_length = 30.f;
    p.top_noise_ratio = 1.f / 3.f;
    p.max_psi = 60.f;
    p.msac_thresh = 5.0f;
    p.msac_iters = 16;
    p.fit_quad = 1;
    p.fit_rot = 1;
    return p;
}

// 24 维特征构造（与 cluster.cpp extractScorerFeats 一致；rank 为当前排序位次）
void buildFeatures(const FmlSegCluster &c, int rank, int nclu,
                   float maxScore, float maxSumc, float imgW, float imgH,
                   float bands, float f[24])
{
    const int n = (int)c.centers.size();
    f[0] = (float)n;
    f[1] = c.length;
    f[2] = c.score;
    f[3] = c.conf;
    const float sumc = c.conf * (float)std::max(n, 0);
    f[4] = sumc;
    f[5] = c.theta;
    f[6] = std::fabs(c.a1);
    f[7] = c.a2;
    float yBot = 0, yTop = 0, xBot = 0, xTop = 0;
    float cxmin = 1e30f, cxmax = -1e30f;
    double cxsum = 0, cx2sum = 0;
    float maxYgap = 0;
    int nbGap = 0;
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
    f[8] = std::fabs(c.a2) * ySpan;
    f[9] = std::fabs(c.a0 - 0.5f * imgW);
    f[10] = ySpan;
    f[11] = cxmean;
    f[12] = cxstd;
    f[13] = std::max(0.f, cxmax - cxmin);
    f[14] = xTop;
    f[15] = xBot;
    f[16] = tilt;
    f[17] = maxYgap;
    f[18] = (float)nbGap;
    f[19] = (float)n / ySpan;
    f[20] = maxScore > 1e-6f ? c.score / maxScore : 0.f;
    f[21] = maxSumc > 1e-6f ? sumc / maxSumc : 0.f;
    f[22] = nclu > 0 ? (float)rank / (float)nclu : 0.f;
    f[23] = std::fabs(tilt);
}

// M5 乘法核分数（与 cluster.cpp score_M5 一致；默认参数，标注参考用）
float m5Score(const float f[24], float imgH)
{
    const float sumc = std::max(f[4], 0.f);
    const float lenw = f[1] / std::max(1.f, 0.5f * imgH);
    const float capp = std::min(lenw, 0.7f);
    const float logc = std::log10(1.f + sumc);
    float s = std::pow(logc, 0.7f) * std::pow(capp, 1.3f);
    const float rnPen = std::max(0.f, f[22] - (1.f / 3.f));
    s -= 0.30f * rnPen;
    return s;
}

bool isImageExt(const std::string &f)
{
    const std::string ext = std::filesystem::path(f).extension().string();
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

void jsonEscape(std::string &s)
{
    std::string o;
    for (char ch : s)
    {
        if (ch == '"' || ch == '\\')
        {
            o.push_back('\\');
            o.push_back(ch);
        }
        else
            o.push_back(ch);
    }
    s.swap(o);
}
} // namespace

static const char *kFeatureNames = "npts,length,score,conf,sumc,theta,absA1,a2,"
    "absA2yspan,absA0dist,ySpan,cx_mean,cx_std,cx_range,xTop,xBot,tilt_dxdy,"
    "maxYgap,nbGap,ptDensity,relScore,relSumc,rankNorm,tiltAbs";

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "用法: extract_json <图像目录> <输出.json>\n");
        return 1;
    }
    const std::string dir = argv[1];
    const std::string outPath = argv[2];

    // 收集图像文件（自然排序）
    std::vector<std::string> files;
    for (const auto &e : std::filesystem::directory_iterator(dir))
    {
        if (e.is_regular_file() && isImageExt(e.path().string()))
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
    {
        std::fprintf(stderr, "无可用图片: %s\n", dir.c_str());
        return 1;
    }

    const fml_elsed_params ep = detParams();
    const fml_cluster_params cp = cluParams();

    FILE *fo = std::fopen(outPath.c_str(), "w");
    if (!fo)
    {
        std::fprintf(stderr, "无法写输出: %s\n", outPath.c_str());
        return 1;
    }

    std::fprintf(fo, "{\n");
    std::fprintf(fo, "  \"meta\": {\n");
    std::fprintf(fo, "    \"det_w\": %d, \"det_h\": %d,\n", DS_W, DS_H);
    std::fprintf(fo, "    \"bands\": %d,\n", cp.bands);
    std::fprintf(fo, "    \"feature_names\": [");
    {
        std::string fn = kFeatureNames;
        bool first = true;
        for (char ch : fn)
        {
            if (ch == ',')
            {
                std::fprintf(fo, "\",\"");
            }
            else
            {
                if (first)
                {
                    std::fprintf(fo, "\"");
                    first = false;
                }
                std::fputc(ch, fo);
            }
        }
        std::fputc('"', fo);
    }
    std::fprintf(fo, "],\n");
    std::fprintf(fo, "    \"suggest_note\": \"建议: 1=是(绿) 0=否(灰) 2=难例(蓝)\"\n");
    std::fprintf(fo, "  },\n");
    std::fprintf(fo, "  \"images\": [\n");

    size_t imgCount = 0, cluTotal = 0;
    for (const std::string &path : files)
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty())
            continue;
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(DS_W, DS_H), 0, 0, cv::INTER_AREA);

        std::vector<cv::Vec4f> lines;
        std::vector<float> confs;
        fml_detect_elsed(gray, ep, lines, &confs);
        std::vector<FmlSegCluster> cl;
        fml_cluster_lines(lines, confs, DS_W, DS_H, cl, cp);
        if (cl.empty())
            continue;

        // 帧内 max（rel 特征用）
        float maxScore = 0, maxSumc = 0;
        for (const FmlSegCluster &c : cl)
        {
            maxScore = std::max(maxScore, c.score);
            maxSumc = std::max(maxSumc,
                c.conf * (float)std::max((int)c.centers.size(), 0));
        }
        const int nclu = (int)cl.size();

        std::string name = std::filesystem::path(path).filename().string();
        jsonEscape(name);
        std::fprintf(fo, imgCount ? ",\n    {\n" : "    {\n");
        std::fprintf(fo, "      \"file\": \"%s\",\n", name.c_str());
        std::fprintf(fo, "      \"clusters\": [\n");
        for (int i = 0; i < nclu; ++i)
        {
            const FmlSegCluster &c = cl[i];
            float f[24];
            buildFeatures(c, i, nclu, maxScore, maxSumc, (float)DS_W,
                          (float)DS_H, (float)cp.bands, f);
            const float m5 = m5Score(f, (float)DS_H);
            const int suggest = c.valid ? 1 : 0;
            std::fprintf(fo, i ? ",\n        {\n" : "        {\n");
            std::fprintf(fo, "          \"rank\": %d,\n", i);
            std::fprintf(fo, "          \"npts\": %d,\n", (int)c.centers.size());
            std::fprintf(fo, "          \"length\": %.1f,\n", c.length);
            std::fprintf(fo, "          \"score\": %.4f,\n", c.score);
            std::fprintf(fo, "          \"m5\": %.6f,\n", m5);
            std::fprintf(fo, "          \"conf\": %.2f,\n", c.conf);
            std::fprintf(fo, "          \"valid\": %d,\n", (int)c.valid);
            std::fprintf(fo, "          \"suggest\": %d,\n", suggest);
            std::fprintf(fo, "          \"a2\": %.6f, \"a1\": %.6f, \"a0\": %.2f,\n",
                         c.a2, c.a1, c.a0);
            // centers（自底向上；整数像素）
            std::fprintf(fo, "          \"centers\": [");
            for (size_t k = 0; k < c.centers.size(); ++k)
            {
                std::fprintf(fo, k ? ",[%d,%d]" : "[%d,%d]",
                             cvRound(c.centers[k].x), cvRound(c.centers[k].y));
            }
            std::fprintf(fo, "],\n");
            // features（24 维）
            std::fprintf(fo, "          \"features\": [");
            for (int j = 0; j < 24; ++j)
                std::fprintf(fo, j ? ",%.6f" : "%.6f", f[j]);
            std::fprintf(fo, "]\n");
            std::fprintf(fo, "        }");
            ++cluTotal;
        }
        std::fprintf(fo, "\n      ]\n    }");
        ++imgCount;
    }
    std::fprintf(fo, "\n  ]\n}\n");
    std::fclose(fo);

    std::printf("[extract_json] 图 %zu 张, 簇 %zu 条 -> %s\n",
                imgCount, cluTotal, outPath.c_str());
    return 0;
}