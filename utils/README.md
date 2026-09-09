# 车道线数据工具链（utils）

车道线检测打分器的数据采集、标注与训练工具链。三件套 + 库，跨平台数据管线：

```
[检测/特征提取]  extract_json   <图像目录> <输出.json>   （Linux 预编译，无 GUI）
[人工标注]       annotate.py    <图像目录> <检测.json>   （跨平台 Python GUI）
[权重训练]       train.py       <标注.json> --mode   （跨平台，输出 weights.txt）
[库接口]         libfml.a + cluster.hpp/fml.hpp          （C++ 打分器加载权重）
```

JSON 内嵌 24 维打分/训练特征 + M5 参考分 + 默认标注建议，Python 侧全程无需 C++
参与重算特征。

---

## 1. extract_json —— 批量检测 + 特征导出（C++，无 GUI）

用法：

```
./extract_json <图像目录> <输出.json>
```

逐图执行 fml 检测 + 车道线聚类，每簇输出：

| 字段 | 说明 |
|---|---|
| `rank` | 输出排序位次 |
| `npts` / `length` | 中心链点数和长度 |
| `score` | 原始加法分数 |
| `m5` | M5 乘法核参考分（`log10(1+Σc)^a·min(lenw,cap)^b`） |
| `conf` | 合并置信度均值 |
| `valid` / `suggest` | 默认有效性建议（1=是 0=否；供标注起点） |
| `a2/a1/a0` | 描述性拟合参数 |
| `centers` | 中心点链（显示用） |
| `features` | 24 维打分/训练特征（与 `meta.feature_names` 一一对应） |

`meta.feature_names` 固定 24 项，顺序即下游契约。
JSON 平均每图约 3-6KB，175 图全量约 1MB。

---

## 2. annotate.py —— 标注 GUI（跨平台）

用法：

```
pip install opencv-python
python annotate.py <图像目录> <检测.json> [输出.json]
```

- 默认输出 = 输入检测.json（在原文件上增量回写 `suggest`）
- 三档标注：**绿(1)=是 / 灰(0)=否 / 蓝(2)=难例**
- 鼠标点击中心链循环切换，按 `c` 在「仅链 / 链+M5分」之间切换显示
- `a`/`d` 翻图自动保存；`s` 手动保存；`q` 退出保存
- 再次打开同一 JSON 自动继承上次标注

---

## 3. train.py —— 权重训练（跨平台）

用法：

```
pip install numpy scikit-learn
python train.py <标注.json> --mode m5   [--out weights.txt]
python train.py <标注.json> --mode lr
python train.py <标注.json> --mode rbf
```

| 模式 | 打分档 | 说明 |
|---|---|---|
| `m5` | lane_scorer=1 | M5 乘法核，网格搜索 a/b/cap/rank_w |
| `lr` | lane_scorer=2 | 白盒线性 LR（24 特征标准化） |
| `rbf` | lane_scorer=3 | RBF-SVM（效果最强） |

输出 `weights.txt`（文本），C++ 端加载：

```cpp
fml_lane_scorer_weights w;
fml_lane_scorer_init(w, mode);          // 或 fml_lane_scorer_load(&w, "weights.txt")
fml_cluster_params p;
p.lane_scorer = mode;                   // 1=M5 2=LR 3=RBF
p.lane_scorer_w = &w;
fml_cluster_lines(lines, confs, W, H, clusters, p);
```

---

## 4. 内置权重说明

- `cluster.cpp` 内置当前数据集训练的三档权重（`scorer_weights_data.h`），
  用 `fml_lane_scorer_default(mode)` 直接读取，或 `fml_lane_scorer_init` 初始化。
- 如需在新场景重训，用本工具链完成 标注 → 训练 → weights.txt → load 即可。

---

## 5. 典型全流程

```
# 1. 批量检测 + 特征导出（Linux）
./extract_json ./images ./detections.json

# 2. Windows / 任意平台标注
python annotate.py ./images ./detections.json

# 3. 训练（输出到上位机可加载的权重）
python train.py ./detections.json --mode rbf --out weights.txt

# 4. 上位机加载（见 cluster.hpp：fml_lane_scorer_load）
```