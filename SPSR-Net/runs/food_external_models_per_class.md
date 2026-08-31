# FOOD test：外部模型分类型实体识别结果

评测口径为**实体边界与实体类型均完全一致**的 micro P/R/F1（单位：%）。除 LEBERT 外均以原始 FOOD test 的 822 个实体作为 gold；LEBERT 是平面 BIO 标注模型，3 个重叠实体无法由其标签序列表示，因此 gold 为 819。Flat-Lattice-Transformer 的训练也是平面 BMES；下表给出其对原始 822 个实体的结果，并单独保留平面 gold 版本以供核验。

## 总体

| 模型 | Precision | Recall | F1 | Gold |
|---|---:|---:|---:|---:|
| CNN_Nested_NER | 98.40 | 97.08 | 97.73 | 822 |
| LEBERT | 97.31 | 97.19 | 97.25 | 819 |
| SPSR-Net | 96.56 | 95.62 | 96.09 | 822 |
| W2NER | 98.47 | 93.67 | 96.01 | 822 |
| locate-and-label | 96.99 | 94.16 | 95.56 | 822 |
| DiffusionNER | 97.56 | 87.47 | 92.24 | 822 |
| Flat-Lattice-Transformer | 70.49 | 86.01 | 77.48 | 822 |

## 各类别 F1

每格为 `F1`。括号内是原始 test 的该类 gold 数；`PER=1`、`SN=6`，不应据此作稳定性结论。

| 模型 | FC (197) | FN (37) | IN (156) | IV (151) | ORG (35) | PER (1) | SN (6) | SNum (239) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN_Nested_NER | 100.00 | 95.89 | 97.39 | 98.66 | 83.58 | 0.00 | 60.00 | 98.76 |
| LEBERT* | 99.49 | 100.00 | 97.37 | 98.66 | 87.32 | 0.00 | 42.11 | 97.92 |
| SPSR-Net | 99.24 | 83.08 | 96.75 | 98.33 | 78.26 | 0.00 | 46.15 | 97.52 |
| W2NER | 100.00 | 45.83 | 97.39 | 98.66 | 76.06 | 0.00 | 50.00 | 99.16 |
| locate-and-label | 98.22 | 86.15 | 95.08 | 94.08 | 86.49 | 0.00 | 40.00 | 98.56 |
| DiffusionNER | 94.95 | 5.26 | 97.37 | 98.32 | 84.51 | 0.00 | 28.57 | 92.13 |
| Flat-Lattice-Transformer† | 99.24 | 87.32 | 51.90 | 78.86 | 72.73 | 0.00 | 0.00 | 84.43 |

\* LEBERT 的 FC/FN/IN gold 分别为 196/36/155；总 gold 为 819，因为 3 个原始重叠实体在 BIO 序列中无法表达。

† Flat-Lattice-Transformer 评估时对原始 822 实体计分；其训练/推理本身是非重叠 BMES。若按其平面标签 gold 计分，overall 为 P=70.49、R=86.22、F1=77.56，gold=820。

## 可追溯原始结果

| 模型 | 结果文件/日志 |
|---|---|
| SPSR-Net | `food_spsr_net_bhpc_per_class_test.json` |
| CNN_Nested_NER | `../../CNN_Nested_NER/outputs/food/cnnner_per_class.json` |
| LEBERT | `food_external_per_class_lebert.json` |
| W2NER | `../../W2NER/outputs/food/w2ner_per_class.json` |
| Flat-Lattice-Transformer | `../../Flat-Lattice-Transformer/outputs/food/flat_lattice_per_class.json` |
| DiffusionNER | `../../DiffusionNER/outputs/food/per_class_eval_logs/food_per_class_eval/2026-08-16_17:24:37.175399/all.log` |
| locate-and-label | `../../locate-and-label/outputs/food/per_class_eval_logs/food_per_class_eval/2026-08-16_17:42:39.856962/all.log` |
