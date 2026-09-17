# 同数据动作加权 SFT 实测

结论：FAIL_BOUNDED_TEN_TRAIN_ROUTE_LEARNABILITY。完整训练：True。可学性：False。

固定10训练路线565决策、同initial、同两rank顺序、最多200更新，唯一训练目标变化为温和类别加权。
这是已知标准工程修复，不是原创算法；无机制监督、无dev选优、无新导航闭环。

权重：{"actions": ["move_forward", "turn_left", "turn_right", "STOP"], "counts": {"turn_right": 90, "move_forward": 339, "turn_left": 126, "STOP": 10}, "raw": [1.0, 1.6402671094904606, 1.9407902170679516, 4.0], "normalizer": 1.3457429651892276, "values": [0.7430839512948061, 1.2188561648990819, 1.442170063133158, 2.9723358051792244], "formula": "min(4,sqrt(max_train_count/count)); normalize train-frequency mean to 1", "denominator": "global sum of target weights per optimizer update", "source": "same fixed 10 train routes, not sampled schedule or dev"}

## 实测对照

{
  "control": {
    "stage": "after",
    "decisions": 565,
    "CE": 0.9083674631023829,
    "accuracy": 0.6247787610619469,
    "macro_recall": 0.2853736479842674,
    "recall": [
      0.9970501474926253,
      0.05555555555555555,
      0.08888888888888889,
      0.0
    ],
    "STOP_precision": 0.0,
    "STOP_recall": 0.0,
    "confusion": [
      [
        338,
        1,
        0,
        0
      ],
      [
        119,
        7,
        0,
        0
      ],
      [
        77,
        5,
        8,
        0
      ],
      [
        10,
        0,
        0,
        0
      ]
    ],
    "learnability_pass": false
  },
  "treatment": {
    "stage": "after",
    "decisions": 565,
    "CE": 0.9470334432821358,
    "accuracy": 0.6265486725663717,
    "macro_recall": 0.2968780727630285,
    "recall": [
      0.976401179941003,
      0.1111111111111111,
      0.1,
      0.0
    ],
    "STOP_precision": 0.0,
    "STOP_recall": 0.0,
    "confusion": [
      [
        331,
        8,
        0,
        0
      ],
      [
        112,
        14,
        0,
        0
      ],
      [
        70,
        11,
        9,
        0
      ],
      [
        10,
        0,
        0,
        0
      ]
    ],
    "learnability_pass": false
  },
  "differences": {
    "CE": 0.03866598017975287,
    "accuracy": 0.0017699115044248481,
    "macro_recall": 0.011504424778761124,
    "STOP_precision": 0.0,
    "STOP_recall": 0.0
  },
  "recall_differences": [
    -0.020648967551622377,
    0.05555555555555555,
    0.011111111111111113,
    0.0
  ],
  "claim": "descriptive single-seed matched exposed training-set comparison; not novelty or generalization",
  "paired_newly_correct": 10,
  "paired_newly_wrong": 9,
  "per_route": [
    {
      "row": 0,
      "decisions": 62,
      "control_correct": 35,
      "treatment_correct": 33
    },
    {
      "row": 1,
      "decisions": 51,
      "control_correct": 31,
      "treatment_correct": 34
    },
    {
      "row": 2,
      "decisions": 52,
      "control_correct": 40,
      "treatment_correct": 38
    },
    {
      "row": 3,
      "decisions": 42,
      "control_correct": 28,
      "treatment_correct": 28
    },
    {
      "row": 4,
      "decisions": 84,
      "control_correct": 50,
      "treatment_correct": 50
    },
    {
      "row": 5,
      "decisions": 58,
      "control_correct": 37,
      "treatment_correct": 40
    },
    {
      "row": 6,
      "decisions": 39,
      "control_correct": 31,
      "treatment_correct": 29
    },
    {
      "row": 7,
      "decisions": 47,
      "control_correct": 27,
      "treatment_correct": 28
    },
    {
      "row": 8,
      "decisions": 56,
      "control_correct": 24,
      "treatment_correct": 24
    },
    {
      "row": 9,
      "decisions": 74,
      "control_correct": 50,
      "treatment_correct": 50
    }
  ]
}

正向局部变化与固定通过门槛分别报告。仅同一暴露训练集、单seed的描述性结果，不能当泛化或导航收益。

## 效率与交付

训练更新：200，训练决策：6277。
效率：{"median_update_seconds": 9.131670713424683, "steady_effective_decisions_per_second": 3.4578979118213335, "training_wall_seconds": 1815.3948285579681}
执行与恢复：{"error": null, "returncodes": [0, 0], "wall_seconds": 2009.8303031921387, "GPU_count": 2, "conservative_GPU_seconds": 4019.660608291626, "real_tasks_stopped": 0, "all_leased_holders_restored": true}
来源/代码/权重哈希：606项通过。失败旧实验未修改。
导航基座尚不能宣布完成；不自动扩大训练或增加机制损失。
