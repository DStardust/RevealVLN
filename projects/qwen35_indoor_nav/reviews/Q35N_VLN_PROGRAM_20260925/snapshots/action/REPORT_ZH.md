COMPLETE

固定已暴露 unseen200；两种全新 TRAIN-only 纠错头与同轮原生模型完成真实配对。

NATIVE: 112.0/200, SR=0.5600, SPL=0.5111
CONCAT: 116.0/200, SR=0.5800, SPL=0.5210
EVIDENCE: 111.0/200, SR=0.5550, SPL=0.5130

{"CONCAT_vs_NATIVE": {"wins": [4, 168, 37, 39, 149, 95], "losses": [22, 74], "delta_sr": 0.02}, "EVIDENCE_vs_NATIVE": {"wins": [4, 30, 36, 37, 39, 89], "losses": [22, 53, 79, 195, 80, 120, 121], "delta_sr": -0.005}, "EVIDENCE_vs_CONCAT": {"wins": [30, 36, 74, 89], "losses": [168, 53, 149, 79, 195, 80, 95, 120, 121], "delta_sr": -0.025}}

单种子工程对照；未证明论文贡献，不自动采用。
