1. 第一层
第一层我会放一个可选的 RevIN / instance normalization。
理由是很多 TSF 数据存在明显分布漂移，单纯全局标准化不够。数据处理阶段已经做了 train-only 标准化，但模型内部再加入 RevIN，可以提高跨时间段鲁棒性。

2. 第二层
类似于HAO，映射到欧式 / Poincare / Lorentz 双曲空间。

3. 第三层
这一层分为几个部分，关系可以理解为并行。
    第一部分：多尺度 Patch Embedding。
    输入：
    > x: [B, L, C].
    模型内部做多尺度 patch：
    > patch_len = 8, 16, 32 或 64.
    > patch_stride = patch_len // 2 或自定义
    得到：
    > X_patch_s: [B, C, N_s, patch_len_s].
    然后经过共享或半共享 patch embedding：
    > Z_s: [B, C, N_s, d].
    里的 s 表示不同尺度。
    为什么要多尺度？因为时间序列预测中：
    - 短 patch 捕获局部波动；
    - 中 patch 捕获周期片段；
    - 长 patch 捕获趋势/ regime；
    - 双曲空间适合把这些尺度组织成层次结构。
    这部分会吸收 PatchTST / HyperTime 的思想，但不会直接把 Patch 化写进数据层。
    然后对于多尺度的embedding进行学习时间依赖图。
    
    第二部分：变量耦合图构建
    类似于HAO，可以拿第一部分最长的patch embedding去学习变量耦合图。具体的结构可以参考HAO源码论文。

    第三部分：用变量的耦合关系去“指导”“优化”时间上的依赖关系。
    具体的设计我还没有想好，你可以代替我来设计规划一下。

第四层：未规划，先不进行实现。