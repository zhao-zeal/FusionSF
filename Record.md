本阶段研究计划分为四步推进，并与 Codex 协作完成。

第一步，修复 FusionSF 的数据处理、实验注册、跨站 scaler 注入和评估流程，在 MMSP 上完成 Power、Power+NWP、全模态及 zero-shot 初步实验，并提取 TS embedding 和 Fusion embedding。

阶段一完成的判定标准就是：四组干净初步结果 + 两个最佳模型的TS/Fusion embedding。


第二步，将 FusionSF 迁移到 solar-energy 的三个数据集，统一使用已有的数据划分和评估流程，分别完成单功率和功率+NWP预测，重点验证 `seq_len=336`、`pred_len=1/16/288` 下的适配效果，其中GEFCom是1h时间粒度，`pred_len=1/4/72`。

第三步，将 FusionSF 提取的 embedding 输入 Chronos-2，比较原始 Chronos-2、Chronos-2+TS embedding 和 Chronos-2+Fusion embedding，目标是提高 MMSP 和 solar-energy 三个数据集上的 zero-shot 预测精度。

第四步，整理阶段性结果并制作周报 PPT。

整体采用“小任务—Codex实现—ChatGPT审核—通过后继续”的协作方式。每次 Codex 完成关键修改或实验后，需要提交代码差异、测试结果、配置、实验记录和指标，由网页版 ChatGPT 审核后再进入下一阶段。
