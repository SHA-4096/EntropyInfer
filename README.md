# EntropyInfer

An efficient train-free system that accelerates LLM inference, utilizing entropy information of the context.

Experiments on Llama, Qwen and Pangu model series have shown that our method achieve high end-to-end speedup while maintaining generation quality. We currently release implementation that supports openPangu model series, including [openPangu-Embedded-1B-v1.1](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1) and [openPangu-Embedded-7B-v1.1](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1).

![framework](resources/framework.png)

## Environment Preparation

```bash
pip install -r requirements.txt
```

## Conducting Experiments

To conduct experiment on LongBench dataset, run the following script:

```bash
# Run experiment
bash experiment/LongBench/evaluate_longbench.sh
# Run evaluation
python experiment/LongBench/eval.py
```
