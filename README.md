# RL-TRIM: Reinforcement Learning drivenTransformer Model Structured Pruning
###  This framework employs reinforcement learning for the structured pruning of Transformer models, specifically targeting models like LLaMA


Our approach involves pruning at different granularities, including head pruning and
intermediate dimension pruning, which directly reduces memory size and computational load,
facilitating acceleration on consumer GPUs. By utilizing a reinforcement learning agent to
determine the optimal pruning strategy, RL-TRIM achieves a significant balance between model
size reduction and performance retention, offering a scalable and efficient solution for optimizing
various Transformer architectures.
ii


___________________________________________________________________________________________________________
# Acknowledgments
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices. Thanks for providing the pruning framework
- LLM-Pruner, which utilizes LM Evaluation Harness, PEFT, and Alpaca-LoRA. Thanks for the pioneering work on structured pruning of LLMs!
