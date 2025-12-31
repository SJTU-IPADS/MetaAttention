# AttentionEngine: Responsible AI FAQ

## What is AttentionEngine?
**Introduce:** AttentionEngine is an accelerator designed for Transformer attention variants, providing highly efficient kernels for custom attention mechanisms. AttentionEngine should only be used for research and experimental purposes. Further testing and validation is needed before application in real-world settings.

## What it does:
AttentionEngine applies FlashAttention algorithms to a broader range of tasks and automatically tunes the optimal configurations for custom kernels.

## Inputs and Outputs:
**Input:** Customized attention function.  
**Output:** Compiled high efficient kernels of the attention function.

## What can AttentionEngine do?
**Enhanced training and Inference Speed:** By providing highly efficient attention kernels, AttentionEngine reduces the time spent on the attention module, accelerating both training and inference.  
**Empowering model designers to experiment with new attention mechanisms:** With AttentionEngine, model designers can explore various attention variants without concerns about efficiency. This allows them to quickly test the performance of their new models, accelerating advancements in this research field.

## What is/are AttentionEngine’s intended use(s)?
AttentionEngine is an academic and experimental research project. AttentionEngine should only be used for research and experimental purposes. Further testing and validation are needed before application in real-world settings.  
**Enhancing training and Inference Speed:** By providing highly efficient attention kernels, AttentionEngine reduces the time spent on the attention module, accelerating both training and inference.  
**Empowering model designers to experiment with new attention mechanisms:** With AttentionEngine, model designers can explore various attention variants without concerns about efficiency. This allows them to quickly test the performance of their new models, accelerating advancements in this research field.

## How was AttentionEngine evaluated? What metrics are used to measure performance?
**TFLOPS of attention module:** This metric evaluates the efficiency of the attention module after being accelerated by AttentionEngine. Higher TFLOPS indicate more efficient performance, benefiting both training and inference.

## What are the limitations of AttentionEngine? How can users minimize the impact of AttentionEngine’s limitations when using the system?
**Poor performance:**  
**Potential harms:** AttentionEngine applies a general solution to all attention variants, which may not be the most effective approach for certain attention mechanisms. In such cases, users may find the speedup provided by AttentionEngine to be suboptimal.  
**Potential mitigation:** AttentionEngine is an open-source research project, and its capabilities have only been demonstrated and evaluated on a limited set of attention variants. It does not guarantee excellent results under all configurations. Therefore, users should carefully compare and evaluate the performance of the attention module before and after applying AttentionEngine, and determine whether to use the provided kernels.

## What operational factors and settings allow for effective and responsible use of AttentionEngine?
**Comprehensive Testing:** Before integrating the kernels provided by AttentionEngine, users must conduct extensive testing to ensure safe and adequate performance for their intended use cases. This includes validating models on relevant datasets and in scenarios that closely resemble real-world applications.  
**Performance Verification:** It is imperative for users to verify that the performance of the attention module accelerated by AttentionEngine meets their specific requirements. This entails thorough evaluations and, if necessary, recalibrations based on performance outcomes to ensure the model's effectiveness and reliability in operational settings.  
**Open Communication:** Users should be aware that AttentionEngine is an academic project with a limited validation scope. It is crucial to communicate the potential risks of integrating AttentionEngine kernels into critical systems without thorough testing and validation.
