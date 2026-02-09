# Neural Network Watermarking (Course Project)

Course project on **neural network watermarking**: techniques to embed a hidden “signature” into a model so ownership can be **verified** later, ideally with minimal impact on accuracy and reasonable robustness to model modifications (e.g., fine-tuning, pruning, compression).

## Project Goals
- Implement and compare watermarking approaches for deep neural networks.
- Evaluate trade-offs between:
  - **Fidelity**: clean-task performance drop after watermarking
  - **Capacity**: how many bits can be embedded
  - **Robustness**: survives fine-tuning / pruning / quantization / distillation (as tested)
  - **Security**: resistance to watermark removal / forgery (as tested)
- Provide a reproducible training + verification pipeline and a short report.