# Variational Information Inference (IDTL) for Multirate Industrial Processes

This repository contains the official implementation of **Variational Information Inference: An Interpretable Disentangled Transfer Learning Quality Prediction for Multirate Industrial Processes** (TCYB 2025).

## 📖 Project Overview

Multirate industrial processes generate data at different sampling rates, which poses challenges for accurate soft sensing and quality prediction. We introduce a novel set-based perspective and an **Interpretable Disentangled Transfer Learning (IDTL)** methodology to effectively handle multirate data without information loss.

### 📷 Illustrations

![Model Architecture](https://github.com/user-attachments/files/20201576/framework.pdf)

![Prediction Scatter](https://github.com/user-attachments/files/20201773/scatter-eps-converted-to.pdf)

![Domain-invariant Representations](https://github.com/user-attachments/files/20201781/inv-eps-converted-to.pdf)

![Domain-specific Representations](https://github.com/user-attachments/files/20201853/spf-eps-converted-to.pdf)




### 🚀 Highlights

1. **Novel Perspective**: Treat multirate processes as sets, preserving all information and avoiding the downsides of up/down-sampling and delayed features.
2. **Novel Methodology**: IDTL disentangles domain-invariant and domain-specific representations of multirate sets. We derive a principled optimization objective and evidence lower bound for set-based disentanglement.
3. **Theoretical Insights**: We prove that maximizing the IDTL objective infers optimal fused disentangled representations, enhancing interpretability in industrial soft sensing.
4. **Insightful Results**: Extensive experiments on a debutanizer column and actual polyester esterification data demonstrate IDTL’s superiority in quality prediction and interpretability.

## 🛠️ Installation

```bash
# Create a dedicated conda environment
conda create -n IDTL python=3.8.17
conda activate IDTL

# Install the package and dependencies
pip install -all .
```

## ▶️ Usage Example

Due to corporate confidentiality, we only provide examples on the public debutanizer column dataset.

```bash
# Run the IDTL pipeline
python IDTL.py
```

Adjust any configuration flags or data paths inside `IDTL.py` as needed for your setup.

## 📚 Citation

If you use this code in your research, please cite:

> H. Ding, K. Hao, L. Chen, and X. Cai, "Industrial Metaverse for Smart Manufacturing: Model, Architecture, and Applications," *IEEE Transactions on Cybernetics*, 2025.

## 🙏 Acknowledgements

This work was supported by:

* Fundamental Research Funds for the Central Universities (2232022D-08, 2232021D-36)
* National Natural Science Foundation of China (62403121)
* Shanghai Sailing Program, China (Grant No. 22YF1401500)
* Chenguang Program of Shanghai Education Development Foundation and Shanghai Municipal Education Commission (22CGA36)

