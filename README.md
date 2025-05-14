# Variational Information Inference (IDTL) for Multirate Industrial Processes

This repository contains the official implementation of **Variational Information Inference: An Interpretable Disentangled Transfer Learning Quality Prediction for Multirate Industrial Processes** (TCYB 2025).

## 📖 Project Overview

Multirate industrial processes generate data at different sampling rates, which poses challenges for accurate soft sensing and quality prediction. We introduce a novel set-based perspective and an **Interpretable Disentangled Transfer Learning (IDTL)** methodology to effectively handle multirate data without information loss.

### 📷 Illustrations

![Model Architecture](https://github.com/user-attachments/files/20201576/framework.pdf)

![Prediction Scatter](https://github.com/user-attachments/files/20201773/scatter-eps-converted-to.pdf)

![Domain-invariant Representations](https://github.com/user-attachments/files/20201781/inv-eps-converted-to.pdf)

![Domain-specific Representations](https://github.com/user-attachments/files/20201853/spf-eps-converted-to.pdf)

## 📊 Performance Comparison

As shown in the table, IDTL outperforms the other methods in both MAE and RMSE:

| Method              | SVR [1] | CIDA [2] | SAD [3] | VDI [4] | VPTN [5] | **IDTL**   |
|---------------------|---------------:|---------------:|--------------:|-------------:|---------------:|-----------:|
| **MAE**             | 0.0963         | 0.0851         | 0.0936        | 0.0823       | 0.0819         | **0.0638** |
| **RMSE**            | 0.1412         | 0.1048         | 0.1149        | 0.0995       | 0.1207         | **0.0805** |




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

## References
@Inbook{1,
author="Awad, Mariette
and Khanna, Rahul",
title="Support Vector Regression",
bookTitle="Efficient Learning Machines: Theories, Concepts, and Applications for Engineers and System Designers",
year="2015",
publisher="Apress",
address="Berkeley, CA",
pages="67--80",
isbn="978-1-4302-5990-9"
}
@inproceedings{2,
author = {Wang, Hao and He, Hao and Katabi, Dina},
title = {Continuously indexed domain adaptation},
year = {2020},
booktitle = {Proc. Int. Conf. Mach. Learn.},
articleno = {918},
numpages = {10}
}
@ARTICLE{3,
  author={Zhou, Qianyu and Gu, Qiqi and Pang, Jiangmiao and Lu, Xuequan and Ma, Lizhuang},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  title={Self-adversarial disentangling for specific domain adaptation},
  year={2023},
  volume={45},
  number={7},
  pages={8954-8968}}

@inproceedings{4,
title={Domain-indexing variational bayes: interpretable domain index for domain adaptation},
author={Zihao Xu and Guang-Yuan Hao and Hao He and Hao Wang},
booktitle={Proc. Int. Conf. Learn. Represent.},
year={2023}
}
@ARTICLE{5,
  author={Chai, Zheng and Zhao, Chunhui and Huang, Biao},
  journal={IEEE Trans. Cybern.},
  title={Variational progressive-transfer network for soft sensing of multirate industrial processes},
  year={2022},
  volume={52},
  number={12},
  pages={12882-12892}}
