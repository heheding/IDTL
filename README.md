# Variational Information Inference: An Interpretable Disentangled Transfer Learning Quality Prediction for Multirate Industrial Processes

This repository contains the official implementation of **Variational Information Inference: An Interpretable Disentangled Transfer Learning Quality Prediction for Multirate Industrial Processes** (TCYB 2025).

## 📖 Project Overview

Multirate industrial processes generate data at different sampling rates, which poses challenges for accurate soft sensing and quality prediction. We introduce a novel set-based perspective and an **Interpretable Disentangled Transfer Learning (IDTL)** methodology to effectively handle multirate data without information loss.

### 📷 Illustrations

<p align="center">
  <img src="https://github.com/user-attachments/assets/45565b50-1f23-4ed9-8434-22bc9e50e3b3" alt="Model Architecture"/>
  <br>
  <em>Model Architecture</em>
</p>

<div align="center">
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
    <div style="text-align: center; margin: 10px;">
      <img src="https://github.com/user-attachments/assets/cee6a10c-5d2f-4a5d-8846-65d7fbd63874" width="10%" alt="Prediction Scatter"/>
      <p>Prediction Scatter</p>
    </div>
    <div style="text-align: center; margin: 10px;">
      <img src="https://github.com/user-attachments/assets/97dcc11b-a01d-49ce-8e2d-65e9da279380" width="30%" alt="Domain-invariant Representations"/>
      <p>Domain-invariant Representations</p>
    </div>
    <div style="text-align: center; margin: 10px;">
      <img src="https://github.com/user-attachments/assets/7ccf4909-5065-41e6-af89-ac24d9c60f82" width="30%" alt="Domain-specific Representations"/>
      <p>Domain-specific Representations</p>
    </div>
  </div>
</div>


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


## 📊 Performance Comparison

As shown in the table, IDTL outperforms the other methods in both MAE and RMSE:

| Method              | SVR [1] | CIDA [2] | SAD [3] | VDI [4] | VPTN [5] | **IDTL**   |
|---------------------|---------------:|---------------:|--------------:|-------------:|---------------:|-----------:|
| **MAE**             | 0.0963         | 0.0851         | 0.0936        | 0.0823       | 0.0819         | **0.0638** |
| **RMSE**            | 0.1412         | 0.1048         | 0.1149        | 0.0995       | 0.1207         | **0.0805** |



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
[1] M. Awad and R. Khanna, Support Vector Regression. Berkeley, CA: Apress, 2015, pp. 67–80.

[2] H. Wang, H. He, and D. Katabi, “Continuously indexed domain adaptation,” in Proc. Int. Conf. Mach. Learn., 2020.

[3] Q. Zhou, Q. Gu, J. Pang, X. Lu, and L. Ma, “Self-adversarial disentangling for specific domain adaptation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 7, pp. 8954–8968, 2023.

[4] Z. Xu, G.-Y. Hao, H. He, and H. Wang, “Domain-indexing variational bayes: interpretable domain index for domain adaptation,” in Proc. Int. Conf. Learn. Represent., 2023.

[5] Z. Chai, C. Zhao, and B. Huang, “Variational progressive-transfer networkforsoftsensingofmultirateindustrialprocesses,” IEEETrans. Cybern.,vol.52,no.12,pp.12882–12892,2022.
