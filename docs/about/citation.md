---
layout: default
title: Citation
parent: About
nav_order: 1
---

# Citation
{: .no_toc }

How to cite Emotion-LLaMA in your research.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Main Paper (NeurIPS 2024)

If you find our work helpful for your research, please consider giving a star ⭐ and citing our paper:

### BibTeX

```bibtex
@inproceedings{NEURIPS2024_c7f43ada,
  author = {Cheng, Zebang and Cheng, Zhi-Qi and He, Jun-Yan and Wang, Kai and Lin, Yuxiang and Lian, Zheng and Peng, Xiaojiang and Hauptmann, Alexander},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {110805--110853},
  publisher = {Curran Associates, Inc.},
  title = {Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f43ada17acc234f568dc66da527418-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```

### APA Style

Cheng, Z., Cheng, Z.-Q., He, J.-Y., Wang, K., Lin, Y., Lian, Z., Peng, X., & Hauptmann, A. (2024). Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning. In *Advances in Neural Information Processing Systems* (Vol. 37, pp. 110805-110853). Curran Associates, Inc.

### IEEE Style

Z. Cheng et al., "Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning," in *Advances in Neural Information Processing Systems*, vol. 37, pp. 110805-110853, 2024.

---

## MER2024 Challenge Paper

If you use our Conv-Attention enhancement or reference our MER2024 challenge work:

### BibTeX

```bibtex
@inproceedings{10.1145/3689092.3689404,
  author = {Cheng, Zebang and Tu, Shuyuan and Huang, Dawei and Li, Minghan and Peng, Xiaojiang and Cheng, Zhi-Qi and Hauptmann, Alexander G.},
  title = {SZTU-CMU at MER2024: Improving Emotion-LLaMA with Conv-Attention for Multimodal Emotion Recognition},
  year = {2024},
  isbn = {9798400712036},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3689092.3689404},
  doi = {10.1145/3689092.3689404},
  abstract = {This paper presents our winning approach for the MER-NOISE and MER-OV tracks of the MER2024 Challenge on multimodal emotion recognition. Our system leverages the advanced emotional understanding capabilities of Emotion-LLaMA to generate high-quality annotations for unlabeled samples, addressing the challenge of limited labeled data. To enhance multimodal fusion while mitigating modality-specific noise, we introduce Conv-Attention, a lightweight and efficient hybrid framework. Extensive experimentation validates the effectiveness of our approach. In the MER-NOISE track, our system achieves a state-of-the-art weighted average F-score of 85.30\%, surpassing the second and third-place teams by 1.47\% and 1.65\%, respectively. For the MER-OV track, our utilization of Emotion-LLaMA for open-vocabulary annotation yields an 8.52\% improvement in average accuracy and recall compared to GPT-4V, securing the highest score among all participating large multimodal models.},
  booktitle = {Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing},
  pages = {78–87},
  numpages = {10},
  keywords = {mer2024, noise robustness, open-vocabulary recognition},
  location = {Melbourne VIC, Australia},
  series = {MRAC '24}
}
```

### APA Style

Cheng, Z., Tu, S., Huang, D., Li, M., Peng, X., Cheng, Z.-Q., & Hauptmann, A. G. (2024). SZTU-CMU at MER2024: Improving Emotion-LLaMA with Conv-Attention for Multimodal Emotion Recognition. In *Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing* (pp. 78-87). Association for Computing Machinery.

---

## MERR Dataset

If you use the MERR dataset in your research:

```bibtex
@inproceedings{NEURIPS2024_c7f43ada,
  author = {Cheng, Zebang and Cheng, Zhi-Qi and He, Jun-Yan and Wang, Kai and Lin, Yuxiang and Lian, Zheng and Peng, Xiaojiang and Hauptmann, Alexander},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  year = {2024}
}
```

For the MER-Factory pipeline:
- Visit [MER-Factory GitHub](https://github.com/Lum1104/MER-Factory)
- Check [MER-Factory Documentation](https://lum1104.github.io/MER-Factory/)

---

## Related Work

### MiniGPT-v2

Our work builds upon MiniGPT-v2. If you use the MiniGPT-v2 components:

```bibtex
@article{chen2023minigptv2,
  title={MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning},
  author={Chen, Jun and Zhu, Deyao and Shen, Xiaoqian and Li, Xiang and Liu, Zechu and Zhang, Pengchuan and Krishnamoorthi, Raghuraman and Chandra, Vikas and Xiong, Yunyang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2310.09478},
  year={2023}
}
```

Website: [MiniGPT-v2](https://minigpt-v2.github.io/)

### AffectGPT

For emotion reasoning evaluation methodology:

```bibtex
@article{lian2023affectgpt,
  title={AffectGPT: Explainable Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Licai and Sun, Haiyang and Chen, Kang and Wen, Zhuofan and Gu, Hao and Tao, Jinming and Niu, Mingyu and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2306.15401},
  year={2023}
}
```

Website: [AffectGPT](https://github.com/zeroQiaoba/AffectGPT)

### LLaVA

For vision-language understanding:

```bibtex
@inproceedings{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle={NeurIPS},
  year={2023}
}
```

Website: [LLaVA](https://llava-vl.github.io/)

---

## Acknowledgements

We would like to acknowledge the following projects and datasets that made this work possible:

### Models and Frameworks

- **LLaMA-2**: Meta AI's large language model
- **MiniGPT-v2**: Vision-language multi-task learning framework
- **HuBERT**: Audio feature extraction
- **EVA**: Visual representation learning
- **MAE**: Masked autoencoders for visual learning
- **VideoMAE**: Video masked autoencoders

### Datasets

- **MER2023**: Multimodal Emotion Recognition Challenge 2023
- **MER2024**: Multimodal Emotion Recognition Challenge 2024
- **EMER**: Emotion reasoning evaluation dataset
- **DFEW**: Dynamic Facial Expression in the Wild

### Tools and Libraries

- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: NLP and multimodal models
- **Gradio**: Demo interface
- **OpenFace**: Facial action unit detection

---

## Using This Work

### For Academic Research

When citing this work in academic publications:

1. ✅ Cite the main NeurIPS 2024 paper
2. ✅ If using MERR dataset, acknowledge the dataset
3. ✅ If using Conv-Attention, cite the MER2024 paper
4. ✅ Acknowledge the base datasets (MER2023, etc.)

### For Commercial Use

Please review the [license](license.md) for commercial use restrictions.

### Attribution Example

```
This work uses Emotion-LLaMA (Cheng et al., NeurIPS 2024), a multimodal 
emotion recognition model with instruction tuning capabilities, trained 
on the MERR dataset for enhanced emotion reasoning.
```

---

## Star History

Support our project by giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=ZebangCheng/Emotion-LLaMA&type=Date)](https://www.star-history.com/#ZebangCheng/Emotion-LLaMA&Date)

---

## Contact for Citations

If you have questions about citing this work:

- **GitHub Issues**: [Ask a question](https://github.com/ZebangCheng/Emotion-LLaMA/issues)
- **Email**: Contact the corresponding author
- **Paper**: [Read the NeurIPS 2024 paper](https://arxiv.org/pdf/2406.11161)

---

## Next Steps

- Review the [license information](license.md)
- Explore the [main documentation](../)
- Visit our [GitHub repository](https://github.com/ZebangCheng/Emotion-LLaMA)

