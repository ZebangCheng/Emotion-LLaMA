---
layout: default
title: License
parent: About
nav_order: 2
---

# License Information
{: .no_toc }

Understanding the licensing terms for Emotion-LLaMA and its components.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Emotion-LLaMA is released under multiple licenses depending on the component:

- **Code**: BSD 3-Clause License
- **Model Weights**: Research-only license
- **MERR Dataset**: EULA for research purposes only
- **Documentation**: CC BY-NC 4.0

---

## Code License (BSD 3-Clause)

The Emotion-LLaMA codebase is licensed under the **BSD 3-Clause License**.

### BSD 3-Clause License

```
Copyright (c) 2024, Emotion-LLaMA Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**Full License**: [LICENSE.md](https://github.com/ZebangCheng/Emotion-LLaMA/blob/main/LICENSE.md)

---

## MiniGPT-4 License

Emotion-LLaMA is based on MiniGPT-4, which is also under the **BSD 3-Clause License**.

### What This Means

- ✅ You can use, modify, and distribute the code
- ✅ You can use it for commercial purposes
- ✅ You must include the license and copyright notice
- ❌ No warranty or liability from the authors

**Full License**: [LICENSE_MiniGPT4.md](https://github.com/ZebangCheng/Emotion-LLaMA/blob/main/LICENSE_MiniGPT4.md)

---

## MERR Dataset License (EULA)

The MERR dataset is based on MER2023 and is licensed under an **End User License Agreement (EULA)** for **research purposes only**.

### Key Restrictions

- ✅ **Research use**: Academic and non-commercial research
- ❌ **Commercial use**: Not permitted without explicit permission
- ❌ **Redistribution**: Cannot redistribute the raw videos
- ✅ **Annotations**: Can share annotation files with proper citation

### Usage Guidelines

1. **Apply for Access**: Request dataset access from [MER Challenge](http://merchallenge.cn/datasets)
2. **Acknowledge Source**: Cite both Emotion-LLaMA and MER2023
3. **Research Only**: Do not use for commercial applications
4. **No Redistribution**: Do not share raw video files

**Full License**: [LICENSE_EULA.md](https://github.com/ZebangCheng/Emotion-LLaMA/blob/main/LICENSE_EULA.md)

---

## Model Weights License

Pre-trained model weights for Emotion-LLaMA are released for **research purposes only**.

### Restrictions

- ✅ Academic research and education
- ✅ Non-commercial applications
- ❌ Commercial deployment without permission
- ❌ Redistribution on other platforms

### Base Model Licenses

Emotion-LLaMA uses components with their own licenses:

#### LLaMA-2

- **License**: [LLaMA 2 Community License](https://ai.meta.com/llama/license/)
- **Commercial Use**: Permitted under certain conditions
- **Restrictions**: See Meta's license for details

#### HuBERT, EVA, MAE, VideoMAE

- Various licenses depending on the specific model
- Generally permissive for research use
- Check individual model repositories for commercial use

---

## Documentation License (CC BY-NC 4.0)

This documentation is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

### What You Can Do

- ✅ **Share**: Copy and redistribute the documentation
- ✅ **Adapt**: Remix, transform, and build upon the documentation
- ✅ **Attribution**: Must give appropriate credit

### Conditions

- 🔒 **NonCommercial**: Cannot use for commercial purposes
- 📝 **Attribution**: Must provide a link to the license and indicate changes

**Full License**: [LICENSE_CC BY-NC 4.0.md](https://github.com/ZebangCheng/Emotion-LLaMA/blob/main/LICENSE_CC%20BY-NC%204.0%20.md)

---

## Commercial Use

### Academic and Research Use

Emotion-LLaMA is **freely available** for:
- Academic research
- Educational purposes
- Non-profit organizations
- Personal learning and experimentation

### Commercial Applications

For commercial use, you must:

1. **Review Component Licenses**: Check licenses of all components (LLaMA-2, datasets, etc.)
2. **Contact Authors**: Discuss commercial licensing
3. **Obtain Permission**: Get written permission for commercial deployment
4. **Alternative Datasets**: Use commercially-licensed datasets if needed

{: .warning }
> The MERR dataset is **NOT** available for commercial use under the current license.

---

## Third-Party Components

Emotion-LLaMA incorporates several third-party components, each with their own licenses:

| Component | License | Use |
|:----------|:--------|:----|
| PyTorch | BSD-style | Deep learning framework |
| Transformers | Apache 2.0 | NLP models |
| Gradio | Apache 2.0 | Demo interface |
| OpenFace | Apache 2.0 | Face analysis |
| LLaMA-2 | Custom | Language model |
| MER2023 | EULA | Dataset |

### Compliance

When using Emotion-LLaMA, ensure you comply with all third-party licenses.

---

## Attribution Requirements

### For Code

Include the BSD 3-Clause license text in your distribution.

### For Papers

Cite our NeurIPS 2024 paper (see [Citation](citation.md)).

### For Datasets

```
This work uses the MERR dataset from Emotion-LLaMA (Cheng et al., NeurIPS 2024),
which is based on MER2023 dataset. Used for research purposes only under EULA.
```

### For Models

```
This application uses Emotion-LLaMA (Cheng et al., NeurIPS 2024), a multimodal
emotion recognition model. Model weights used for research purposes only.
```

---

## Liability and Warranty

### Disclaimer

{: .warning }
> **NO WARRANTY**: This software is provided "as is" without warranty of any kind, 
> express or implied. The authors are not liable for any damages arising from the use 
> of this software.

### User Responsibility

Users are responsible for:
- ✅ Ensuring compliance with all licenses
- ✅ Proper use of the models and datasets
- ✅ Ethical use of emotion recognition technology
- ✅ Privacy and data protection compliance

---

## Frequently Asked Questions

### Can I use Emotion-LLaMA in my company?

For **research purposes** (e.g., R&D, internal testing), yes. For **commercial products**, you need permission from the authors and must comply with all component licenses, especially the MERR dataset EULA.

### Can I modify the code?

Yes, the BSD 3-Clause license allows modifications. You must retain the license notice.

### Can I share the MERR dataset with colleagues?

You can share the **annotation files** with proper citation. You **cannot** redistribute the raw video files - colleagues must apply for access themselves.

### Can I use this for a startup?

You need to:
1. Use a commercially-licensed dataset (not MERR)
2. Check LLaMA-2 commercial terms
3. Contact authors for guidance
4. Ensure compliance with all component licenses

### Can I deploy a public demo?

**Non-commercial demos** (e.g., research showcase, educational tool) are permitted. **Commercial demos** (e.g., paid service, revenue-generating app) require permission.

---

## Ethical Considerations

### Responsible Use

When using Emotion-LLaMA, consider:

- **Privacy**: Respect individuals' privacy in emotion analysis
- **Consent**: Obtain proper consent for analyzing individuals
- **Bias**: Be aware of potential biases in emotion recognition
- **Transparency**: Be transparent about using AI for emotion detection
- **Fairness**: Ensure fair treatment across different demographics

### Prohibited Uses

Do **NOT** use Emotion-LLaMA for:
- ❌ Surveillance without consent
- ❌ Discriminatory practices
- ❌ Manipulative applications
- ❌ Privacy-invasive systems
- ❌ Any unethical or illegal purposes

---

## Updates and Changes

Licenses may be updated over time. Always check the latest version:

- **Repository**: [https://github.com/ZebangCheng/Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA)
- **License Files**: Check the `LICENSE*.md` files in the repository

---

## Contact for Licensing

For licensing questions:

- **General Inquiries**: Open a [GitHub issue](https://github.com/ZebangCheng/Emotion-LLaMA/issues)
- **Commercial Licensing**: Contact the authors directly
- **Dataset Access**: Apply through [MER Challenge](http://merchallenge.cn/datasets)

---

## Summary

| Aspect | License | Commercial Use |
|:-------|:--------|:---------------|
| Code | BSD 3-Clause | ✅ Permitted |
| Model Weights | Research-only | ❌ Permission Required |
| MERR Dataset | EULA | ❌ Not Permitted |
| Documentation | CC BY-NC 4.0 | ❌ Not Permitted |

---

## Next Steps

- Read the [citation guide](citation.md) for proper attribution
- Review the full [license files](https://github.com/ZebangCheng/Emotion-LLaMA/tree/main)
- Check [ethical guidelines](#ethical-considerations) before deployment

