| Dataset | Granularity | Method | Venue (Year) | Acc | Macro-F1 | Notes |
|---|---|---|---:|---:|---:|---|
| Fakeddit | 2-way | BERT+EfficientNet | LREC 2020 | 0.8318 | — | Text+Image (BERT + EfficientNet); test acc |
| Fakeddit | 2-way | BERT+ResNet50 | LREC 2020 | 0.8909 | — | Text+Image (BERT text + ResNet50 image); test acc |
| Fakeddit | 2-way | BERT (text-only) | LREC 2020 | 0.8644 | — | Text only (BERT); test acc |
| Fakeddit | 2-way | BERT+VGG16 | LREC 2020 | 0.8699 | — | Text+Image (BERT + VGG16); test acc |
| Fakeddit | 2-way | EfficientNet (image-only) | LREC 2020 | 0.6087 | — | Image only (EfficientNet); test acc |
| Fakeddit | 2-way | InferSent+ResNet50 | LREC 2020 | 0.8891 | — | Text+Image (InferSent + ResNet50); test acc |
| Fakeddit | 2-way | InferSent (text-only) | LREC 2020 | 0.8631 | — | Text only (InferSent); test acc |
| Fakeddit | 2-way | ResNet50 (image-only) | LREC 2020 | 0.8070 | — | Image only (ResNet50); test acc |
| Fakeddit | 2-way | VGG16 (image-only) | LREC 2020 | 0.7376 | — | Image only (VGG16); test acc |
| Fakeddit | 2-way | MAGIC | arXiv 2024 | 0.9760 | — | author subset n≈3.1k; not directly comparable |
| Fakeddit | 3-way | BERT+EfficientNet | LREC 2020 | 0.8255 | — | Text+Image (BERT + EfficientNet); test acc |
| Fakeddit | 3-way | BERT+ResNet50 | LREC 2020 | 0.8890 | — | Text+Image (BERT text + ResNet50 image); test acc |
| Fakeddit | 3-way | BERT (text-only) | LREC 2020 | 0.8580 | — | Text only (BERT); test acc |
| Fakeddit | 3-way | BERT+VGG16 | LREC 2020 | 0.8655 | — | Text+Image (BERT + VGG16); test acc |
| Fakeddit | 3-way | EfficientNet (image-only) | LREC 2020 | 0.5828 | — | Image only (EfficientNet); test acc |
| Fakeddit | 3-way | InferSent+ResNet50 | LREC 2020 | 0.8863 | — | Text+Image (InferSent + ResNet50); test acc |
| Fakeddit | 3-way | InferSent (text-only) | LREC 2020 | 0.8570 | — | Text only (InferSent); test acc |
| Fakeddit | 3-way | ResNet50 (image-only) | LREC 2020 | 0.7988 | — | Image only (ResNet50); test acc |
| Fakeddit | 3-way | VGG16 (image-only) | LREC 2020 | 0.7293 | — | Image only (VGG16); test acc |
| Fakeddit | 6-way | BERT+EfficientNet | LREC 2020 | 0.7272 | — | Text+Image (BERT + EfficientNet); test acc |
| Fakeddit | 6-way | BERT+ResNet50 | LREC 2020 | 0.8588 | — | Text+Image (BERT text + ResNet50 image); test acc |
| Fakeddit | 6-way | BERT (text-only) | LREC 2020 | 0.7677 | — | Text only (BERT); test acc |
| Fakeddit | 6-way | BERT+VGG16 | LREC 2020 | 0.8208 | — | Text+Image (BERT + VGG16); test acc |
| Fakeddit | 6-way | EfficientNet (image-only) | LREC 2020 | 0.4153 | — | Image only (EfficientNet); test acc |
| Fakeddit | 6-way | InferSent+ResNet50 | LREC 2020 | 0.8526 | — | Text+Image (InferSent + ResNet50); test acc |
| Fakeddit | 6-way | InferSent (text-only) | LREC 2020 | 0.7666 | — | Text only (InferSent); test acc |
| Fakeddit | 6-way | ResNet50 (image-only) | LREC 2020 | 0.7549 | — | Image only (ResNet50); test acc |
| Fakeddit | 6-way | VGG16 (image-only) | LREC 2020 | 0.6516 | — | Image only (VGG16); test acc |
| GossipCop | binary | SAFE | PAKDD 2020 | 0.838 | 0.895 | — |
| MFND19 | 3-way | MAGIC | arXiv 2024 | 0.863 | — | Weibo 3-way (MFND); author numbers; not directly comparable to binary Weibo |
| NA | — | AMFB | 2019 content | — | — | Original paper |
| NA | — | DUCK | 2021 graph | — | — | Original paper |
| NA | — | MPFN | 2019 content | — | — | Original paper |
| NA | — | SpotFake | 2019 content | — | — | Original paper |
| NA | — | TI-CNN | 2018 content | — | — | Original paper |
| PHEME | 3-way | DAN-Tree | PeerJ CS 2023 | 0.845 | 0.830 | Veracity FR/TR/UR (3-class); conversation-level; not directly comparable to content-only |
| PolitiFact | binary | SAFE | PAKDD 2020 | 0.874 | 0.896 | — |
| TI-CNN-News | binary | TI-CNN | arXiv 2018 | — | — | F1 (not Macro-F1); reported on TI-CNN news dataset |
| Twitter | binary | EANN | KDD 2018 | 0.715 | 0.719 | content-level text+image; Table 2 |
| Twitter | binary | MPFN | IPM 2023 | 0.833 | — | Acc from paper; Weibo to-be-added after verification |
| Twitter | binary | SEMI-FND | JEIT 2022 | 0.8580 | — | Semi-supervised multimodal; Twitter accuracy |
| Twitter15 | binary | GCAN | ACL 2020 | 0.8767 | 0.8250 | conversation graph + user/retweet features |
| Twitter16 | binary | GCAN | ACL 2020 | 0.9084 | 0.7593 | conversation graph + user/retweet features |
| WeChat | binary | WeFEND (w/ RL) | AAAI 2020 | 0.824 | — | Table 2; weak supervision with RL selector (headline text only) |
| WeChat | binary | WeFEND (ablated) | AAAI 2020 | 0.807 | — | Ablation variant reported alongside WeFEND; accuracy on WeChat |
| WeFEND | binary | DMPD(student) | — 2025 | — | — | To be filled from our experiments |
| Weibo | binary | EANN | KDD 2018 | 0.827 | 0.829 | content-level text+image; Table 2 |
| Weibo | binary | FMFN | JEIT 2022 | 0.885 | — | Multi-level self-fusing multimodal baseline; Weibo accuracy |
| Weibo | binary | SEMI-FND | JEIT 2022 | 0.8683 | — | Semi-supervised multimodal; Weibo accuracy |
| official | 0.8890 | GACN | 2021 graph | — | — | Original paper |
| official | 0.9000 | GACN | 2021 graph | — | — | Original paper |
