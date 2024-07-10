This is a repository of the segmentation method and graph method. In the segmentation part, the instance, semantic, panoptic and referring segmentation are included. In the graph part, the representation learning, clustering and so on methods are included. We will update continue...


------------

- [Instance, Semantic and Panoptic Segmentation](#instance--semantic-and-panoptic-segmentation)
- [Graph Representation and Cluster](#graph-representation-and-cluster)
- [Referring Segmentation](#referring-segmentation)


## Instance, Semantic and Panoptic Segmentation

|   |   |   |   |
| :------------: | :------------: | :------------: | :------------: |
| ECCV2024 | [Semantic-aware SAM for Point-Prompted Instance Segmentation](https://arxiv.org/abs/2312.15895) | [code](https://github.com/zhaoyangwei123/SAPNet) | √√√ |
| ECCV2022 | [Point-to-Box Network for Accurate Object Detection via Single Point Supervision](https://arxiv.org/abs/2207.06827) | [code](https://github.com/ucas-vg/P2BNet) | √√√ |
| ECCV2024 | [Semantic-SAM: Segment and Recognize Anything at Any Granularity](https://arxiv.org/pdf/2307.04767.pdf) | [code](https://github.com/UX-Decoder/Semantic-SAM) | √√√ |
| ECCV2024 | [Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively](https://arxiv.org/abs/2401.02955) | [code](https://github.com/HarborYuan/ovsam) | √√√ |
| arxiv | [Segment Anything without Supervision](http://arxiv.org/abs/2406.20081) | [code](https://github.com/frank-xwang/UnSAM) | √√√ |
| CVPR024 | [Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion](https://arxiv.org/abs/2308.12469) | [code](https://sites.google.com/view/diffseg/home) | √ |
| CVPR024 | [CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor](https://arxiv.org/abs/2312.07661) | [code](https://torrvision.com/clip_as_rnn/) | √√ |
| CVPR024 | [PEM: Prototype-based Efficient MaskFormer for Image Segmentation](https://arxiv.org/abs/2402.19422) | [code](https://github.com/NiccoloCavagnero/PEM) |  |
| CVPR024 | Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation | [code](https://github.com/Jittor/JSeg) |  |
| CVPR024 | [Clustering Propagation for Universal Medical Image Segmentation](https://arxiv.org/abs/2403.16646) | [code](https://github.com/dyh127/S2VNet) | √ |
| CVPR024 | [Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2402.18467) | [code](https://github.com/zwyang6/SeCo.git) |  |
| CVPR024 | [Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation](https://arxiv.org/abs/2311.18363) | [code](https://github.com/Chen-Ziyang/VPTTA) |  |
| CVPR024 | [Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2405.00378) | [code](ttps://github.com/chyupc/ABD) |  |
| CVPR024 | [Training Like a Medical Resident: Context-Prior Learning Toward Universal Medical Image Segmentation](https://arxiv.org/abs/2306.02416) | [code](https://github.com/yhygao/universal-medical-image-segmentationA) | √ |
| CVPR024 | From SAM to CAMs: Exploring Segment Anything Model for Weakly Supervised Semantic Segmentation] | [code](https://github.com/sangrockEG/S2C) |  |
| CVPR024 | [Frequency-Adaptive Dilated Convolution for Semantic Segmentation](https://arxiv.org/abs/2403.05369) | [code](https://github.com/ying-fu/FADC) |  |
| CVPR024 | [Tyche: Stochastic In-Context Learning for Medical Image Segmentation](https://arxiv.org/abs/2401.13650) | [code](https://tyche.csail.mit.edu/) | √ |
| CVPR024 | [EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation](https://arxiv.org/abs/2405.06880) | [code](https://github.com/SLDGroup/EMCAD) | √ |
| CVPR024 | [ASAM: Boosting Segment Anything Model with Adversarial Tuning](https://arxiv.org/abs/2405.00256) | [code](https://asam2024.github.io/) |  |
| CVPR024 | [Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2312.04265) | [code](https://github.com/w1oves/Rein.git) |  |
| CVPR024 | [Open-Vocabulary Semantic Segmentation with Image Embedding Balancing](https://arxiv.org/abs/2406.09829) | [code](https://github.com/slonetime/EBSeg) |  |
| CVPR024 | [Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance](https://arxiv.org/abs/2312.10671) | [code](https://open3dis.github.io/) |  |
| CVPR024 | [Contextrast: Contextual Contrastive Learning for Semantic Segmentation](https://arxiv.org/abs/2404.10633) | None |  |
| CVPR024 | [Image-to-Image Matching via Foundation Models: A New Perspective for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2404.00262) | None |  |
| CVPR024 | Incremental Nuclei Segmentation from Histopathological Images via Future-class Awareness and Compatibility-inspired Distillation | [code](https://github.com/why19991/InSeg) | √√√ |
| CVPR024 | [ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation with Plain Vision Transformers](https://arxiv.org/abs/2406.09936) | [code](https://tue-mps.github.io/ALGM) |  |
| CVPR024 | [CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2303.11797) | None |  |
| CVPR024 | [Towards the Uncharted: Density-Descending Feature Perturbation for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/2403.06462) | [code](https://github.com/Gavinwxy/DDFP) |  |
| CVPR024 | [ZePT: Zero-Shot Pan-Tumor Segmentation via Query-Disentangling and Self-Prompting](https://arxiv.org/abs/2312.04964) | [code](https://github.com/Yankai96/ZePT) | √√√ |
| CVPR024 | [Rethinking Prior Information Generation with CLIP for Few-Shot Segmentation](https://arxiv.org/abs/2405.08458) | [code](https://github.com/vangjin/PI-CLIP) | √√√ |
| CVPR024 | [Unsupervised Universal Image Segmentation](https://arxiv.org/abs/2312.17243) | [code](https://github.com/u2seg/U2Seg) | √√√ |
| CVPR024 | [Open-Vocabulary Attention Maps with Token Optimization for Semantic Segmentation in Diffusion Models](https://arxiv.org/abs/2403.14291) | [code](https://github.com/vpulab/ovam) |  |
| CVPR024 | Edge-Aware 3D Instance Segmentation Network with Intelligent Semantic Prior | [code](https://kuai-lab.github.io/ease2024) |  |
| CVPR024 | Unlocking the Potential of Pre-trained Vision Transformers for Few-Shot Semantic Segmentation through Relationship Descriptors | [code](https://github.com/ZiqinZhou66/FewSegwithRD.git) |  |
| CVPR024 | Mudslide: A Universal Nuclear Instance Segmentation Method | None | √ |
| CVPR024 | Class Tokens Infusion for Weakly Supervised Semantic Segmentation | [code](https://github.com/yoon307/CTI) |  |
| CVPR024 | [Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling](https://arxiv.org/abs/2309.12378) | [code]() | √√√ |
| CVPR024 | [ToNNO: Tomographic Reconstruction of a Neural Network’s Output for Weakly Supervised Segmentation of 3D Medical Images](https://arxiv.org/abs/2404.13103) | None |  |
| CVPR024 | [Domain-Rectifying Adapter for Cross-Domain Few-Shot Segmentation](https://arxiv.org/abs/2404.10322) | [code](https://github.com/Matt-Su/DR-Adapter) |  |
| CVPR024 | Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation | [code](https://github.com/dogehhh/ReCLIP) |  |
| CVPR024 | [EAGLE: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation](https://arxiv.org/abs/2403.01482) | [code](https://micv-yonsei.github.io/eagle2024/) |  |
| CVPR024 | [MaskClustering: View Consensus based Mask Graph Clustering for Open-Vocabulary 3D Instance Segmentation](https://arxiv.org/abs/2401.07745) | [code](https://pku-epic.github.io/MaskClustering) |  |
| CVPR024 | Open-Vocabulary 3D Semantic Segmentation with Foundation Models | None |  |
| CVPR024 | [AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2403.01818) | [code](https://github.com/xmed-lab/AllSpark) | √√√ |
| CVPR024 | [Collaborating Foundation Models for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2312.09788) | [code](https://github.com/yasserben/CLOUDS) | √√√ |
| CVPR024 | [SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2311.15537) | [code](https://github.com/xb534/SED) |  |
| CVPR024 | [Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2406.11189) | [code](https://github.com/zbf1991/WeCLIP) | √√√ |
| CVPR024 | PSDPM: Prototype-based Secondary Discriminative Pixels Mining for Weakly Supervised Semantic Segmentation | [code](https://github.com/xinqiaozhao/PSDPM) |  |
| CVPR024 | [Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation](https://arxiv.org/abs/2402.17614) | [code](https://github.com/Vision-Kek/ABCDFSS) |  |
| CVPR024 | [USE: Universal Segment Embeddings for Open-Vocabulary Image Segmentation](https://arxiv.org/abs/2406.05271) | None |  |
| CVPR024 | [Constructing and Exploring Intermediate Domains in Mixed Domain Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/2404.08951) | [code](https://github.com/MQinghe/MiDSS) |  |
| CVPR024 | [Hunting Attributes: Context Prototype-Aware Learning for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2403.07630) | [code](https://github.com/Barrett-python/CPAL) |  |
| CVPR024 | [One-Prompt to Segment All Medical Images](https://arxiv.org/abs/2305.10300) | [code](https://github.com/KidsWithTokens/one-prompt) | √√√ |
| CVPR024 | [DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2403.11184) | [code](https://github.com/Wu0409/DuPL) |  |
| CVPR024 | RankMatch: Exploring the Better Consistency Regularization for Semi-supervised Semantic Segmentation |None |  |
| CVPR024 | [Open-World Semantic Segmentation Including Class Similarity](https://arxiv.org/abs/2403.07532) | [code](https://github.com/PRBonn/ContMAV) |  |
| CVPR024 | [Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincar´e Ball](https://arxiv.org/abs/2404.03778) | [code](https://github.com/tum-vision/hierahyp) |  |
| CVPR024 | [Diversified and Personalized Multi-rater Medical Image Segmentation](https://arxiv.org/abs/2403.13417) | [code](https://github.com/ycwu1997/D-Persona) | √ |
| CVPR024 | [MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling](https://arxiv.org/abs/2303.09373) | [code](https://github.com/XuzheZ/MAPSeg/) |  |
| CVPR024 | [Image-Text Co-Decomposition for Text-Supervised Semantic Segmentation](https://arxiv.org/abs/2404.04231) | [code](https://github.com/072jiajia/image-text-co-decomposition) |  |
| CVPR024 | [AlignSAM: Aligning Segment Anything Model to Open Context via Reinforcement Learning](https://arxiv.org/abs/2406.00480) | [code](https://github.com/Duojun-Huang/AlignSAM-CVPR2024) |  |
| CVPR024 | [UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes](https://arxiv.org/abs/2303.14541) | [code](https://rozdavid.github.io/unscene3d) |  |
| CVPR024 | [Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations](https://arxiv.org/abs/2406.10114) | [code](https://tue-mps.github.io/tapps/) |  |
| CVPR024 | [Semantic-aware SAM for Point-Prompted Instance Segmentation](https://arxiv.org/abs/2312.15895) | [code](https://github.com/zhaoyangwei123/SAPNet) |  |
| CVPR024 | [RobustSAM: Segment Anything Robustly on Degraded Images](https://arxiv.org/abs/2406.09627) | [code](https://robustsam.github.io/) | √ |  
| CVPR024 | [SAI3D: Segment Any Instance in 3D Scenes](https://arxiv.org/abs/2312.11557) | [code](https://yd-yin.github.io/SAI3D) |  |  
| CVPR024 | [CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2306.04300) | [code](https://github.com/BBBBchan/CorrMatch) |  |  
| CVPR024 | [Open-Set Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2405.19899) | [code](https://github.com/KHU-AGI/BUS) | √ |  
| CVPR024 | [BSNet: Box-Supervised Simulation-assisted Mean Teacher for 3D Instance](https://arxiv.org/abs/2403.15019) | [code](https://github.com/peoplelu/BSNet) |  |
| CVPR024 | PH-Net: Semi-Supervised Breast Lesion Segmentation via Patch-wise Hardness | [code](https://github.com/jjjsyyy/PH-Net) | √√ |
| CVPR024 | [Extreme Point Supervised Instance Segmentation](https://arxiv.org/abs/2405.20729) | [code](https://github.com/xingyizhou/ExtremeNet) |  |
| CVPR024 | Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation | [code](https://github.com/Coradlut/Bi-JROS) |  |
| CVPR024 | [Teeth-SEG: An Efficient Instance Segmentation Framework for Orthodontic Treatment based on Multi-Scale Aggregation and Anthropic Prior Knowledge](https://arxiv.org/abs/2404.01013) | [code](github.io/TeethSEG/) | √√ |
| CVPR024 | [EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything](https://arxiv.org/abs/2312.00863) | [code](https://yformer.github.io/efficient-sam/) | √ |
| CVPR024 | [Open-Vocabulary Segmentation with Semantic-Assisted Calibration](https://arxiv.org/abs/2312.04089) | [code](https://github.com/yongliu20/SCAN) |  |
| CVPR024 | [MFP: Making Full Use of Probability Maps for Interactive Image Segmentation](https://arxiv.org/abs/2404.18448) | [code](https://github.com/cwlee00/MFP) |  |
| CVPR024 | [GoodSAM: Bridging Domain and Capacity Gaps via Segment Anything Model for Distortion-aware Panoramic Semantic Segmentation](https://arxiv.org/abs/2403.16370) | [code](https://vlislab22.github.io/GoodSAM/) |  |
| CVPR024 | [Guided Slot Attention for Unsupervised Video Object Segmentation](https://arxiv.org/abs/2303.08314) | [code](https://github.com/Hydragon516/GSANet) |  |
| WACV2024 | [AnyStar: Domain randomized universal star-convex 3D instance segmentation](https://arxiv.org/abs/2307.07044) | [code](https://github.com/neel-dey/AnyStar) | √√√ |
| WACV2024 | [Revisiting Token Pruning for Object Detection and Instance Segmentation](https://arxiv.org/abs/2306.07050) | [code](https://github.com/uzh-rpg/svit/) |   |
| WACV2024 | [BPKD: Boundary Privileged Knowledge Distillation for Semantic Segmentation](https://arxiv.org/abs/2306.08075) | [code](https://github.com/AkideLiu/BPKD) |   |
| WACV2024 | [Guided Distillation for Semi-Supervised Instance Segmentation](https://arxiv.org/abs/2308.02668) | [code](https://github.com/facebookresearch/GuidedDistillation) |   |
| ICCV2023 | [SegPrompt: Boosting Open-world Segmentation via Category-level Prompt Learning](https://arxiv.org/abs/2308.06531) | [code](https://github.com/aim-uofa/SegPrompt) |   |
| ICCV2023 | [SegRCDB: Semantic Segmentation via Formula-Driven Supervised Learning](https://arxiv.org/abs/2309.17083) | [code](https://github.com/dahlian00/SegRCDB) |   |
| ICCV2023 | [Exploring Transformers for Open-world Instance Segmentation](https://arxiv.org/abs/2308.04206) | [code]() |   |
| ICCV2023 | [Class-incremental Continual Learning for Instance Segmentation with Image-level Weak Supervision](https://openaccess.thecvf.com/content/ICCV2023/papers/Hsieh_Class-incremental_Continual_Learning_for_Instance_Segmentation_with_Image-level_Weak_Supervision_ICCV_2023_paper.pdf) | [code](https://github.com/AI-Application-and-Integration-Lab/CL4WSIS) |  |
| ICCV2023  | [Treating Pseudo-labels Generation as Image Matting for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Treating_Pseudo-labels_Generation_as_Image_Matting_for_Weakly_Supervised_Semantic_ICCV_2023_paper.pdf)  | None |  |
| ICCV2023  | [Probabilistic Modeling of Inter- and Intra-observer Variability in Medical Image Segmentation](https://arxiv.org/abs/2307.11397)  | [code](https://github.com/arneschmidt/pionono_segmentation) | √√√ |
| ICCV2023  | [Learning Cross-Representation Affinity Consistency for Sparsely Supervised Biomedical Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Learning_Cross-Representation_Affinity_Consistency_for_Sparsely_Supervised_Biomedical_Instance_Segmentation_ICCV_2023_paper.pdf) | [code](https://github.com/liuxy1103/CRAC) | √√ |
| ICCV2023  | [MSI: Maximize Support-Set Information for Few-Shot Segmentation](https://arxiv.org/abs/2212.04673) | [code](https://github.com/moonsh/MSI-Maximize-Support-Set-Information) | √√ |
| ICCV2023  | [TopoSeg: Topology-Aware Nuclear Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/He_TopoSeg_Topology-Aware_Nuclear_Instance_Segmentation_ICCV_2023_paper.pdf) | [code](https://github.com/hhlisme/toposeg) | √√√ |
| ICCV2023  | [MemorySeg: Online LiDAR Semantic Segmentation with a Latent Memory](https://arxiv.org/abs/2311.01556) | [code](https://waabi.ai/research/memoryseg) |  |
| ICCV2023  | [Informative Data Mining for One-shot Cross-Domain Semantic Segmentation](https://arxiv.org/abs/2309.14241) | [code](https://github.com/yxiwang/IDM) |  |
| ICCV2023  | [Segmenting Known Objects and Unseen Unknowns without Prior Knowledge](https://arxiv.org/abs/2209.05407) | [code](https://holisticseg.github.io) |  |
| ICCV2023  | [RbA: Segmenting Unknown Regions Rejected by All](https://arxiv.org/abs/2211.14293) | [code](https://kuis-ai.github.io/RbA) | √ |
| ICCV2023  | [Segment Anything](https://arxiv.org/abs/2304.02643) | [code](https://segment-anything.com) |  |
| ICCV2023  | [SimpleClick: Interactive Image Segmentation with Simple Vision Transformers](https://arxiv.org/abs/2210.11006) | [code](https://github.com/uncbiag/SimpleClick) |  |
| ICCV2023  | [Texture Learning Domain Randomization for Domain Generalized Segmentation](https://arxiv.org/abs/2303.11546) | [code](https://github.com/ssssshwan/TLDR) |  |
| ICCV2023  | [Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images](https://arxiv.org/abs/2310.08501) | [code](https://github.com/funkelab/cellulus) | √√√ |
| ICCV2023  | [XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.pdf) | [code](https://github.com/Yanfeng-Zhou/XNet) | √√√ |
| ICCV2023  | [CoinSeg: Contrast Inter- and Intra- Class Representations for Incremental Segmentation](https://arxiv.org/abs/2310.06368) | [code](https://github.com/zkzhang98/CoinSeg) |  |
| ICCV2023  | [Open-Vocabulary Semantic Segmentation with Decoupled One-Pass Network](https://arxiv.org/abs/2304.01198) | [code](https://github.com/CongHan0808/DeOP.git) |  |
| ICCV2023  | [BoxSnake: Polygonal Instance Segmentation with Box Supervision](https://arxiv.org/abs/2303.11630) | [code](https://github.com/Yangr116/BoxSnake) |  |
| ICCV2023  | [UniverSeg: Universal Medical Image Segmentation](https://arxiv.org/abs/2304.06131) | [code](https://universeg.csail.mit.edu) | √√√ |
| ICCV2023  | [Enhancing Modality-Agnostic Representations via Meta-learning for Brain Tumor Segmentation](https://arxiv.org/abs/2302.04308) | None | √√√ |
| ICCV2023  | [CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_CauSSL_Causality-inspired_Semi-supervised_Learning_for_Medical_Image_Segmentation_ICCV_2023_paper.pdf) | [code](https://github.com/JuzhengMiao/CauSSL) |  |
| ICCV2023  | [MARS: Model-agnostic Biased Object Removal without Additional Supervision for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2304.09913) | [code](https://github.com/shjo-april/MARS) |  |
| ICCV2023  | [Learning Neural Eigenfunctions for Unsupervised Semantic Segmentation](https://arxiv.org/abs/2304.02841) | [code](https://github.com/thudzj/NeuralEigenfunctionSegmentor) |  | 
| ICCV2023  | [Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2307.09755)  | [code](https://github.com/WangChangqi98/CSS) |  |
| ICCV2023  | [Global Knowledge Calibration for Fast Open-Vocabulary Segmentation](https://arxiv.org/abs/2303.09181)  | [code](https://github.com/yongliu20/GKC) |  |
| ICCV2023  | [Open-vocabulary Panoptic Segmentation with Embedding Modulation](https://arxiv.org/abs/2303.11324)  | [code](https://opsnet-page.github.io/) |  |
| ICCV2023  | [MasQCLIP for Open-Vocabulary Universal Image Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_MasQCLIP_for_Open-Vocabulary_Universal_Image_Segmentation_ICCV_2023_paper.pdf)  | [code](https://masqclip.github.io/) |  |
| ICCV2023  | [WaterMask: Instance Segmentation for Underwater Imagery](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf)  | [code](https://github.com/LiamLian0727/WaterMask) |  |
| ICCV2023  | [CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection](https://arxiv.org/abs/2301.00785)  | [code](https://github.com/ljwztc/CLIP-Driven-Universal-Model) | √√√ |
| ICCV2023  | [SegGPT: Towards Segmenting Everything In Context](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf)  | [code](https://github.com/baaivision/Painter) |  |
| ICCV2023  | [Towards Deeply Unified Depth-aware Panoptic Segmentation with Bi-directional Guidance Learning](https://arxiv.org/abs/2307.14786)  | [code](https://github.com/jwh97nn/DeepDPS) |  |
| ICCV2023  | [DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models](https://arxiv.org/abs/2303.11681)  | [code](https://github.com/weijiawu/DiffuMask) | √ |
| ICCV2023  | [Scratch Each Other’s Back: Incomplete Multi-modal Brain Tumor Segmentation Via Category Aware Group Self-Support Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Qiu_Scratch_Each_Others_Back_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_via_ICCV_2023_paper.pdf)  | [code](https://github.com/qysgithubopen/GSS) | √ |
| ICCV2023  | [EDAPS: Enhanced Domain-Adaptive Panoptic Segmentation](https://arxiv.org/abs/2304.14291)  | [code](https://github.com/susaha/edaps) |  |
| ICCV2023  | [Exploring Open-Vocabulary Semantic Segmentation from CLIP Vision Encoder Distillation Only](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Exploring_Open-Vocabulary_Semantic_Segmentation_from_CLIP_Vision_Encoder_Distillation_Only_ICCV_2023_paper.pdf)  | [code](https://github.com/facebookresearch/ZeroSeg) |  |
| ICCV2023  | [Pseudo-label Alignment for Semi-supervised Instance Segmentation](https://arxiv.org/abs/2308.05359)  | [code](https://github.com/hujiecpp/PAIS) |  |
| ICCV2023  | [LD-ZNet: A Latent Diffusion Approach for Text-Based Image Segmentation](https://arxiv.org/abs/2303.12343)  | [code](https://koutilya-pnvr.github.io/LD-ZNet/) |  |
| ICCV2023  | [Enhanced Soft Label for Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Enhanced_Soft_Label_for_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.pdf)  | [code](https://github.com/unrealMJ/ESL) |  |
| ICCV2023  | [Stochastic Segmentation with Conditional Categorical Diffusion Models](https://arxiv.org/abs/2303.08888)  | [code](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation) |  |
| CVPR2022  | [Deep Hierarchical Semantic Segmentation](https://arxiv.org/abs/2203.14335)  | [code](https://github.com/0liliulei/HieraSeg) |  |
| CVPR2022  | [Rethinking Semantic Segmentation: A Prototype View](https://arxiv.org/abs/2203.15102) | [code](https://github.com/tfzhou/ProtoSeg)  |  |
| CVPR2022  | [Sparse Instance Activation for Real-Time Instance Segmentation](https://arxiv.org/abs/2203.12827) | [code](https://github.com/hustvl/SparseInst) |  |
| CVPR2022 | [Instance Segmentation with Mask-supervised Polygonal Boundary Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Lazarow_Instance_Segmentation_With_Mask-Supervised_Polygonal_Boundary_Transformers_CVPR_2022_paper.pdf) | [code](https://github.com/mlpc-ucsd/BoundaryFormer) |  |
| CVPR2022 | [CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_CMT-DeepLab_Clustering_Mask_Transformers_for_Panoptic_Segmentation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Panoptic_SegFormer_Delving_Deeper_Into_Panoptic_Segmentation_With_Transformers_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_PanopticDepth_A_Unified_Framework_for_Depth-Aware_Panoptic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/NaiyuGao/PanopticDepth) |  |
| CVPR2022 | [Mask Transfiner for High-Quality Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/SysCV/transfiner) |  |
| CVPR2022 | [Unsupervised Hierarchical Semantic Segmentation With Multiview Cosegmentation and Clustering Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Unsupervised_Hierarchical_Semantic_Segmentation_With_Multiview_Cosegmentation_and_Clustering_Transformers_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Coarse-To-Fine Feature Mining for Video Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Coarse-To-Fine_Feature_Mining_for_Video_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/GuoleiSun/VSS-CFFM) |  |
| CVPR2022 | [Sparse Object-Level Supervision for Instance Segmentation With Pixel Embeddings](https://openaccess.thecvf.com/content/CVPR2022/papers/Wolny_Sparse_Object-Level_Supervision_for_Instance_Segmentation_With_Pixel_Embeddings_CVPR_2022_paper.pdf) | [code](https://github.com/kreshuklab/spoco) |  |
| CVPR2022 | [Segment-Fusion: Hierarchical Context Fusion for Robust 3D Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Thyagharajan_Segment-Fusion_Hierarchical_Context_Fusion_for_Robust_3D_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Pin the Memory: Learning To Generalize Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/Genie-Kim/PintheMemory) |  |
| CVPR2022 | [SoftGroup for 3D Instance Segmentation on Point Clouds](https://openaccess.thecvf.com/content/CVPR2022/papers/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.pdf) | [code](https://github.com/thangvubk/SoftGroup) |  |
| CVPR2022 | [SharpContour: A Contour-based Boundary Refinement Approach for Efficient and Accurate Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_SharpContour_A_Contour-Based_Boundary_Refinement_Approach_for_Efficient_and_Accurate_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Beyond_Semantic_to_Instance_Segmentation_Weakly-Supervised_Instance_Segmentation_via_Semantic_CVPR_2022_paper.pdf) | [code](https://github.com/clovaai/BESTIE) |  |
| CVPR2022 | [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Semi-Supervised_Semantic_Segmentation_Using_Unreliable_Pseudo-Labels_CVPR_2022_paper.pdf) | [code](https://haochen-wang409.github.io/U2PL) |  |
| CVPR2022 | [Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization](https://openaccess.thecvf.com/content/CVPR2022/papers/Melas-Kyriazi_Deep_Spectral_Methods_A_Surprisingly_Strong_Baseline_for_Unsupervised_Semantic_CVPR_2022_paper.pdf) | [code](https://lukemelas.github.io/deep-spectral-segmentation/) |  |
| CVPR2022 | [Self-Supervised Learning of Object Parts for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/MkuuWaUjinga/leopart) |  |
| CVPR2022 | [Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Multi-Scale_High-Resolution_Vision_Transformer_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/facebookresearch/HRViT) |  |
| CVPR2022 | [C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_C-CAM_Causal_CAM_for_Weakly_Supervised_Semantic_Segmentation_on_Medical_CVPR_2022_paper.pdf) | [code](https://github.com/Tian-lab/C-CAM) |  |
| CVPR2022 | [Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Dynamic_Prototype_Convolution_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [ContrastMask: Contrastive Learning to Segment Every Thing](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ContrastMask_Contrastive_Learning_To_Segment_Every_Thing_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Generalized Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Tian_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/dvlab-research/GFS-Seg) |  |
| CVPR2022 | [Decoupling Zero-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Decoupling_Zero-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/dingjiansw101/ZegFormer) |  |
| CVPR2022 | [TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_TopFormer_Token_Pyramid_Transformer_for_Mobile_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/hustvl/TopFormer) |  |
| CVPR2022 | [Cross-Image Relational Knowledge Distillation for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.pdf) | [code](https://github.com/VITA-Group/CADTransformer) |  |
| CVPR2022 | [GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [TubeFormer-DeepLab: Video Mask Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_TubeFormer-DeepLab_Video_Mask_Transformer_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Class-Balanced_Pixel-Level_Self-Labeling_for_Domain_Adaptive_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/lslrh/CPSL) |  |
| CVPR2022 | [Learning Non-target Knowledge for Few-shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_Non-Target_Knowledge_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/LIUYUANWEI98/NERTNet) |  |
| CVPR2022 | [SemAffiNet: Semantic-Affine Transformation for Point Cloud Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_SemAffiNet_Semantic-Affine_Transformation_for_Point_Cloud_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/wangzy22/SemAffiNet) |  |
| CVPR2022 | [Real-Time, Accurate, and Consistent Video Semantic Segmentation via Unsupervised Adaptation and Cross-Unit Deployment on Mobile Device](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Real-Time_Accurate_and_Consistent_Video_Semantic_Segmentation_via_Unsupervised_Adaptation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Tree_Energy_Loss_Towards_Sparsely_Annotated_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/megvii-research/TreeEnergyLoss) |  |
| CVPR2022 | [Amodal Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Mohan_Amodal_Panoptic_Segmentation_CVPR_2022_paper.pdf) | [code](http://amodal-panoptic.cs.uni-freiburg.de) |  |
| CVPR2022 | [Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Ru_Learning_Affinity_From_Attention_End-to-End_Weakly-Supervised_Semantic_Segmentation_With_Transformers_CVPR_2022_paper.pdf) | [code](https://github.com/rulixiang/afa) |  |
| CVPR2022 | [Partial Class Activation Attention for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Partial_Class_Activation_Attention_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/lsa1997/PCAA) |  |
| CVPR2022 | [Weakly Supervised Semantic Segmentation using Out-of-Distribution Data](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Weakly_Supervised_Semantic_Segmentation_Using_Out-of-Distribution_Data_CVPR_2022_paper.pdf) | [code](https://github.com/naver-ai/w-ood) |  |
| CVPR2022 | [Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Phan_Class_Similarity_Weighted_Knowledge_Distillation_for_Continual_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |  |
| CVPR2022 | [Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Bending_Reality_Distortion-Aware_Transformers_for_Adapting_to_Panoramic_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/jamycheung/Trans4PASS) |  |
| CVPR2022 | [Towards Noiseless Object Contours for Weakly Supervised Semantic](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Noiseless_Object_Contours_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/BraveGroup/SANCE) |  |
| CVPR2022 | [L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.pdf) | [code](https://github.com/PengtaoJiang/L2G) |  |
| CVPR2022| [DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf)| [code](https://github.com/lhoyer/DAFormer)|  |
| CVPR2022 | [Stratified Transformer for 3D Point Cloud Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)| [code](https://github.com/dvlab-research/StratifiedTransformer)|  |
| CVPR2022 | [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupViT_Semantic_Segmentation_Emerges_From_Text_Supervision_CVPR_2022_paper.pdf) | [code](https://github.com/NVlabs/GroupViT) | enlightened |
| WACV2022 | [UNETR: Transformers for 3D Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) | [code](https://monai.io/research/unetr) |  |
| WACV2022 | [Pixel-Level Bijective Matching for Video Object Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf) | [code](https://github.com/suhwan-cho/BMVOS) |  |
| WACV2022 | [A Pixel-Level Meta-Learner for Weakly Supervised Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Lee_A_Pixel-Level_Meta-Learner_for_Weakly_Supervised_Few-Shot_Semantic_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Plugging Self-Supervised Monocular Depth into Unsupervised Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Cardace_Plugging_Self-Supervised_Monocular_Depth_Into_Unsupervised_Domain_Adaptation_for_Semantic_WACV_2022_paper.pdf) | [code](https://github.com/CVLAB-Unibo/d4-dbst) |  |
| WACV2022 | [Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Sun_Inferring_the_Class_Conditional_Response_Map_for_Weakly_Supervised_Semantic_WACV_2022_paper.pdf) | [code](https://github.com/weixuansun/InferCam) |  |
| WACV2022 | [Pixel-by-Pixel Cross-Domain Alignment for Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Tavera_Pixel-by-Pixel_Cross-Domain_Alignment_for_Few-Shot_Semantic_Segmentation_WACV_2022_paper.pdf) | [code](https://github.com/taveraantonio/PixDA) |  |
| WACV2022 | [Maximizing Cosine Similarity Between Spatial Features for Unsupervised Domain Adaptation in Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Chung_Maximizing_Cosine_Similarity_Between_Spatial_Features_for_Unsupervised_Domain_Adaptation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Self-Supervised Generative Style Transfer for One-Shot Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Tomar_Self-Supervised_Generative_Style_Transfer_for_One-Shot_Medical_Image_Segmentation_WACV_2022_paper.pdf) | [code](https://github.com/devavratTomar/SST/) |  |
| WACV2022 | [AFTer-UNet: Axial Fusion Transformer UNet for Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Single-shot Path Integrated Panoptic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Hwang_Single-Shot_Path_Integrated_Panoptic_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Time-Space Transformers for Video Panoptic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Petrovai_Time-Space_Transformers_for_Video_Panoptic_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [FASSST: Fast Attention Based Single-Stage Segmentation Net for Real-Time Instance Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Cheng_FASSST_Fast_Attention_Based_Single-Stage_Segmentation_Net_for_Real-Time_Instance_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Shallow Features Guide Unsupervised Domain Adaptation for Semantic Segmentation at Class Boundaries](https://openaccess.thecvf.com/content/WACV2022/papers/Cardace_Shallow_Features_Guide_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_at_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Co-Net: A Collaborative Region-Contour-Driven Network for Fine-to-Finer Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Liu_Co-Net_A_Collaborative_Region-Contour-Driven_Network_for_Fine-to-Finer_Medical_Image_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [T-Net: A Resource-Constrained Tiny Convolutional Neural Network for Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Khan_T-Net_A_Resource-Constrained_Tiny_Convolutional_Neural_Network_for_Medical_Image_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Multi-Domain Incremental Learning for Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Garg_Multi-Domain_Incremental_Learning_for_Semantic_Segmentation_WACV_2022_paper.pdf) | [code](https://github.com/prachigarg23/MDIL-SS) |  |
| WACV2022 | [Active Learning for Improved Semi-Supervised Semantic Segmentation in Satellite Images](https://openaccess.thecvf.com/content/WACV2022/papers/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [MaskSplit: Self-supervised Meta-learning for Few-shot Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Amac_MaskSplit_Self-Supervised_Meta-Learning_for_Few-Shot_Semantic_Segmentation_WACV_2022_paper.pdf) | None |  |
| WACV2022 | [Hyper-Convolution Networks for Biomedical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Ma_Hyper-Convolution_Networks_for_Biomedical_Image_Segmentation_WACV_2022_paper.pdf) | [code](https://github.com/tym002/Hyper-Convolution) |  |
| WACV2022 | [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_AuxAdapt_Stable_and_Efficient_Test-Time_Adaptation_for_Temporally_Consistent_Video_WACV_2022_paper.pdf) | None |  |
| ICCV2021 | [Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Chan_Entropy_Maximization_and_Meta_Classification_for_Out-of-Distribution_Detection_in_Semantic_ICCV_2021_paper.pdf) | [code](https://github.com/robin-chan/meta-ood) |  |
| ICCV2021 | [Perturbed Self-Distillation: Weakly Supervised Large-Scale Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Perturbed_Self-Distillation_Weakly_Supervised_Large-Scale_Point_Cloud_Semantic_Segmentation_ICCV_2021_paper.pdf) | None | enlightened |
| CVPR2020 | [Context Prior for Scene Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Context_Prior_for_Scene_Segmentation_CVPR_2020_paper.pdf) | [code](https://git.io/ContextPrior) | enlightened |
|  | [Deep Transport Network for Unsupervised Video Object Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Deep_Transport_Network_for_Unsupervised_Video_Object_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [TempNet: Online Semantic Segmentation on Large-scale Point Cloud Series](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_TempNet_Online_Semantic_Segmentation_on_Large-Scale_Point_Cloud_Series_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Generalize then Adapt: Source-Free Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Kundu_Generalize_Then_Adapt_Source-Free_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://sites.google.com/view/sfdaseg) |  |
| ICCV2021 | [Region-Aware Contrastive Learning for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Region-Aware_Contrastive_Learning_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Graph Constrained Data Representation Learning for Human Motion Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Dimiccoli_Graph_Constrained_Data_Representation_Learning_for_Human_Motion_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/mdimiccoli/GCRL-for-HMS/) | enlightened |
| ICCV2021 | [Motion Segmentation and Tracking using Normalized Cuts ](https://ieeexplore.ieee.org/document/710861) | None | enlightened |
| ICCV2015 | [Temporal Subspace Clustering for Human Motion Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410863) | None |  |
| ICCV2021 | [Region-aware Contrastive Learning for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Region-Aware_Contrastive_Learning_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | None | enlightened |
| ICCV2021 | [GP-S3Net: Graph-Based Panoptic Sparse Semantic Segmentation Network](https://openaccess.thecvf.com/content/ICCV2021/papers/Razani_GP-S3Net_Graph-Based_Panoptic_Sparse_Semantic_Segmentation_Network_ICCV_2021_paper.pdf) | None | enlightened |
| ICCV2021 | [Joint Inductive and Transductive Learning for Video Object Segmentation](https://arxiv.org/pdf/2108.03679.pdf) | [code](https://github.com/maoyunyao/JOINT) |  |
| ICCV2021 | [Guided Point Contrastive Learning for Semi-supervised Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Guided_Point_Contrastive_Learning_for_Semi-Supervised_Point_Cloud_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zang_FASA_Feature_Augmentation_and_Sampling_Adaptation_for_Long-Tailed_Instance_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/yuhangzang/FASA) |  |
| ICCV2021 | [Full-Duplex Strategy for Video Object Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ji_Full-Duplex_Strategy_for_Video_Object_Segmentation_ICCV_2021_paper.pdf) | [code](http://dpfan.net/FSNet/) |  |
| ICCV2021 | [Robust Trust Region for Weakly Supervised Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Marin_Robust_Trust_Region_for_Weakly_Supervised_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/dmitrii-marin/robust_trust_region) |  |
| ICCV2021 | [Exploring Cross-Image Pixel Contrast for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/tfzhou/ContrastiveSeg) | enlightened |
| ICCV2021 | [Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Instance_Segmentation_in_3D_Scenes_Using_Semantic_Superpoint_Tree_Networks_ICCV_2021_paper.pdf) | [code](https://github.com/Gorilla-Lab-SCUT/SSTNet) |  |
| ICCV2021 | [Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Leveraging_Auxiliary_Tasks_With_Affinity_Learning_for_Weakly_Supervised_Semantic_ICCV_2021_paper.pdf) | [code](https://github.com/xulianuwa/AuxSegNet) | enlightened |
| ICCV2021 | [Learning With Noisy Labels for Robust Point Cloud Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Learning_With_Noisy_Labels_for_Robust_Point_Cloud_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Dynamic_Divide-and-Conquer_Adversarial_Training_for_Robust_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Self-Regulation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Self-Regulation_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Uncertainty-Aware Pseudo Label Refinery for Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Mining Contextual Information Beyond Image for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_Mining_Contextual_Information_Beyond_Image_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/CharlesPikachu/mcibi) |  |
| ICCV2021 | [Contrastive Learning for Label Efficient Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Contrastive_Learning_for_Label_Efficient_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Truong_BiMaL_Bijective_Maximum_Likelihood_Approach_to_Domain_Adaptation_in_Semantic_ICCV_2021_paper.pdf) | [code](https://github.com/uark-cviu/BiMaL) |  |
| ICCV2021 | [BAPA-Net: Boundary Adaptation and Prototype Alignment for Cross-Domain Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_BAPA-Net_Boundary_Adaptation_and_Prototype_Alignment_for_Cross-Domain_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/manmanjun/BAPA-Net) |  |
| ICCV2021 | [RECALL: Replay-based Continual Learning in Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Maracani_RECALL_Replay-Based_Continual_Learning_in_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/LTTM/RECALL) |  |
| ICCV2021 | [Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Simpler_Is_Better_Few-Shot_Semantic_Segmentation_With_Classifier_Weight_Transformer_ICCV_2021_paper.pdf) | [code](https://github.com/zhiheLu/CWTfor-FSS) |  |
| ICCV2021 | [Unlocking the Potential of Ordinary Classifier: Class-specific Adversarial Erasing Framework for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf) | [code](https://github.com/KAIST-vilab/OC-CSE) |  |
| ICCV2021 | [Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Sparse-to-Dense_Feature_Matching_Intra_and_Inter_Domain_Cross-Modal_Learning_in_ICCV_2021_paper.pdf) | [code](https://github.com/leolyj/DsCML) |  |
| ICCV2021 | [VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_VMNet_Voxel-Mesh_Network_for_Geodesic-Aware_3D_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/hzykent/VMNet) |  |
| ICCV2021 | [ShapeConv: Shape-aware Convolutional Layer for Indoor RGB-D Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Cao_ShapeConv_Shape-Aware_Convolutional_Layer_for_Indoor_RGB-D_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Seminar Learning for Click-Level Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Seminar_Learning_for_Click-Level_Weakly_Supervised_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Complementary Patch for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Complementary_Patch_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Baek_Exploiting_a_Joint_Embedding_Space_for_Generalized_Zero-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://cvlab.yonsei.ac.kr/projects/JoEm) |  |
| ICCV2021 | [Deep Metric Learning for Open World Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Cen_Deep_Metric_Learning_for_Open_World_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_ReDAL_Region-Based_and_Diversity-Aware_Active_Learning_for_Point_Cloud_Semantic_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Geometric Unsupervised Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Guizilini_Geometric_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/tri-ml/packnet-sfm) |  |
| ICCV2021 | [Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Spurr_Self-Supervised_3D_Hand_Pose_Estimation_From_Monocular_RGB_via_Contrastive_ICCV_2021_paper.pdf) | [code](https://ait.ethz.ch/projects/2021/PeCLR/) | enlightened |
| ICCV2021 | [How Shift Equivariance Impacts Metric Learning for Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Rumberger_How_Shift_Equivariance_Impacts_Metric_Learning_for_Instance_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/Kainmueller-Lab/shift_equivariance_unet) | enlightened |
| ICCV2021 | [Personalized Image Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Personalized_Image_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://mmcheng.net/pss/) |  |
| ICCV2021 | [Learning Meta-class Memory for Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Meta-Class_Memory_for_Few-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Hierarchical Aggregation for 3D Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Hierarchical_Aggregation_for_3D_Instance_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/hustvl/HAIS) |  |
| ICCV2021 | [Perception-Aware Multi-Sensor Fusion for 3D LiDAR Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhuang_Perception-Aware_Multi-Sensor_Fusion_for_3D_LiDAR_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/ICEORY/PMF) |  |
| ICCV2021 | [Segmenter: Transformer for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/rstrudel/segmenter) |  |
| ICCV2021 | [Multi-Anchor Active Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ning_Multi-Anchor_Active_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/munanning/MADA) |  |
| ICCV2021 | [Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_Pixel_Contrastive-Consistent_Semi-Supervised_Semantic_Segmentation_ICCV_2021_paper.pdf) | None | enlightened |
| ICCV2021 | [Panoptic Narrative Grounding](https://openaccess.thecvf.com/content/ICCV2021/papers/Gonzalez_Panoptic_Narrative_Grounding_ICCV_2021_paper.pdf) | [code](https://github.com/BCV-Uniandes/PNG) | new task |
| ICCV2021 | [Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals](https://openaccess.thecvf.com/content/ICCV2021/papers/Van_Gansbeke_Unsupervised_Semantic_Segmentation_by_Contrasting_Object_Mask_Proposals_ICCV_2021_paper.pdf) | [code](https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation.git) | enlightened |
| ICCV2021 | [Rank & Sort Loss for Object Detection and Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/kemaloksuz/RankSortLoss) |  |
| ICCV2021 | [ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_ISNet_Integrate_Image-Level_and_Semantic-Level_Context_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/SegmentationBLWX/sssegmentation) |  |
| CVPR2021 | [Learning Calibrated Medical Image Segmentation via Multi-rater Agreement Modeling](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Learning_Calibrated_Medical_Image_Segmentation_via_Multi-Rater_Agreement_Modeling_CVPR_2021_paper.pdf) | [code](https://github.com/jiwei0921/MRNet/) |  |
| CVPR2021 | [4D Panoptic LiDAR Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Aygun_4D_Panoptic_LiDAR_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/mehmetaygun/4d-pls) | new task |
| CVPR2021 | [Zero-Shot Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Zero-Shot_Instance_Segmentation_CVPR_2021_paper.pdf) | None | new task |
| CVPR2021 | [Learning To Count Everything](https://openaccess.thecvf.com/content/CVPR2021/papers/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.pdf) | [code](https://github.com/cvlab-stonybrook/LearningToCountEverything) |  |
| CVPR2021 | [Toward Joint Thing-and-Stuff Mining for Weakly Supervised Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Toward_Joint_Thing-and-Stuff_Mining_for_Weakly_Supervised_Panoptic_Segmentation_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Cluster, Split, Fuse, and Update: Meta-Learning for Open Compound Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_Cluster_Split_Fuse_and_Update_Meta-Learning_for_Open_Compound_Domain_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spatial_Feature_Calibration_and_Temporal_Fusion_for_Effective_One-Stage_Video_CVPR_2021_paper.pdf) | [code](https://github.com/MinghanLi/STMask) |  |
| CVPR2021 | [Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Panoptic-PolarNet_Proposal-Free_LiDAR_Point_Cloud_Panoptic_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/edwardzhou130/Panoptic-PolarNet) |  |
| CVPR2021 | [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) | [code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/copy_paste) | data-argu |
| CVPR2021 | [A2-FPN: Attention Aggregation based Feature Pyramid Network for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_A2-FPN_Attention_Aggregation_Based_Feature_Pyramid_Network_for_Instance_Segmentation_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.pdf) | [code](https://github.com/tinyalpha/BPR) |  |
| CVPR2021 | [BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_BBAM_Bounding_Box_Attribution_Map_for_Weakly_Supervised_Semantic_and_CVPR_2021_paper.pdf) | [code](https://github.com/jbeomlee93/BBAM) |  |
| CVPR2021 | [HyperSeg: Patch-wise Hypernetwork for Real-time Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Nirkin_HyperSeg_Patch-Wise_Hypernetwork_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf) | [code](https://nirkin.com/hyperseg) |  |
| CVPR2021 | [Point Cloud Instance Segmentation using Probabilistic Embeddings](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Point_Cloud_Instance_Segmentation_Using_Probabilistic_Embeddings_CVPR_2021_paper.pdf) | None | enlightened |
| CVPR2021 | [FAPIS: A Few-shot Anchor-free Part-based Instance Segmenter](https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_FAPIS_A_Few-Shot_Anchor-Free_Part-Based_Instance_Segmenter_CVPR_2021_paper.pdf) | [code](https://github.com/ducminhkhoi/FAPIS) |  |
| CVPR2021 | [Background-Aware Pooling and Noise-Aware Loss for Weakly-SupervisedSemantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Oh_Background-Aware_Pooling_and_Noise-Aware_Loss_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf) | [code](https://cvlab.yonsei.ac.kr/projects/BANA) |  |
| CVPR2021 | [Hierarchical Lovasz Embeddings for Proposal-free Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Kerola_Hierarchical_Lovasz_Embeddings_for_Proposal-Free_Panoptic_Segmentation_CVPR_2021_paper.pdf) | None | enlightened |
| CVPR2021 | [Learning to Associate Every Segment for Video Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Woo_Learning_To_Associate_Every_Segment_for_Video_Panoptic_Segmentation_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds](https://openaccess.thecvf.com/content/CVPR2021/papers/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Cylindrical_and_Asymmetrical_3D_Convolution_Networks_for_LiDAR_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/xinge008/Cylinder3D) |  |
| CVPR2021 | [DCT-Mask: Discrete Cosine Transform Mask Representation for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_DCT-Mask_Discrete_Cosine_Transform_Mask_Representation_for_Instance_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/aliyun/DCT-Mask) |  |
| CVPR2021 | [Deeply Shape-guided Cascade for Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_Deeply_Shape-Guided_Cascade_for_Instance_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/hding2455/DSC) |  |
| CVPR2021 | [Part-aware Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/de_Geus_Part-Aware_Panoptic_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/tue-mps/panoptic_parts) | new task |
| CVPR2021 | [ColorRL: Reinforced Coloring for End-to-End Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Tuan_ColorRL_Reinforced_Coloring_for_End-to-End_Instance_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/anhtuanhsgs/ColorRL) | novelty, enlightened |
| CVPR2021 | [MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.pdf) | [code](https://github.com/conradry/max-deeplab) | enlightened |
| CVPR2021 | [Fully Convolutional Networks for Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Fully_Convolutional_Networks_for_Panoptic_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/Jia-Research-Lab/PanopticFCN) |  |
| CVPR2021 | [(AF)2-S3Net: Attentive Feature Fusion with Adaptive Feature Selection for Sparse Semantic Segmentation Network](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_AF2-S3Net_Attentive_Feature_Fusion_With_Adaptive_Feature_Selection_for_Sparse_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [PiCIE: Unsupervised Semantic Segmentation using Invariance and Equivariance in Clustering](https://openaccess.thecvf.com/content/CVPR2021/papers/Cho_PiCIE_Unsupervised_Semantic_Segmentation_Using_Invariance_and_Equivariance_in_Clustering_CVPR_2021_paper.pdf) | [code](https://github.com/janghyuncho/PiCIE) | enlightened |
| CVPR2021 | [Learning Dynamic Routing for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_Dynamic_Routing_for_Semantic_Segmentation_CVPR_2020_paper.pdf) | [code](https://github.com/yanwei-li/DynamicRouting) |  |
| CVPR2021 | [PolyTransform: Deep Polygon Transformer for Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liang_PolyTransform_Deep_Polygon_Transformer_for_Instance_Segmentation_CVPR_2020_paper.pdf) | None |  |
| CVPR2021 | [PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_PolarNet_An_Improved_Grid_Representation_for_Online_LiDAR_Point_Clouds_CVPR_2020_paper.pdf) | [code](https://github.com/edwardzhou130/PolarSeg) |  |
| AAAI2022 | [Learning to Model Pixel-Embedded Affinity for Homogeneous Instance Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/19984) | [code](https://github.com/weih527/) |  |
| AAAI2022 | [Multi-Knowledge Aggregation and Transfer for Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/20077) | None |  |
| AAAI2022 | [Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/20085) | None |  |
| AAAI2022 | [Unsupervised Representation for Semantic Segmentation by Implicit Cycle-Attention Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/20100) | None |  |
| AAAI2022 | [CPRAL: Collaborative Panoptic-Regional Active Learning for Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/20107) | None |  |
| CVPR2020 | [BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_BlendMask_Top-Down_Meets_Bottom-Up_for_Instance_Segmentation_CVPR_2020_paper.pdf) | [code](https://github.com/aim-uofa/AdelaiDet) |  |
| CVPR2020 | [Pixel Consensus Voting for Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Pixel_Consensus_Voting_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) | [code](https://github.com/w-hc/pcv) | enlightened |
| CVPR2020 | [3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Engelmann_3D-MPA_Multi-Proposal_Aggregation_for_3D_Semantic_Instance_Segmentation_CVPR_2020_paper.pdf) | None |  |
| CVPR2020 | [PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.pdf) | None |  |
| CVPR2020 | [Real-Time Panoptic Segmentation from Dense Detections](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hou_Real-Time_Panoptic_Segmentation_From_Dense_Detections_CVPR_2020_paper.pdf) | None |  |
| CVPR2020 | [Affinity Graph Supervision for Visual Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Affinity_Graph_Supervision_for_Visual_Recognition_CVPR_2020_paper.pdf) | None |  |
| CVPR2020 | [Spatial Pyramid Based Graph Reasoning for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Spatial_Pyramid_Based_Graph_Reasoning_for_Semantic_Segmentation_CVPR_2020_paper.pdf) | None |  |
| CVPR2020 | [Bidirectional Graph Reasoning Network for Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Bidirectional_Graph_Reasoning_Network_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) | None |  |
| ECCV2022 | [OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers](https://arxiv.org/pdf/2207.02255.pdf) | [code](https://github.com/PJLallen/OSFormer) |  |
| ECCV2022 | [PseudoClick: Interactive Image Segmentation with Click Imitation](https://arxiv.org/pdf/2207.05282.pdf) | None | enlightened |
| ECCV2022 | [Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding](https://arxiv.org/pdf/2207.08455.pdf) | None | enlightened |
| ICLR2022 | [LABEL-EFFICIENT SEMANTIC SEGMENTATION WITH DIFFUSION MODELS](https://arxiv.org/pdf/2112.03126.pdf) | [code](https://github.com/yandex-research/ddpm-segmentation) | diffusion model |
| arxiv | [Diffusion Models as Plug-and-Play Priors](https://arxiv.org/pdf/2206.09012.pdf) | [code](https://github.com/AlexGraikos/diffusion_priors) |  |
| WACV2020 | [Distributed Iterative Gating Networks for Semantic Segmentation](https://arxiv.org/abs/1909.12996) | None |  |
| WACV2020 | [Multi Receptive Field Network for Semantic Segmentation](https://arxiv.org/abs/2011.08577) | None |  |
| WACV2020 | [MaskPlus: Improving Mask Generation for Instance Segmentation](https://arxiv.org/abs/1907.06713) | None |  |
| WACV2020 | [Efficient Video Semantic Segmentation with Labels Propagation and Refinement](https://arxiv.org/abs/1912.11844) | None |  |
| WACV2020 | [Global Context Reasoning for Semantic Segmentation of 3D Point Clouds](https://openaccess.thecvf.com/content_WACV_2020/papers/Ma_Global_Context_Reasoning_for_Semantic_Segmentation_of_3D_Point_Clouds_WACV_2020_paper.pdf) | None |  |
| ICCV2019 | [Asymmetric Non-local Neural Networks for Semantic Segmentation](https://arxiv.org/abs/1908.07678) | [code](https://github.com/MendelXu/ANN.git) |  |
| ICCV2019 | [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721) | [code](https://github.com/speedinghzl/CCNet) |  |
| ICCV2019 | [SSAP: Single-Shot Instance Segmentation With Affinity Pyramid](https://arxiv.org/abs/1909.01616) | [code](https://github.com/yanziwei/ssap) | enlightened |
| ICCV2019 | [ACE: Adapting to Changing Environments for Semantic Segmentation](https://arxiv.org/abs/1904.06268) | None |  |
| ICCV2019 | [Dynamic Multi-scale Filters for Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf) | [code](https://github.com/Junjun2016/DMNet) | draw pic |
| ICCV2019 | [Towards Bridging Semantic Gap to Improve Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Towards_Bridging_Semantic_Gap_to_Improve_Semantic_Segmentation_ICCV_2019_paper.pdf) | None |  |
| ICCV2019 | [Feature Weighting and Boosting for Few-Shot Segmentation](https://arxiv.org/abs/1909.13140) | None |  |
| ICCV2019 | [Explicit Shape Encoding for Real-Time Instance Segmentation](https://arxiv.org/abs/1908.04067) | [code](https://github.com/WenqiangX/ese_seg) |  |
| ICCV2019 | [IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things](https://arxiv.org/abs/1906.06597) | None |  |
| ICCV2019 | [Video Instance Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Video_Instance_Segmentation_ICCV_2019_paper.pdf) | None |  |
| ICCV2019 | [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](https://arxiv.org/abs/1907.05740) | None |  |
| ICCV2019 | [Self-Supervised Difference Detection for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/1911.01370) | None |  |
| ICCV2019 | [Universal Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/1811.10323) | None |  |
| ICCV2019 | [ACFNet: Attentional Class Feature Network for Semantic Segmentation](https://arxiv.org/abs/1909.09408) | [code](https://github.com/zrl4836/ACFNet) |  |
| ICCV2019 | [YOLACT Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) | [code](https://github.com/dbolya/yolact) |  |
| ICCV2019 | [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/2006.09214) | [code](tinyurl.com/FCOSv1) |  |
| ICCV2019 |  [Semantic Instance Segmentation via Deep Metric Learning](https://arxiv.org/abs/1703.10277) | None | enlighten |
| ECCV2022 | [SeqFormer: Sequential Transformer for Video Instance Segmentation](https://arxiv.org/abs/2112.08275) | [code](https://github.com/wjf5203/SeqFormer) |  |
| ECCV2022 | [On Mitigating Hard Clusters for Face Clustering](https://arxiv.org/abs/2207.11895) | [code](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |  |
| ECCV2022 | [CenterFormer: Center-based Transformer for 3D Object Detection](https://arxiv.org/abs/2209.05588) | [code](https://github.com/TuSimple/centerformer) |  |
| ECCV2022 | [ObjectBox: From Centers to Boxes for Anchor-Free Object Detection](https://arxiv.org/abs/2207.06985) | [code](https://github.com/MohsenZand/ObjectBox) |  |
| ECCV2022 | [Learning Topological Interactions for Multi-Class Medical Image Segmentation](https://arxiv.org/abs/2207.09654) | [code](https://github.com/TopoXLab/TopoInteraction) |  |
| ECCV2022 | [CAR: Class-aware Regularizations for Semantic Segmentation](https://arxiv.org/abs/2203.07160) | [code](https://github.com/edwardyehuang/CAR) |  |
| ECCV2022 | [Active Pointly-Supervised Instance Segmentation](https://arxiv.org/abs/2207.11493) | None |  |
| ECCV2022 | [DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2207.09988) | [code](https://github.com/dvlab-research/DecoupleNet) |  |
| ECCV2022 | [Video Mask Transfiner for High-Quality Video Instance Segmentation](https://arxiv.org/abs/2207.14012) | [code](http://vis.xyz/pub/vmt) |  |
| ECCV2022 | [Box-supervised Instance Segmentation with Level Set Evolution](https://arxiv.org/abs/2207.09055) | [code](https://github.com/LiWentomng/boxlevelset) |  |
| ECCV2022 | [Mining Relations among Cross-Frame Affinities for Video Semantic Segmentation](https://arxiv.org/abs/2207.10436) | [code](https://github.com/GuoleiSun/VSS-MRCFA) |  |
| ECCV2022 | 2D Amodal Instance Segmentation Guided by 3D Shape Prior | None | updating |
| ECCV2022 | Max Pooling with Vision Transformers reconciles class and shape in weakly supervised semantic segmentation | None | updating |
| ECCV2022 | [CP2: Copy-Paste Contrastive Pretraining for Semantic Segmentation](https://arxiv.org/abs/2203.11709) | [code](https://github.com/wangf3014/CP2) | enlightened |
| ECCV2022 | Continual Semantic Segmentation via Structure Preserving and Projected Feature Alignment | None |  |
| ECCV2022 | [Multi-scale and Cross-scale Contrastive Learning for Semantic Segmentation](https://arxiv.org/abs/2203.13409) | [code](https://github.com/RViMLab/MS_CS_ContrSeg) | enlightened |
| ECCV2022 | [Learning Implicit Feature Alignment Function for Semantic Segmentation](https://arxiv.org/abs/2206.08655) | [code](https://github.com/hzhupku/IFA) |  |
| ECCV2022 | [Open-world Semantic Segmentation via Contrasting and Clustering Vision-language Embedding](https://arxiv.org/abs/2207.08455) | None |  |
| ECCV2022 | [RBC: Rectifying the Biased Context in Continual Semantic Segmentation](https://arxiv.org/abs/2203.08404) | None |  |
| ECCV2022 | [Panoptic-PartFormer: Learning a Unified model for Panoptic Part Segmentation](https://arxiv.org/abs/2204.04655) | [code](https://github.com/lxtGH/Panoptic-PartFormer) |  |
| ECCV2022 | [Language-Grounded Indoor 3D Semantic Segmentation in the Wild](https://arxiv.org/abs/2204.07761) | [code](https://rozdavid.github.io/scannet200) | loss enlightened |
| ECCV2022 | [Learning with Free Object Segments for Long-Tailed Instance Segmentation](https://arxiv.org/abs/2202.11124) |  |  |
| ECCV2022 | [A Simple Single-Scale Vision Transformer for Object Detection and Instance Segmentation](https://arxiv.org/abs/2112.09747) | [code](https://github.com/tensorflow/models/tree/master/official/projects/uvit) |  |
| ICLR2022 | [Label-Efficient Semantic Segmentation with Diffusion Models](https://openreview.net/pdf?id=SlxSY2UZQT) | [code](https://github.com/yandex-research/ddpm-segmentation) | diffusion for seg |
| ICLR2022 |  |  |  |
| ICLR2022 |  |  |  |
| ICLR2022 |  |  |  |



------------


## Graph Representation and Cluster
|   |   |   |  |
| :------------: | :------------: | :------------: | :------------: |
| ICME2020 | [S3NET: GRAPH REPRESENTATIONAL NETWORK FOR SKETCH RECOGNITION](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102957) | [code](https://github.com/yanglan0225/s3net) |  |
| NeurIPS2018 | [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf) | [code](https://github.com/RexYing/diffpool) or [code](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/diff_pool.py)|  |
| NeurIPS2020 | [DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation](https://arxiv.org/pdf/2007.11301.pdf) | [code](https://github.com/alexandre01/deepsvg) |  |
| ICLR2018 | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | [code](https://github.com/PetarV-/GAT) |  |
| IJCAI2019 | [DAEGC: Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/pdf/1906.06532.pdf) | [code](https://github.com/Tiger101010/DAEGC) |  |
| ICML2020 | [Mini-cut: Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/pdf/1907.00481.pdf) | [code](https://graphneural.network/layers/pooling/#mincutpool) or [code](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/mincut_pool.py)|  |
| KDD2019 | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) | [code](https://github.com/benedekrozemberczki/ClusterGCN) |  |
| ICCV2021 | [Graph Contrastive Clustering](https://arxiv.org/pdf/2104.01429.pdf) | [code](https://github.com/mynameischaos/GCC) |  |
| ECCV2020 | [Scan: Learning to classify images without labels](https://arxiv.org/pdf/2005.12320.pdf) | [code](https://github.com/wvangansbeke/Unsupervised-Classification) | enlightened |
|  | [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/pdf/1511.06335.pdf) | [code](https://github.com/piiswrong/dec) |  |
| CVPR2022 | [DeepDPM: Deep Clustering With an Unknown Number of Clusters](https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf) | [code](https://github.com/BGU-CS-VIL/DeepDPM) |  |
| CVPR2022 | [Lifelong Graph Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Lifelong_Graph_Learning_CVPR_2022_paper.pdf) | [code](https://github.com/wang-chen/LGL) |  |
| ICCV2021 | [Clustering by Maximizing Mutual Information Across Views](https://openaccess.thecvf.com/content/ICCV2021/papers/Do_Clustering_by_Maximizing_Mutual_Information_Across_Views_ICCV_2021_paper.pdf) | None |  |
| ICCV2021 | [Region Similarity Representation Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiao_Region_Similarity_Representation_Learning_ICCV_2021_paper.pdf) | [code](https://github.com/Tete-Xiao/ReSim) |  |
| AAAI2022 | [Deep Graph Clustering via Dual Correlation Reduction](https://www.researchgate.net/profile/Yue-Liu-240/publication/357271184_Deep_Graph_Clustering_via_Dual_Correlation_Reduction/links/61c466e68bb20101842f9a92/Deep-Graph-Clustering-via-Dual-Correlation-Reduction.pdf) | [code](https://github.com/yueliu1999/DCRN) |  |
| AAAI2021 | [Contrastive Clustering](https://arxiv.org/pdf/2009.09687.pdf) | [code](https://github.com/Yunfan-Li/Contrastive-Clustering) | enlightened |
| CVPR2021 | [Nearest Neighbor Matching for Deep Clustering](https://openaccess.thecvf.com/content/CVPR2021/papers/Dang_Nearest_Neighbor_Matching_for_Deep_Clustering_CVPR_2021_paper.pdf) | [code](https://github.com/ZhiyuanDang/NNM) | enlightened |
| CVPR2020 | [Deep Semantic Clustering by Partition Confidence Maximisation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Deep_Semantic_Clustering_by_Partition_Confidence_Maximisation_CVPR_2020_paper.pdf) | [code](https://github.com/Raymond-sci/PICA) | enlightened |
| CVPR2021 | [Deep Fair Clustering for Visual Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.pdf) | None |  |
| CVPR2021 | [Distribution-induced Bidirectional Generative Adversarial Network for Graph Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Distribution-Induced_Bidirectional_Generative_Adversarial_Network_for_Graph_Representation_Learning_CVPR_2020_paper.pdf) | [code](https://github.com/SsGood/) |  |
| AAAI2022 | [GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/20377) | None |  |
| AAAI2022 | [Structure-Aware Transformer for Graph Representation Learning](https://proceedings.mlr.press/v162/chen22r/chen22r.pdf) | [code](https://github.com/BorgwardtLab/SAT) |  |
| CVPR2020 | [Large Scale Video Representation Learning via Relational Graph Clustering](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Large_Scale_Video_Representation_Learning_via_Relational_Graph_Clustering_CVPR_2020_paper.pdf) | None |  |
| ECCV2018 | [Universal Sketch Perceptual Grouping](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_LI_Universal_Sketch_Perceptual_ECCV_2018_paper.pdf) | [code](https://github.com/KeLi-SketchX/Universal-sketch-perceptual-grouping) | enlightened |
| WACV2020 | [MaskPlus: Improving Mask Generation for Instance Segmentation](https://openaccess.thecvf.com/content_WACV_2020/papers/Xu_MaskPlus_Improving_Mask_Generation_for_Instance_Segmentation_WACV_2020_paper.pdf) | None |  |
| WACV2020 | [GAR: Graph Assisted Reasoning for Object Detection](https://openaccess.thecvf.com/content_WACV_2020/papers/Li_GAR_Graph_Assisted_Reasoning_for_Object_Detection_WACV_2020_paper.pdf) | None |  |
| WACV2020 | [Differentiable Scene Graphs](https://arxiv.org/abs/1902.10200) | [code](https://github.com/shikorab/DSG) | enlightened |
| WACV2020 | [Multi-Level Representation Learning for Deep Subspace Clustering](https://arxiv.org/abs/2001.08533) | None |  |
| ICCV2019 | [Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition](https://arxiv.org/abs/1908.07325) | [code](https://github.com/HCPLab-SYSU/SSGRL) |  |
| ECCV2022 | [On Mitigating Hard Clusters for Face Clustering](https://arxiv.org/abs/2207.11895) | [code](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |  |
| ECCV2022 | [Generative Subgraph Contrast for Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2207.11996) | [code](https://github.com/yh-han/GSC.git) | enlightened |
| ICLR2022 | [Ada-NETS: Face Clustering via Adaptive Neighbour Discovery in the Structure Space](https://openreview.net/pdf?id=QJWVP4CTmW4) | [code](https://github.com/damo-cv/Ada-NETS) | enlightened |
| ICLR2022 |  |  |  |
| ICLR2022 |  |  |  |

------------


## Referring Segmentation
|  |  |  |  |
| :------------: | :------------: | :------------: | :------------: |
| CVPR024 | []() | [code]() |  |
| CVPR024 | []() | [code]() |  |
| CVPR024 | [Decoupling Static and Hierarchical Motion Perception for Referring Video Segmentation](https://arxiv.org/abs/2404.03645) | [code](https://github.com/heshuting555/DsHmp) |  |
| CVPR024 | LQMFormer: Language-aware Query Mask Transformer for Referring Image Segmentation | None |  |
| CVPR024 | [Mask Grounding for Referring Image Segmentation](https://arxiv.org/abs/2312.12198) | None |  |
| CVPR024 | [LoSh: Long-Short Text Joint Prediction Network for Referring Video Object Segmentation](https://arxiv.org/abs/2306.08736) | [code](https://github.com/LinfengYuan1997/LoSh) |  |
| CVPR024 | Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation | [code](https://github.com/Rubics-Xuan/MRES) |  |
| CVPR024 | Prompt-Driven Referring Image Segmentation with Instance Contrasting | None |  |
| CVPR024 | [Curriculum Point Prompting for Weakly-Supervised Referring Image Segmentation](https://arxiv.org/abs/2404.11998) | [code]() |  |
| ICCV2021 | [Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/abs/2108.05565) | [code](https://github.com/henghuiding/Vision-Language-Transformer) |  |
| CVPR2021 | [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://arxiv.org/abs/2105.01839) | [code](https://github.com/fengguang94/CEFNet) |  |
| CVPR2021 | [CRIS: CLIP-Driven Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/DerrickWang005/CRIS.pytorch) |  |
| CVPR2022 | [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/yz93/LAVT-RIS) |  |
| ICCV2021 | [Vision-Language Transformer and Query Generation for Referring Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.pdf) | [code](https://github.com/henghuiding/VisionLanguage-Transformer) |  |
| CVPR2021 | [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Encoder_Fusion_Network_With_Co-Attention_Embedding_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | None |  |
| CVPR2021 | [Bottom-Up Shift and Reasoning for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Bottom-Up_Shift_and_Reasoning_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/incredibleXM/BUSNet) |  |
| WACV2020 | [Leveraging Pretrained Image Classifiers for Language-Based Segmentation](https://arxiv.org/abs/1911.00830) | [code](https://sites.google.com/stanford.edu/cls-seg) |  | 
| ICCV2019 | [Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding](https://arxiv.org/abs/1908.10568) | [code](https://github.com/GingL/ARN) |  |
| ICCV2019 | [G3RAPHGROUND: Graph-based Language Grounding](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bajaj_G3raphGround_Graph-Based_Language_Grounding_ICCV_2019_paper.pdf) | None |  |
| ICCV2019 | [Dynamic Graph Attention for Referring Expression Comprehension](https://arxiv.org/abs/1909.08164) | None |  |
| ICCV2019 | [Learning to Assemble Neural Module Tree Networks for Visual Grounding](https://arxiv.org/abs/1812.03299) | None |  |
| ICCV2019 | [A Fast and Accurate One-Stage Approach to Visual Grounding](https://arxiv.org/abs/1908.06354) | [code](https://github.com/zyang-ur/onestage_grounding) |  |
| ICCV2019 | [See-Through-Text Grouping for Referring Image Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_See-Through-Text_Grouping_for_Referring_Image_Segmentation_ICCV_2019_paper.pdf) | None |  |
| ECCV2022 | [UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling](https://arxiv.org/abs/2111.12085) | [code](https://github.com/microsoft/UniTAB) |  |
| ECCV2022 | [SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding](https://arxiv.org/abs/2207.13325) | [code](https://github.com/qumengxue/siri-vg.git) |  |
| ECCV2022 | [Bottom Up Top Down Detection Transformers for Language Grounding in Images and Point Clouds](https://arxiv.org/abs/2112.08879) | [code](https://butd-detr.github.io) |  |
| ICCV2023 | [Bridging Vision and Language Encoders: Parameter-Efficient Tuning for Referring Image Segmentation](https://arxiv.org/abs/2307.11545) | [code](https://github.com/kkakkkka/ETRIS) |  |








