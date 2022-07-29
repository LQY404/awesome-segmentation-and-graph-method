This is a repository of the segmentation method and graph method. In the segmentation part, the instance, sematic, panoptic and referring segmentation are included. In the graph part, the representation learning, clustering and so on methods are included. We will update continue...


------------

TOC{:toc}

------------


## Instance, Semantic and Panoptic Segmentation.

|   |   |   |
| :------------: | :------------: | :------------: |
| CVPR2022  | [Deep Hierarchical Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Deep_Hierarchical_Semantic_Segmentation_CVPR_2022_paper.pdf)  | [code](https://github.com/0liliulei/HieraSeg) |
| CVPR2022  | [Rethinking Semantic Segmentation: A Prototype View](https://arxiv.org/abs/2203.15102) | [code](https://github.com/tfzhou/ProtoSeg)  |
| CVPR2022  | [Sparse Instance Activation for Real-Time Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Sparse_Instance_Activation_for_Real-Time_Instance_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/hustvl/SparseInst) |
| CVPR2022 | [Instance Segmentation with Mask-supervised Polygonal Boundary Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Lazarow_Instance_Segmentation_With_Mask-Supervised_Polygonal_Boundary_Transformers_CVPR_2022_paper.pdf) | [code](https://github.com/mlpc-ucsd/BoundaryFormer) |
| CVPR2022 | [CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_CMT-DeepLab_Clustering_Mask_Transformers_for_Panoptic_Segmentation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Panoptic_SegFormer_Delving_Deeper_Into_Panoptic_Segmentation_With_Transformers_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_PanopticDepth_A_Unified_Framework_for_Depth-Aware_Panoptic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/NaiyuGao/PanopticDepth) |
| CVPR2022 | [Mask Transfiner for High-Quality Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/SysCV/transfiner) |
| CVPR2022 | [Unsupervised Hierarchical Semantic Segmentation With Multiview Cosegmentation and Clustering Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Unsupervised_Hierarchical_Semantic_Segmentation_With_Multiview_Cosegmentation_and_Clustering_Transformers_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Coarse-To-Fine Feature Mining for Video Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Coarse-To-Fine_Feature_Mining_for_Video_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/GuoleiSun/VSS-CFFM) |
| CVPR2022 | [Sparse Object-Level Supervision for Instance Segmentation With Pixel Embeddings](https://openaccess.thecvf.com/content/CVPR2022/papers/Wolny_Sparse_Object-Level_Supervision_for_Instance_Segmentation_With_Pixel_Embeddings_CVPR_2022_paper.pdf) | [code](https://github.com/kreshuklab/spoco) |
| CVPR2022 | [Segment-Fusion: Hierarchical Context Fusion for Robust 3D Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Thyagharajan_Segment-Fusion_Hierarchical_Context_Fusion_for_Robust_3D_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Pin the Memory: Learning To Generalize Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/Genie-Kim/PintheMemory) |
| CVPR2022 | [SoftGroup for 3D Instance Segmentation on Point Clouds](https://openaccess.thecvf.com/content/CVPR2022/papers/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.pdf) | [code](https://github.com/thangvubk/SoftGroup) |
| CVPR2022 | [SharpContour: A Contour-based Boundary Refinement Approach for Efficient and Accurate Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_SharpContour_A_Contour-Based_Boundary_Refinement_Approach_for_Efficient_and_Accurate_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Beyond_Semantic_to_Instance_Segmentation_Weakly-Supervised_Instance_Segmentation_via_Semantic_CVPR_2022_paper.pdf) | [code](https://github.com/clovaai/BESTIE) |
| CVPR2022 | [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Semi-Supervised_Semantic_Segmentation_Using_Unreliable_Pseudo-Labels_CVPR_2022_paper.pdf) | [code](https://haochen-wang409.github.io/U2PL) |
| CVPR2022 | [Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization](https://openaccess.thecvf.com/content/CVPR2022/papers/Melas-Kyriazi_Deep_Spectral_Methods_A_Surprisingly_Strong_Baseline_for_Unsupervised_Semantic_CVPR_2022_paper.pdf) | [code](https://lukemelas.github.io/deep-spectral-segmentation/) |
| CVPR2022 | [Self-Supervised Learning of Object Parts for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/MkuuWaUjinga/leopart) |
| CVPR2022 | [Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Multi-Scale_High-Resolution_Vision_Transformer_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/facebookresearch/HRViT) |
| CVPR2022 | [C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_C-CAM_Causal_CAM_for_Weakly_Supervised_Semantic_Segmentation_on_Medical_CVPR_2022_paper.pdf) | [code](https://github.com/Tian-lab/C-CAM) |
| CVPR2022 | [Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Dynamic_Prototype_Convolution_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [ContrastMask: Contrastive Learning to Segment Every Thing](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ContrastMask_Contrastive_Learning_To_Segment_Every_Thing_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Generalized Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Tian_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/dvlab-research/GFS-Seg) |
| CVPR2022 | [Decoupling Zero-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Decoupling_Zero-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/dingjiansw101/ZegFormer) |
| CVPR2022 | [TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_TopFormer_Token_Pyramid_Transformer_for_Mobile_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/hustvl/TopFormer) |
| CVPR2022 | [Cross-Image Relational Knowledge Distillation for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.pdf) | [code](https://github.com/VITA-Group/CADTransformer) |
| CVPR2022 | [GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [TubeFormer-DeepLab: Video Mask Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_TubeFormer-DeepLab_Video_Mask_Transformer_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Class-Balanced_Pixel-Level_Self-Labeling_for_Domain_Adaptive_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/lslrh/CPSL) |
| CVPR2022 | [Learning Non-target Knowledge for Few-shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_Non-Target_Knowledge_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/LIUYUANWEI98/NERTNet) |
| CVPR2022 | [SemAffiNet: Semantic-Affine Transformation for Point Cloud Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_SemAffiNet_Semantic-Affine_Transformation_for_Point_Cloud_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/wangzy22/SemAffiNet) |
| CVPR2022 | [Real-Time, Accurate, and Consistent Video Semantic Segmentation via Unsupervised Adaptation and Cross-Unit Deployment on Mobile Device](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Real-Time_Accurate_and_Consistent_Video_Semantic_Segmentation_via_Unsupervised_Adaptation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Tree_Energy_Loss_Towards_Sparsely_Annotated_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/megvii-research/TreeEnergyLoss) |
| CVPR2022 | [Amodal Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Mohan_Amodal_Panoptic_Segmentation_CVPR_2022_paper.pdf) | [code](http://amodal-panoptic.cs.uni-freiburg.de) |
| CVPR2022 | [Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Ru_Learning_Affinity_From_Attention_End-to-End_Weakly-Supervised_Semantic_Segmentation_With_Transformers_CVPR_2022_paper.pdf) | [code](https://github.com/rulixiang/afa) |
| CVPR2022 | [Partial Class Activation Attention for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Partial_Class_Activation_Attention_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/lsa1997/PCAA) |
| CVPR2022 | [Weakly Supervised Semantic Segmentation using Out-of-Distribution Data](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Weakly_Supervised_Semantic_Segmentation_Using_Out-of-Distribution_Data_CVPR_2022_paper.pdf) | [code](https://github.com/naver-ai/w-ood) |
| CVPR2022 | [Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Phan_Class_Similarity_Weighted_Knowledge_Distillation_for_Continual_Semantic_Segmentation_CVPR_2022_paper.pdf) | None |
| CVPR2022 | [Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Bending_Reality_Distortion-Aware_Transformers_for_Adapting_to_Panoramic_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/jamycheung/Trans4PASS) |
| CVPR2022 | [Towards Noiseless Object Contours for Weakly Supervised Semantic](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Towards_Noiseless_Object_Contours_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/BraveGroup/SANCE) |
| CVPR2022 | [L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.pdf) | [code](https://github.com/PengtaoJiang/L2G) |
| CVPR2022 |  |  |



------------


## Graph Learning (representation, cluster...)
|   |   |   |
| :------------: | :------------: | :------------: |
| ICME2020 | [S3NET: GRAPH REPRESENTATIONAL NETWORK FOR SKETCH RECOGNITION](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102957) | [code](https://github.com/yanglan0225/s3net) |
| NeurIPS2018 | [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf) | [code](https://github.com/RexYing/diffpool) or [code](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/diff_pool.py)|
| NeurIPS2020 | [DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation](https://arxiv.org/pdf/2007.11301.pdf) | [code](https://github.com/alexandre01/deepsvg) |
| ICLR2018 | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | [code](https://github.com/PetarV-/GAT) |
| IJCAI2019 | [DAEGC: Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/pdf/1906.06532.pdf) | [code](https://github.com/Tiger101010/DAEGC) |
| ICML2020 | [Mini-cut: Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/pdf/1907.00481.pdf) | [code](https://graphneural.network/layers/pooling/#mincutpool) or [code](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/mincut_pool.py)|
| KDD2019 | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) | [code](https://github.com/benedekrozemberczki/ClusterGCN) |
| ICCV2021 | [Graph Contrastive Clustering](https://arxiv.org/pdf/2104.01429.pdf) | [code](https://github.com/mynameischaos/GCC) |
| ECCV2020 | [Scan: Learning to classify images without labels](https://arxiv.org/pdf/2005.12320.pdf) | [code](https://github.com/wvangansbeke/Unsupervised-Classification) |
|  | [Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/pdf/1511.06335.pdf) | [code](https://github.com/piiswrong/dec) |
| CVPR2022 | [DeepDPM: Deep Clustering With an Unknown Number of Clusters](https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf) | [code](https://github.com/BGU-CS-VIL/DeepDPM) |
| CVPR2022| [DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf)| [code](https://github.com/lhoyer/DAFormer)|
| CVPR2022 | [Stratified Transformer for 3D Point Cloud Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)| [code](https://github.com/dvlab-research/StratifiedTransformer)|
| CVPR2022 | [Lifelong Graph Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Lifelong_Graph_Learning_CVPR_2022_paper.pdf) | [code](https://github.com/wang-chen/LGL) |
| CVPR2022 |  |  |
| CVPR2022 |  |  |
| CVPR2022 |  |  |
| CVPR2022 |  |  |


------------


## Referring Segmentation
|  |  |  |
| :------------: | :------------: | :------------: |
| ICCV2021 | [Vision-Language Transformer and Query Generation for Referring Segmentation](https://arxiv.org/abs/2108.05565) | [code](https://github.com/henghuiding/Vision-Language-Transformer) |
| CVPR2021 | [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://arxiv.org/abs/2105.01839) | [code](https://github.com/fengguang94/CEFNet) |
| CVPR2021 | [CRIS: CLIP-Driven Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/DerrickWang005/CRIS.pytorch) |
| CVPR2022 | [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | [code](https://github.com/yz93/LAVT-RIS) |
| CVPR2022 |  |  |
| CVPR2022 |  |  |
| CVPR2022 |  |  |
| CVPR2022 |  |  |


