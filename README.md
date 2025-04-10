# vae_embeddings
*Exploring Gene Module Encoding and Latent Space Properties in scRNA-seq VAE Embeddings*

Rapid developments in single-cell omics technologies over the past two decades have provided an ever-growing pool of single-cell data. Using such techniques as well as combining several kinds of single-cell data (multimodal) provides insight into cellular heterogeneity, can help identify cell types and states as well as may enable biomarker discovery, thereby offering a novel way of understanding complex diseases [1]. 

One of the first methods arising in that field was single-cell RNA sequencing (scRNA-seq). Analyzing transcriptomic data can provide a deeper understanding of the complexity of gene expression, as well as the regulation and networks within individual cells [2]. 

Capturing the chromatin accessibility within single cells is another example of the mentioned technologies. Also known as Assay for Transposase-Accessible Chromatin using sequencing (ATAC-seq), this technique can assist in understanding the hierarchical structure of the genome by identifying regions of open chromatin and providing insights into gene regulatory mechanisms [3]. 

In this work, both scRNA-seq and ATAC-seq data of bone marrow mononuclear cells (BMMCs) were utilized to investigate the latent space of a Variational Autoencoder (VAE) trained on this dataset. BMMCs are a heterogeneous population of cells isolated from bone marrow, encompassing key stages of erythrocyte differentiation and B cell maturation, and undergoing chromatin remodeling, thereby making the chosen methods for investigating them highly informative. They are also relevant for studying hematological diseases and immune responses [4]. 

Multimodal single-cell data, such as scRNA-seq and ATAC-seq, as well as data from multiple batches, provide deeper biological insights, making horizontal and vertical data integration essential. The VAE model "leveraging information across modalities" (liam) [5] learned a shared latent space of the different data modalities while preserving meaningful biological variation. Additionally, its data integration capabilities help minimizing differences in data quality across modalities and can correct for complex batch effects and transcriptional noise using tunable combination of conditional and adversarial training [5].

The shared latent space, also known as the embedding space, represents the samples (cells) of the data set in a reduced dimensionality, in this case by 20 dimensions. Each dimension can capture specific biological properties, such as cell type or regulatory activity. A stable and smooth embedding maintains meaningful relationships within the data, enabling interpretability and reliable downstream analysis [6]. Disentanglement ensures that each dimension corresponds to a distinct biological factor, preventing mixed signals and improving analytical clarity [7].

This project focused on examining the embedding of a VAE model used for single-cell analysis on bone marrow mononuclear cells in biological ways as well as technical ways in order to assess its ability to capture interesting features:

1. Gene Module Encoding Analysis:
   
   1.1 [Correlation Analysis](https://github.com/olxssa/vae_embeddings/blob/main/1a_Correlation_Analysis.ipynb)
     
   1.2 [Gene Ontology (GO) Enrichment Analysis](https://github.com/olxssa/vae_embeddings/blob/main/1b_GO_Enrichment_Analysis.ipynb)

2. Embedding Space Properties Investigation:

   2.1 [Stability Analysis](https://github.com/olxssa/vae_embeddings/blob/main/2a_Stability_Analysis.ipynb)

   2.2 [Smoothness Analysis](https://github.com/olxssa/vae_embeddings/blob/main/2b_Smoothness_Analysis.ipynb)

   2.3 [Disentanglement Analysis](https://github.com/olxssa/vae_embeddings/blob/main/2c_Disentanglement_Analysis.ipynb)


References:
 
[1] Q. Shi, X. Chen, and Z. Zhang, “Decoding Human Biology and Disease Using Single-cell Omics Technologies,” Genomics, Proteomics & Bioinformatics, vol. 21, no. 5, p. 926, Sep. 20, 2023. DOI: 10.1016/j.gpb. 2023 . 06 . 003. pmid: 37739168. [Online]. Available: https : / / pmc . ncbi . nlm . nih . gov / articles / PMC10928380/ (visited on 03/21/2025)

[2] F. Tang, C. Barbacioru, Y. Wang, et al., “mRNA-Seq whole-transcriptome analysis of a single cell,” Nature Methods, vol. 6, no. 5, pp. 377–382, May 2009, ISSN: 1548-7105. DOI: 10.1038/nmeth.1315. [Online]. Available: https://www.nature.com/articles/nmeth.1315 (visited on 03/21/2025).

[3] J. D. Buenrostro, P. G. Giresi, L. C. Zaba, H. Y. Chang, and W. J. Greenleaf, “Transposition of native chromatin for fast and sensitive epigenomic profiling of open chromatin, DNA-binding proteins and nucleosome position,” Nature Methods, vol. 10, no. 12, pp. 1213–1218, Dec. 2013, ISSN: 1548-7105. DOI: 10.1038/nmeth.2688. [Online]. Available: https://www.nature.com/articles/nmeth.2688 (visited on 03/21/2025).

[4] M. Luecken, D. Burkhardt, R. Cannoodt, et al., “A sandbox for prediction and integration of dna, rna, and proteins in single cells,” vol. 1, J. Vanschoren and S. Yeung, Eds., 2021. [Online]. Available: https : / / datasets - benchmarks - proceedings . neurips . cc / paper _ files / paper / 2021 / file / 158f3069a435b314a80bdcb024f8e422-Paper-round2.pdf.

[5] P. Rautenstrauch and U. Ohler, “Liam tackles complex multimodal single-cell data integration challenges,” Nucleic Acids Research, vol. 52, 12, 2024. DOI: 10.1093/nar/gkae409.

[6] Y. Rosen, Y. Roohani, A. Agrawal, et al., “Universal cell embeddings: A foundation model for cell biology,” bioRxiv, 2024. DOI: 10 . 1101 / 2023 . 11 . 28 . 568918. eprint: https : / / www . biorxiv . org / content / early/2024/10/06/2023.11.28.568918.full.pdf. [Online]. Available: https://www.biorxiv. org/content/early/2024/10/06/2023.11.28.568918.

[7] X. Wang, H. Chen, S. Tang, Z. Wu, and W. Zhu, “Disentangled representation learning,” 2024. arXiv: 2211.11695 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2211.11695.
