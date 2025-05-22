# BSc Project: Estimating multi-topic polarization on Reddit network (Spring 2025)

## 🧠 Overview
This project aims to compute multi-topic polarization on a network of users of the platform Reddit. To extract users' stances from their messages, we use topic modelling and stance detection techniques, leveraging several transformers-based models.

## 🚀 Methods and models
- ✅ Topic modelling with BERTopic
- ✅ Stance detection. To extract the opinion scores of users on each of the selected topics without a large dataset of labelled, we experimented with several methods:
    - Baseline method without the use of NLP; based on upvotes and downvotes from each subreddit.
    - Zero-shot classification with BART-MNLI-large
    - Few-shot classification with SetFit
    - ⭐Classification using prompting of GenAI (Llama) <- the final selected model
- ✅ Dimensionality reduction of stance vectors using PCA and UMAP.
- ✅ Analysis of performance on a test set we labelled ourselves.
- ✅ Computing polarization using Generalized Euclidean distance measure.
- ✅ Network analysis and comparison of results of 2 months of data
- ✅ Visualization of embeddings clusters and stance score distributions.

## 📂 Project Structure
```plaintext
├── annotated/              # Our own annotated subset used for model comparison
├── models/                 # Saved BERTopic models
├── notebooks/              # Main code for analysis and modelling
├── topic_modelling/        # experiments in topic modelling 
│   ├── outputs
│   ├── reclustered         # example topics after different reclustering methods
├── requirements.txt        # Dependency list
├── environment.yml         # to re-create the conda environment
```

## ⚙️ How to run

> ‼️Some parts of the project were conducted using high-performance computing (HPC) to make use of GPUs. The data was provided by our supervisor, it may not be included in its entirety.

### Environment setup

```bash
  git clone https://github.com/b0psheva27/BSc_project_polarization.git 
  cd BSc_project_polarization

  #to recreate the conda environment
  conda env create -f environment.yml
``` 

## 📊 Results summary

### Models performance

| Model      | Accuracy | F1-score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| Baseline   | 0.410    | 0.396    | 0.458     | 0.407  |
| Zero-shot  | 0.410    | 0.332    | 0.278     | 0.414  |
| Setfit     | 0.540    | 0.532    | 0.569     | 0.545  |
| Llama  ⭐    | 0.620    | 0.615    | 0.634     | 0.620  |

### Polarization

PCA was deemed more appropriate for dimensionality reduction in our context.
Polarization on the final reduced stance scores is:
- **June network**: 27.430
- **December network**: 15.982

Analysis into the distribution of opinions and the structural aspects of the networks supports the findings.
