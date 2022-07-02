# MusicGenreClassification
### [Full report](./MusicGenreClassification.pdf)
## Abstract
In this article we present our research on applying different methods for music genre classification on a popular database GTZAN. We show our pipeline for extracting features from raw audio signal and scaling them into usable data. Then we list and describe different traditional methods like k-NN, SVM, Random forest, etc. which use these features. We also review application of deep neural networks in this domain. From our results we can deduce that SVM is the most suitable traditional model with classification accuracy of 78.4%, while CNNs work best when input is sub-sampled and majority voting is introduced at the end. Here our classification accuracy is 93.6%.

## TODO
- [x] Data exploration
  - [x] Read music file
  - [x] Vizualize waveform
  - [x] Extract time domain features
  - [x] Extract frequency domain features
- [x] Extract features from data and save them in dataframe
- [x] Vizualize with PCA or t-SNE
- [x] Run traditional machine learning algorithms (k-NN, Random Forest, Gradient Boosting, SVM, Logistical Regression)
  - [x] Preliminary run with default parameters
  - [x] Evaluate features with relifF or similar 
  - [x] Find the best parameters for each method
  - [x] Save results
- [x] Extract image features from data
- [x] Run different CNNs on image data
  - [x] Run my CNN
  - [x] Transfer learning
  - [x] CNN + SVM
  - [x] CNN on 3s clips
  - [x] Evaluate and compare 

## Helpful Links and References

- [Music Genre Classification: A Review of Deep-Learning and Traditional Machine-Learning Approaches](https://ieeexplore.ieee.org/document/9422487)
- [Music Genre Classification and Recommendation by Using Machine Learning Techniques](https://ieeexplore.ieee.org/document/8554016)
- [Music Genre Classification: A Comparative Study Between Deep-Learning And Traditional Machine Learning Approaches](https://www.riteshajoodha.co.za/sitepad-data/uploads/2021/02/2020-Dhiven.pdf)
- [Short Time Fourier Transform based music genre classification](https://www.researchgate.net/publication/325917674_Short_Time_Fourier_Transform_based_music_genre_classification)
- [Musical Genre Classification of Audio Signals](https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)
