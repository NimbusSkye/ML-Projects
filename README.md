# ML-Projects
My ML Papers

![Neural Networks for Image Classification](https://github.com/NimbusSkye/ML-Projects/blob/main/mlp_paper.pdf)
* Used TensorFlow and a multi-layer perceptron to vectorize and classify 70,000 images of clothing from Fashion-MNIST
* Ran hyperparameter tuning in MatPlotLib by plotting epochs vs accuracy curves for models with differing hyperparameters
* More epochs always correlated with better training accuracy, but test accuracy plateaued after 20 epochs in all cases
* Observed higher dropout’s suitability for large batch sizes, little benefit of additional hidden layers
* Learning rates greatly affected the performance of a CNN with Adam GD but barely affected that of a CNN with SGD
* Identified MLP with 1 hidden layer, ReLU activation, and no dropout as the best MLP model with 89% test accuracy
* A CNN with a small learning rate and Adam gradient descent outperformed all MLP models at 90% test accuracy

![Double Descent and Over-parameterization](https://github.com/NimbusSkye/ML-Projects/blob/main/Double%20Descent.pdf)
* Explored “Double Descent” phenomenon where over-parameterized models generalize better, contrary to “overfitting”
* Used NumPy to generate synthetic data with various noise structures to assess effects of noise on deep learning models
* Plotted MSPE of models such as ridge regression and principal component regression on datasets with noise structures such
as double-gapped, isotropic, or equicorrelation against increasing numbers of parameters k
* Observed peak MSPE at k=n and optimal MSPE afterward for most algorithms on equicorrelation and unstructured noise
* Ridge regression (ortho+ridge) was the best overall model, outperforming or tying with all other models on all datasets

![Naive Bayes and Logistic Regression for Text Classification](https://github.com/NimbusSkye/ML-Projects/blob/main/NB-LR%20paper.pdf)
* Used Naïve Bayes and Logistic Regression to classify 18,000 emails and 1,600,000 tweets from 20 Newsgroups and Sentiment140 datasets, analyzed performances of both algorithms and preprocessing methods to boost them
* Used Scikit-Learn to vectorize and clean textual data by removing stopwords and symbols in preparation for machine learning algorithms, used MatPlotLib to graph distribution of categories and produce TFIDF plots
* Noted standout traits of both datasets, such as the word “ax” dominating 20 Newsgroups because it was an image encoding
* Invoked GridSearchCV to optimize both models on both datasets; 20 Newsgroups favored LR and Sentiment140 favored NB
* Both algorithms enjoyed up to a 10% performance boost via textual cleaning, case insensitivity, and lemmatization