# detectfakeNews

The goal of this project was to explore and implement various machine learning models 
to address the task of detecting fake news.

For this project, my teammate and I developed multiple Python based machine learning 
solutions to classify news articles as real or fake using AI models.


I contributed by doing the K-Nearest Neighbor (KNN) and Recurrent Neural Network (RNN) 
models for our classification.

I evaluated and compared model performances on a LIAR dataset of a labeled news article.

I did preprocessing techniques to clean and prepare the data for model training and testing.

KNN (K-Nearest Neighbor): A simple yet effective algorithm that classifies data points based on their proximity to other labeled points.
RNN (Recurrent Neural Network): A neural network designed to capture sequential patterns in data.

Technologies Used:

Programming Language: Python.
Libraries: NumPy, Pandas, Scikit-learn, TensorFlow/Keras.
Data: LIAR dataset.

How both of these models Work:

Data Preprocessing: The raw text data is cleaned, tokenized, and transformed into numerical representations (0 for false and 1 for true) based on a threshold score of 0.75.
The score was extremely close but not exact.
Model Training: Both the KNN and RNN models were trained on the preprocessed data.
Model Evaluation: The models were evaluated using metrics such as accuracy, precision, recall, and F1-score to determine their effectiveness.





