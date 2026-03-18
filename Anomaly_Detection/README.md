# NOTES

## Requirements
I used python3.12 and installed pandas==2.2.3 matplotlib==3.9.4 scikit-learn==1.6.0 jupyter==1.1.1 plotly==5.24.1 tensorflow==2.18.0


## Anomaly detection
**Anomaly detection** AKA outlier detection, is a technique used to identify rare or unusual patterns in data that do not conform to the expected behavior. These irregularities, called anomalies, can indicate significant events, errors, or potentially malicious activities.


## Applications of Anomaly Detection 
Anomaly detection is widely used in various domains, such as:

1.  **Fraud Detection**: In banking and finance, anomalies in transactions can signal potential fraud.
2.	**Network Security**: Identifying unusual patterns in network traffic to detect cyberattacks or data breaches.
3.	**Industrial Equipment Monitoring**: Detecting faults or failures in machinery by identifying deviations from normal operation patterns.
4.	**Healthcare**: Identifying unusual medical data for diagnosing rare diseases or patient health issues.
5.	**Quality Control**: Spotting defective products in manufacturing processes.
6.	**Video Surveillance**: Identifying suspicious activity in security footage.


## Types of Anomalies
1.	**Point Anomalies**: A single instance of data that is significantly different from the rest. Example: A sudden spike in credit card transactions at an unusual time.
2.	**Contextual Anomalies**: Anomalies that are context-dependent. Example: A temperature reading of 40°C may be normal in summer but anomalous in winter.
3.	**Collective Anomalies**: A group of data points that are anomalous when considered together, even though individual points may seem normal. Example: A pattern of network activity that indicates a coordinated cyberattack

## Techniques for Anomaly Detection
Several methods can be used for anomaly detection, including:
1.	**Statistical Methods**: These rely on probabilistic models and assume normal data distribution.
    * Example: Z-scores, hypothesis testing.
2.	**Machine Learning Methods**: 
    * Supervised Learning: Requires labeled data (normal and abnormal). Algorithms like SVM (Support Vector Machines) and classification models can be used.
    * Unsupervised Learning: Works with unlabeled data by identifying deviations from the majority. Popular methods include: K-Means Clustering, DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Isolation Forests
3.	**Deep Learning**: Neural networks like Autoencoders and LSTM (Long Short-Term Memory) networks are used for complex anomaly detection in high-dimensional data.
4.	**Distance-Based Methods**: Identify anomalies by calculating the distance of data points from clusters or neighbors. Example: k-Nearest Neighbors (kNN).
5.	**Density-Based Methods**: Examine the density of points in a data space to detect regions of low density (potential anomalies).
    * Example: LOF (Local Outlier Factor).
  
## Steps in Anomaly Detection
1.	Data Collection: Gather relevant data.
2.	Preprocessing: Clean and normalize the data.
3.	Feature Extraction: Select the features relevant for detecting anomalies.
4.	Model Selection: Choose the appropriate algorithm.
5.	Training (if supervised): Train the model on labeled data.
6.	Detection: Apply the model to identify anomalies.
7.	Evaluation: Measure the model's performance (Precision, Recall, F1 Score).

## Challenges in Anomaly Detection
1.	**Imbalanced Data**: Anomalies are typically rare compared to normal data, making it hard to train effective models.
2.	**High-Dimensional Data**: In some cases, data may have many features, increasing complexity.
3.	**Concept Drift**: Data distributions change over time, requiring model updates.
4.	**False Positives/Negatives**: Incorrectly flagging normal data as anomalies or missing real anomalies


