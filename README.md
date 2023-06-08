# ENet-6mA
Identification of 6mA modification sites in Plant Genomes using ElasticNet and Neural Networks

## Abstract
N6-methyladenine (6mA) has been recognized as a key epigenetic alteration that affects a variety of biological activities. Precise prediction of 6mA modification sites is essential for understanding the logical consistency of biological activity. There are various experimental methods for identifying 6mA modification sites, but in silico prediction has emerged as a potential option due to the very high cost and labor-intensive nature of experimental procedures. Taking this into consideration, developing an efficient and accurate model for identifying N6-methyladenine is one of the top objectives in the field of bioinformatics. Therefore, we have created an in silico model for the classification of 6mA modifications in plant genomes. ENet-6mA uses three encoding methods, including one-hot, nucleotide chemical properties (NCP), and electron–ion interaction potential (EIIP), which are concatenated and fed as input to ElasticNet for feature reduction, and then the optimized features are given directly to the neural network to get classified. We used a benchmark dataset of rice for five-fold cross-validation testing and three other datasets from plant genomes for cross-species testing purposes. The results show that the model can predict the N6-methyladenine sites very well, even cross-species. Additionally, we separated the datasets into different ratios and calculated the performance using the area under the precision–recall curve (AUPRC), achieving 0.81, 0.79, and 0.50 with 1:10 (positive:negative) samples for F. vesca, R. chinensis, and A. thaliana, respectively.

![block](https://user-images.githubusercontent.com/80881943/179127910-bc88a2d1-13bc-40bc-b2d7-ee4ee0489bbb.PNG)

**For more details, please refer to the [paper](https://doi.org/10.3390/ijms23158314)**

## Steps
1. Data preprocessing (Apply encoding methods)
2. Apply ElasticNet 
3. Train using CNN model
4. Prepare Test data using the index numbers saved from training
5. Independent Testing



## Specifications
Python 3.7\
numpy 1.19.4\
pandas 1.1.0\
tensorflow 2.1.0\
keras 2.3.1
