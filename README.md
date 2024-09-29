# Customer_Segmentation
Code to perform clustering using self organizing maps on retail customer data.

code description :(clustering)
This code processes customer product data, trains a Self-Organizing Map (SOM), and then performs hierarchical clustering on the SOM neuron weights. It aims to segment customers into clusters based on the available data. Here's a breakdown of each section:

### 1. **Data Reading and Cleaning:**
   - The code begins by reading a CSV file containing product data using `pd.read_csv()`.
   - Several steps are performed to clean the data:
     - The `customerID` is cleaned by removing a prefix (`BBID_`) and converted to numeric values.
     - Redundant columns are dropped, including `store_description`, `product_description`, `promotion_description`, and `State`.
     - Missing values in the `DOB` (date of birth) and `PinCode` fields are removed.
     - Some categorical fields like `Gender` and `promo_code` are cleaned and filled with placeholder values where necessary.
     - Dates (`transactionDate` and `DOB`) are converted to proper datetime objects.
     - Age is calculated from the `DOB` field and added as a new column.
     - Other date-related features such as `Week` and `Month` are extracted from the `transactionDate`.
     - Finally, missing values in the `Age` column are dropped.

### 2. **Data Encoding and Standardization:**
   - The code uses `LabelEncoder` to encode categorical variables like `discountUsed` and `Gender` into numeric formats.
   - Standardization of the dataset is performed using `StandardScaler`, which normalizes the feature values for better SOM training.

### 3. **Self-Organizing Map (SOM) Training:**
   - A 15x15 SOM is initialized using `MiniSom`.
   - The SOM is trained on the standardized data for 10,000 iterations. SOMs are unsupervised neural networks that project high-dimensional data into a 2D grid of neurons while preserving data structure.

### 4. **SOM Visualization:**
   - The SOM distance map is plotted using `pcolor()`. Each neuron's distance from its neighbors is visualized to show the similarity structure across the grid.
   - Markers (`o`, `s`) are used to represent data points on the map. This visualization helps interpret the results of the SOM training.

### 5. **Hierarchical Clustering:**
   - After training the SOM, the weights of each neuron (a 2D array representing learned features) are extracted and converted to a list for clustering.
   - Hierarchical clustering is performed using the `linkage` method from `scipy` with the "average" linkage criterion. This clusters the neuron weights based on their proximity.
   - A dendrogram is plotted to visualize the hierarchical structure of the clusters.

### 6. **Fancy Dendrogram:**
   - A custom function `fancy_dendrogram` is defined to visualize a truncated dendrogram with annotations and optional cut-off distance lines (`max_d`).
   - This helps in determining the optimal number of clusters.

### 7. **Clustering Based on Dendrogram:**
   - The dendrogram suggests dividing the neurons into 8 clusters. Based on this, the code assigns neurons (SOM map positions) to 8 clusters.
   - For each cluster, the data points corresponding to those neurons are extracted and inverse transformed (i.e., reverted to original scale using `StandardScaler.inverse_transform`).

### 8. **Saving Results:**
   - The clustered data is concatenated and saved into a CSV file (`result.csv`).

### Purpose:
   - The overall goal is to clean and process customer data, project it onto a SOM, and then cluster the results for customer segmentation. The final clusters can be used for marketing or analysis of customer behavior.

### Key Libraries Used:
numpy: For numerical operations and array manipulations.
pandas: For data loading, cleaning, and manipulation.
scikit-learn: For encoding categorical variables and standardizing the dataset.
MiniSom: For training the Self-Organizing Map.
scipy: For hierarchical clustering and dendrogram generation.
matplotlib.pylab: For plotting the SOM map and dendrogram.
   - **`numpy`**: For numerical operations and array manipulations.
   - **`pandas`**: For data loading, cleaning, and manipulation.
   - **`scikit-learn`**: For encoding categorical variables and standardizing the dataset.
   - **`MiniSom`**: For training the Self-Organizing Map.
   - **`scipy`**: For hierarchical clustering and dendrogram generation.
   - **`matplotlib.pylab`**: For plotting the SOM map and dendrogram.


code description:(minisom)

