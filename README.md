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
This code implements a **Self-Organizing Map (SOM)**, a type of unsupervised neural network used for clustering and dimensionality reduction. SOM maps high-dimensional data to a lower-dimensional space (usually 2D) while maintaining the topological structure of the data. Here’s a detailed explanation of the code:

### **Key Components:**

1. **Imports:**
   - The necessary modules (`numpy`, `math`, `collections`, `warnings`) are imported for matrix manipulation, mathematical operations, and utility functions used throughout the SOM implementation.
   
2. **`fast_norm` function:**
   - This function computes the Euclidean (L2) norm of a 1D numpy array using `dot` product. It is faster than `linalg.norm` for 1D arrays. The Euclidean norm is used to calculate the distance between vectors.

   ```python
   def fast_norm(x):
       return sqrt(dot(x, x.T))
   ```

3. **`MiniSom` Class:**
   - The core class that implements the Self-Organizing Map.

   #### **Class Attributes:**
   - **`x` and `y`**: Dimensions of the SOM grid.
   - **`input_len`**: Length of the input vectors.
   - **`sigma`**: Spread of the Gaussian neighborhood function, which determines how neighboring neurons are affected when a neuron "wins" during training.
   - **`learning_rate`**: Initial learning rate, which decreases over time.
   - **`decay_function`**: Function to decay the learning rate and `sigma` over iterations.
   - **`weights`**: The weights of the neurons in the SOM. These are randomly initialized between -1 and 1, and then normalized using the `fast_norm` function.
   - **`activation_map`**: A matrix that stores the activation (response) of the neurons to the input data.
   
   #### **Key Methods:**
   
   - **`_activate`**: This method calculates the activations of the SOM for a given input vector by computing the distance between the input and the weights of each neuron.
   
   - **`winner`**: Identifies the neuron with the smallest distance to the input vector, also known as the **Best Matching Unit (BMU)**. This neuron "wins" for a given input.
   
   - **`update`**: Adjusts the weights of the winning neuron and its neighbors using the neighborhood function. The amount by which the weights are adjusted depends on the distance from the winning neuron, the learning rate, and the neighborhood function.
   
   - **`gaussian`**: Defines the neighborhood function as a Gaussian (bell curve), which controls how strongly the neighboring neurons are updated when a neuron "wins." Neurons closer to the winner are updated more significantly.
   
   - **`train_random`**: Trains the SOM by randomly picking samples from the data and updating the weights. The number of iterations is specified by `num_iteration`.
   
   - **`train_batch`**: Trains the SOM by using all vectors in the dataset sequentially.

   - **`distance_map`**: Computes and returns a **U-matrix**, which represents the distances between neighboring neurons. The U-matrix is useful for visualizing the SOM’s clustering structure.
   
   - **`quantization_error`**: Computes the average distance between input vectors and their closest neurons (quantization error), which indicates how well the SOM has learned the input data.

   - **`win_map`**: Returns a dictionary where the keys are coordinates of the neurons and the values are the input vectors that were mapped to those neurons. This allows you to see which inputs were mapped to which neurons.

### **Unit Tests (`TestMinisom` Class):**

- The `TestMinisom` class is a set of unit tests to ensure the functionality of various aspects of the SOM.

#### Key Unit Tests:
   - **`test_decay_function`**: Checks if the learning rate and sigma decay function works correctly.
   - **`test_gaussian`**: Verifies the Gaussian neighborhood function is computed correctly.
   - **`test_win_map`**: Ensures that the `win_map` method correctly maps input vectors to their winning neurons.
   - **`test_activation_response`**: Tests the activation response of neurons to inputs.
   - **`test_quantization_error`**: Validates the quantization error calculation.
   - **`test_random_seed`**: Ensures that the random seed provides reproducible results, meaning SOM initialization and training should yield the same results if the same seed is used.
   - **`test_train_random` and `test_train_batch`**: Ensure that SOM training (random and batch) reduces the quantization error as expected.

### **Working Mechanism:**

1. **Initialization:**
   - The SOM is initialized with a grid of neurons where each neuron has a weight vector initialized randomly. These weights are normalized using `fast_norm`.

2. **Training Process:**
   - During training, for each input vector, the SOM:
     - Computes the neuron whose weights are closest to the input vector (winner neuron).
     - Updates the winner and its neighboring neurons by adjusting their weights to become more similar to the input vector. The amount of weight adjustment depends on the learning rate and the distance from the winning neuron.
   - The learning rate and neighborhood size decay over time to allow the network to converge gradually.

3. **Post-Training:**
   - After training, the SOM can be used to map new input data to the trained neurons. You can visualize the clusters using the distance map (U-matrix) or analyze which neurons win for specific inputs using the `win_map`.

### **Applications:**
SOMs are used for tasks like:
- **Clustering**: Grouping similar data points together.
- **Dimensionality reduction**: Reducing the number of dimensions while preserving the structure of the data.
- **Visualization**: Mapping high-dimensional data to a 2D grid for easy visualization.

In this code, SOM’s potential applications include clustering and visualizing complex datasets, like customer segmentation in marketing or anomaly detection in sensor networks.



