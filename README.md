# Predicting Customer Churn Through Latent Class Analysis of Retail Transaction Data

**Overview**

This project aims to predict customer churn for a retail business by employing Latent Class Analysis (LCA) on transactional data.  The analysis identifies distinct customer segments exhibiting varying churn probabilities. This segmentation allows for the development of targeted retention strategies to maximize customer lifetime value and improve business profitability.  The project utilizes statistical modeling and data visualization techniques to uncover actionable insights from the retail transaction data.

**Technologies Used**

* Python 3
* Pandas
* NumPy
* Scikit-learn (for LCA)
* Matplotlib
* Seaborn

**How to Run**

1. **Install Dependencies:**  Ensure you have Python 3 installed. Then, navigate to the project directory in your terminal and install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis:** Execute the main script using:

   ```bash
   python main.py
   ```

**Example Output**

The script will print key findings of the Latent Class Analysis to the console, including the number of identified customer segments, their characteristic features (e.g., average transaction value, purchase frequency), and estimated churn probabilities for each segment.  Additionally, the project generates several visualization files (e.g., plots showing the distribution of key variables across segments) in the `output` directory.  These visualizations provide a visual representation of the identified customer segments and their characteristics.  The specific files generated may vary depending on the data and analysis performed.