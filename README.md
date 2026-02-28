# DS-Lab 📊

> A comprehensive Data Science Laboratory collection featuring R and Python implementations designed for learning and practicing fundamental and advanced data science concepts.

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)


## 📝 Description

DS-Lab is a meticulously structured collection of data science laboratory experiments and exercises. It aims to provide practical experience across various data science concepts, ranging from foundational data manipulation to sophisticated statistical analysis and machine learning preprocessing techniques. The repository offers dual implementations in both R and Python, making it an invaluable educational resource for students, researchers, and practitioners looking to deepen their understanding and enhance their skills in data science.


## 🌟 Key Features

The DS-Lab project encompasses a wide array of functionalities and learning modules:

*   **Basic Vector Operations (R)**: Fundamental operations on vectors, array manipulation, and essential R programming constructs.
*   **Data Type Conversion and Encoding Techniques (Python/R)**: Practical examples of converting data types and implementing encoding schemes like Label Encoding and One-Hot Encoding to prepare categorical data for machine learning models.
    *   **One-Hot Encoding**: Demonstrated using both `pandas.get_dummies` and `sklearn.preprocessing.OneHotEncoder` for comprehensive understanding.
*   **Statistical Analysis Tools**: Modules for performing various statistical tests and analyses.
*   **Model Comparison Implementations**: Examples of comparing different statistical or machine learning models to evaluate performance.
*   **Univariate and Bivariate Analysis**: Techniques for exploring single variables and relationships between two variables.
*   **Descriptive Statistical Analysis**: Methods to summarize and describe the main features of a dataset.
*   **Excel Data Processing Capabilities**: Tools and scripts designed for reading, processing, and analyzing data stored in Excel formats.
*   **Comprehensive Laboratory Modules**: Each `Lab` directory contains specific exercises and solutions, often provided in both R and Python.


## 🛠️ Tech Stack

This project leverages a diverse set of technologies to provide robust data science functionalities:

*   **R Programming Language**: Utilized for statistical computing, data analysis, and advanced graphics.
    *   Key packages from `r_requirements.txt`: `readr`, `tibble`, `dplyr`, `ggplot2`.
*   **Python Programming Language**: Employed for general-purpose programming, data manipulation, and machine learning.
    *   Key packages: `pandas`, `scikit-learn`.
*   **Jupyter Notebooks**: Used for interactive computing, allowing for easy execution and visualization of code in both R and Python environments.


## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have R and Python environments set up on your machine.

*   **R Installation**:
    Download and install R from [CRAN](https://cran.r-project.org/).
*   **Python Installation**:
    Download and install Python from [python.org](https://www.python.org/). It is recommended to use `conda` or `venv` for environment management.
*   **Jupyter Installation**:
    ```bash
    pip install jupyterlab
    # Or for R kernel
    # install.packages('IRkernel')
    # IRkernel::installspec()
    ```

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/DS-Lab.git
    cd DS-Lab
    ```

2.  **Set up Python dependencies**:
    It is highly recommended to create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` is not provided in the prompt, but inferred for Python projects. If a Python `requirements.txt` is created, specify its contents here.)*

3.  **Install R dependencies**:
    Open your R console and install the necessary packages listed in `r_requirements.txt`:
    ```R
    install.packages(c("readr", "tibble", "dplyr", "ggplot2"))
    ```


## 🏃 Usage Examples

Navigate into the respective `Lab` directories to find specific examples and exercises. Each lab typically contains R scripts (`.R`), Python scripts (`.py`), or Jupyter notebooks (`.ipynb`).

### Example 1: One-Hot Encoding (Python)

This example demonstrates how to perform One-Hot Encoding on categorical data using both `pandas` and `scikit-learn`.

**File**: `Lab2/label_encoding_one_hot_encoding.py`

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Original Data
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})
print("Original Data:")
print(data)

# One-Hot Encoding with pandas get_dummies
encoded_pandas = pd.get_dummies(data, columns=["Color"])
print("\nOne-Hot Encoding with pandas.get_dummies:")
print(encoded_pandas)

# One-Hot Encoding with sklearn
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(data[['Color']])

encoded_sklearn = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(['Color'])
)
print("\nOne-Hot Encoding with sklearn OneHotEncoder:")
print(encoded_sklearn)
```

To run this example:

```bash
python Lab2/label_encoding_one_hot_encoding.py
```

This script will output the original data, followed by the one-hot encoded versions using both `pandas.get_dummies` and `sklearn.preprocessing.OneHotEncoder`, illustrating the differences and similarities between the two approaches.

### Example 2: Data Manipulation (R - `dplyr`)

Although no specific R script was provided in the context for `dplyr` usage, based on `r_requirements.txt`, `dplyr` is a core dependency for data manipulation. A typical usage might involve filtering and summarizing data.

```R
library(readr)
library(dplyr)
library(tibble)

# Create a sample tibble (similar to pandas DataFrame)
data_r <- tibble(
  ID = 1:5,
  Category = c("A", "B", "A", "C", "B"),
  Value = c(10, 15, 12, 18, 13)
)

print("Original Data in R:")
print(data_r)

# Filter data where Category is 'A' and select specific columns
filtered_data <- data_r %>%
  filter(Category == "A") %>%
  select(ID, Value)

print("\nFiltered Data (Category 'A'):")
print(filtered_data)

# Summarize data by Category
summary_data <- data_r %>%
  group_by(Category) %>%
  summarize(AverageValue = mean(Value), Count = n())

print("\nSummary by Category:")
print(summary_data)
```

To run this R example, you would typically save it as an `.R` file (e.g., `LabX/r_data_manipulation.R`) and execute it from your R console or terminal:

```bash
Rscript LabX/r_data_manipulation.R
```

These examples showcase the practical application of the tools and libraries integrated into the DS-Lab project. Explore the various `Lab` directories for more specific implementations and detailed exercises.


## 📂 Project Structure

The repository is organized into distinct laboratory modules, each typically residing in its own `LabX` directory (e.g., `Lab1`, `Lab2`, etc.).

```
DS-Lab/
├── Lab1/
│   ├── ... # R and Python scripts for Lab 1
├── Lab2/
│   ├── label_encoding_one_hot_encoding.py
│   ├── ... # Other scripts for Lab 2
├── r_requirements.txt # R package dependencies
├── requirements.txt   # Python package dependencies (inferred/planned)
└── README.md          # This documentation file
```

Each `LabX` directory is expected to contain a set of files related to specific data science topics, including code examples and possibly datasets.


## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit a pull request with your enhancements, bug fixes, or new lab exercises. Ensure your code adheres to standard practices and includes appropriate documentation and tests.


## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Note: A `LICENSE` file is not currently provided, but would typically be included in open-source projects).


## 📞 Contact

For any questions or suggestions, please open an issue in the repository or reach out to the maintainers.

---