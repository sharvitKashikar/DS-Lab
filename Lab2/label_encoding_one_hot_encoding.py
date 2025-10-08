# Label Encoding / One-Hot Encoding

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
