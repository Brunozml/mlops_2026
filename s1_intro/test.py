import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
import matplotlib.pyplot as plt

# Test NumPy
print(f"NumPy version: {np.__version__}")
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}, mean: {arr.mean()}")

# Test Pandas
print(f"\nPandas version: {pd.__version__}")
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"Pandas DataFrame:\n{df}")

# Test scikit-learn
print(f"\nscikit-learn version: {sklearn_version}")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(f"Created model: {model}")

print("\n✅ All packages installed correctly!")