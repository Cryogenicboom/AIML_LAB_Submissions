import math
import numpy as np
from scipy import integrate

#  MATH LIBRARY

number = 16
print(f"Square root of {number}: {math.sqrt(number)}")
print(f"Factorial of 5: {math.factorial(5)}")
print(f"Sine of pi/2: {math.sin(math.pi / 2)}\n")

# NUMPY

arr = np.array([1, 2, 3, 4, 5])
print(f"Original Array: {arr}")
print(f"Array squared: {arr ** 2}")
print(f"Mean of Array: {np.mean(arr)}")
print(f"Standard Deviation: {np.std(arr)}\n")

# SCIPY 
# Example: Single integration of a function f(x) = x^2 from 0 to 4
func = lambda x: x**2
integration_result, error = integrate.quad(func, 0, 4)
print(f"Numerical integration of x^2 from 0 to 4: {integration_result}")\

import pandas as pd
import matplotlib.pyplot as plt

# PANDAS
data = {
    'Product': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
    'Sales_USD': [120, 85, 150, 140, 95, 160],
    'Region': ['East', 'East', 'East', 'West', 'West', 'West']
}

df = pd.DataFrame(data)
print("=== Raw Dataset ===")
display(df)

# Grouping and aggregating data
summary_df = df.groupby('Product')['Sales_USD'].sum().reset_index()
print("\n=== Total Sales by Product ===")
display(summary_df)

# MATPLOTLIB
plt.figure(figsize=(8, 5))
plt.bar(summary_df['Product'], summary_df['Sales_USD'], color=['crimson', 'gold', 'orange'])
plt.title('Total Sales by Product Category')
plt.xlabel('Product')
plt.ylabel('Sales (USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
