import pickle
import numpy as np
import json

# Load the scaler from the pickle file
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Extract the mean and scale values
mean_values = scaler.mean_.tolist()
scale_values = scaler.scale_.tolist()

# Create a dictionary with the scaler parameters
scaler_data = {
    'mean': mean_values,
    'scale': scale_values,
    'n_features': len(mean_values)
}

# Save as JSON for Unity
with open('scaler_data.json', 'w') as f:
    json.dump(scaler_data, f, indent=2)

# Also save as a simple text file for easy copying
with open('scaler_data_for_unity.txt', 'w') as f:
    f.write("// Copy these values into your Unity script\n")
    f.write("// scalerMean values:\n")
    f.write("scalerMean = new float[] {\n")
    for i, val in enumerate(mean_values):
        f.write(f"    {val:.6f}f")
        if i < len(mean_values) - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")
    
    f.write("// scalerScale values:\n")
    f.write("scalerScale = new float[] {\n")
    for i, val in enumerate(scale_values):
        f.write(f"    {val:.6f}f")
        if i < len(scale_values) - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n")

print("Scaler data exported successfully!")
print(f"Mean values: {len(mean_values)} features")
print(f"Scale values: {len(scale_values)} features")
print("Files created:")
print("- scaler_data.json (JSON format)")
print("- scaler_data_for_unity.txt (Unity C# format)")
print("\nTo use in Unity, copy the arrays from scaler_data_for_unity.txt")
print("into the LoadScalerParameters() method in your Unity script.")
