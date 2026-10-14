import pandas as pd

input_file = "Network100_Real.txt"
output_file = "Network100_Real_with_node0.csv"

# Read txt (space-separated)
df = pd.read_csv(input_file, delim_whitespace=True, header=None)

# Select the relevant columns
df_nodes = df[[0, 2, 3]]
df_nodes.columns = ["node", "x", "y"]

# Compute mean of x and y
mean_x = df_nodes["x"].mean()
mean_y = df_nodes["y"].mean()

print(mean_x, mean_y)

# Create node 0 row
node0 = pd.DataFrame({"node": [0], "x": [mean_x], "y": [mean_y]})

# Append node 0 to the dataframe
df_final = pd.concat([node0, df_nodes], ignore_index=True)

# Save to CSV
df_final.to_csv(output_file, index=False)

# Get coordinates of node 54
node54 = df_nodes.loc[df_nodes["node"] == 54, ["x", "y"]].iloc[0]
print(node54)
# Compute Euclidean distance
dist = np.sqrt((mean_x - node54["x"])**2 + (mean_y - node54["y"])**2)

print(f"Distance between node 0 and node 54 = {dist:.3f}")