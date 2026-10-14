import numpy as np
import pandas as pd

def generate_realizations(num_nodes=125, p=0.90, num_realizations=1000):
    """
    Generate 1000 realizations of faulty nodes (1-based indexing).
    Node 1 is always faulty by assumption.
    """
    all_realizations = []
    for _ in range(num_realizations):
        realization = [1]  # Node 1 is always faulty
        for node in range(2, num_nodes + 1):
            if np.random.rand()< p:
                realization.append(node)
        all_realizations.append(realization)
    return all_realizations

# Generate and print sample
realizations = generate_realizations()
# Wrap into a DataFrame to match uploaded format
df = pd.DataFrame({
    0: range(len(realizations)),
    1: [str(r) for r in realizations]
})

# Add header row like the uploaded file
header = pd.DataFrame({0: [""], 1: ["realization"]})
df_final = pd.concat([header, df], ignore_index=True)

# Save to CSV
df_final.to_csv("Realization_0.90_Network125.csv", index=False, header=False)