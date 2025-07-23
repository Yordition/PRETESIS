import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os


def calculate_lrei_qlrei(csv_path, window_size_m, elements):
    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Ensure coordinate columns exist
    required_columns = ['EJE_X', 'EJE_Y']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'EJE_X' and 'EJE_Y' columns")

    # Ensure all selected elements exist in the CSV
    if not all(elem in data.columns for elem in elements):
        raise ValueError("Some specified elements are not in the CSV columns")

    # Extract coordinates
    coords = data[['EJE_X', 'EJE_Y']].values
    tree = cKDTree(coords)

    # Initialize output DataFrames for LREI and QLREI
    lrei_output = data[['EJE_X', 'EJE_Y']].copy()
    qlrei_output = data[['EJE_X', 'EJE_Y']].copy()

    # Calculate LREI and QLREI for each element
    for element in elements:
        values = data[element].values
        lrei_values = np.zeros(len(values))

        # Calculate LREI for each point
        for i in range(len(coords)):
            # Query neighbors within window_size_m
            indices = tree.query_ball_point(coords[i], window_size_m)

            # Calculate median of element values within the window
            window_values = data.iloc[indices][element]
            median_j = np.median(window_values)

            # Avoid division by zero
            if median_j == 0:
                lrei_values[i] = np.nan
            else:
                # Calculate LREI: (Xi / MEDIANj) - 1
                lrei_values[i] = (values[i] / median_j) - 1

        # Add LREI to output
        lrei_output[f'LREI_{element}'] = lrei_values

        # Calculate QLREI: Normalize LREI to [0, 1] using min-max scaling
        # Handle NaN values by excluding them from min/max calculation
        valid_lrei = lrei_values[~np.isnan(lrei_values)]
        if len(valid_lrei) > 0:
            lrei_min = np.min(valid_lrei)
            lrei_max = np.max(valid_lrei)
            if lrei_max != lrei_min:  # Avoid division by zero
                qlrei_values = (lrei_values - lrei_min) / (lrei_max - lrei_min)
            else:
                qlrei_values = np.zeros(len(lrei_values))  # If all values are the same, set QLREI to 0
        else:
            qlrei_values = np.full(len(lrei_values), np.nan)  # If all values are NaN

        # Ensure NaN values remain NaN in QLREI
        qlrei_values[np.isnan(lrei_values)] = np.nan

        # Add QLREI to output
        qlrei_output[f'QLREI_{element}'] = qlrei_values

    # Save outputs to the same directory as input CSV
    output_dir = os.path.dirname(csv_path)
    lrei_output_file = os.path.join(output_dir, 'lrei_output.csv')
    qlrei_output_file = os.path.join(output_dir, 'qlrei_output.csv')

    lrei_output.to_csv(lrei_output_file, index=False)
    qlrei_output.to_csv(qlrei_output_file, index=False)

    print(f"LREI output saved to {lrei_output_file}")
    print(f"QLREI output saved to {qlrei_output_file}")

    return lrei_output, qlrei_output


# Main execution
if __name__ == "__main__":
    # Get user input for CSV path
    csv_path = input("Porfavor pegar la ruta del archivo CSV: ")

    # Read CSV to detect element columns (assuming numeric columns are elements)
    data = pd.read_csv(csv_path)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude coordinate columns
    element_candidates = [col for col in numeric_columns if col not in ['EJE_X', 'EJE_Y']]

    if not element_candidates:
        raise ValueError("No numeric columns found in the CSV for element analysis")

    # Display available elements in a single comma-separated line
    print(f"\nElementos identificados: {','.join(element_candidates)}")

    # Get user input for elements
    elements_input = input("\nIngresar los elementos a analizar: ")
    elements = [e.strip().upper() for e in elements_input.split(',')]

    # Get window size
    window_size_m = float(input("Enter the window size in meters: "))

    # Calculate LREI and QLREI and save results
    lrei_result, qlrei_result = calculate_lrei_qlrei(csv_path, window_size_m, elements)