import argparse
import pickle
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell2pin_pkl", type=str, default="cell2pin.pkl", help="input file")
    parser.add_argument("--cell_power_pkl", type=str, default="cell_power.pkl", help="input file")
    parser.add_argument("--cell_feature_npy", type=str, default="cell_feature.npy", help="output cell feature")
    args = parser.parse_args()

    # Load cell2pin.pkl
    with open(args.cell2pin_pkl, 'rb') as file:
        data = pickle.load(file)

    # Get a list of cell type
    cell_types = sorted(set(value[0] for value in data.values()))

    # Load cell_power.pkl
    with open(args.cell_power_pkl, 'rb') as file:
        cell_power = pickle.load(file)

    # Build a feature list
    feature_list = []
    max_length = 0
    for cell_type in cell_types:
        features = cell_power.get(cell_type, [])
        max_length = max(max_length, len(features))
        feature_list.append(features)

    feature_array = np.array([
        features + [0] * (max_length - len(features))
        for features in feature_list
    ])

    # Save the cell feature
    np.save(args.cell_feature_npy, feature_array)
    print("Array saved with shape:", feature_array.shape)

if __name__ == "__main__":
    main()
