import pandas as pd
import torch
from torch_geometric.data import Data


def prepare_ppi_data(protein_links_path, protein_info_path, confidence_threshold=700):
    print("Loading protein links...")
    links_columns = [
        "protein1", "protein2", "neighborhood", "fusion", "cooccurrence",
        "coexpression", "experimental", "database", "textmining", "combined_score"
    ]
    protein_links = pd.read_csv(protein_links_path, sep=" ", names=links_columns, low_memory=False)

    # Convert combined_score to numeric and drop invalid rows
    protein_links["combined_score"] = pd.to_numeric(protein_links["combined_score"], errors="coerce")
    protein_links = protein_links.dropna(subset=["combined_score"])
    protein_links["combined_score"] = protein_links["combined_score"].astype(int)

    # Filter high-confidence interactions
    protein_links = protein_links[protein_links["combined_score"] > confidence_threshold]

    print("Loading protein info...")
    info_columns = ["string_protein_id", "preferred_name", "protein_size", "annotation"]
    protein_info = pd.read_csv(protein_info_path, sep="\t", names=info_columns, skiprows=1)

    print("Mapping proteins to indices...")
    proteins = pd.concat([protein_links["protein1"], protein_links["protein2"]]).unique()
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}

    # Map STRING Protein IDs to indices and ensure integer indexing
    protein_info["string_protein_id"] = protein_info["string_protein_id"].map(protein_to_idx)
    protein_info = protein_info.dropna(subset=["string_protein_id"])
    protein_info["string_protein_id"] = protein_info["string_protein_id"].astype(int)  # Ensure integer type
    protein_info = protein_info.set_index("string_protein_id")

    print("Creating edge index...")
    edge_index = torch.tensor(
        [[protein_to_idx[p1], protein_to_idx[p2]] for p1, p2 in
         zip(protein_links["protein1"], protein_links["protein2"])],
        dtype=torch.long,
    ).t().contiguous()

    print("Generating node features...")
    num_proteins = len(proteins)
    x = torch.zeros((num_proteins, 1))  # Single feature: protein size
    for idx, row in protein_info.iterrows():
        x[int(idx)] = row["protein_size"] / protein_info["protein_size"].max()  # Normalize protein size

    print("Generating labels and train/test masks...")
    y = torch.randint(0, 2, (num_proteins,))  # Random binary labels
    train_mask = torch.zeros(num_proteins, dtype=torch.bool)
    test_mask = torch.zeros(num_proteins, dtype=torch.bool)
    train_mask[:int(0.6 * num_proteins)] = True
    test_mask[int(0.6 * num_proteins):] = True

    print("Creating PyTorch Geometric Data object...")
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    return data


# Example Usage
if __name__ == "__main__":
    # Paths to the extracted STRING database files
    protein_links_path = "10090.protein.links.detailed.v12.0.txt"
    protein_info_path = "10090.protein.info.v12.0.txt"

    # Prepare data
    #data = prepare_ppi_data(protein_links_path, protein_info_path)

    # Save the processed data for future use
    #torch.save(data, "processed_ppi_data.pt")
    #print("Data processing complete. Saved to 'processed_ppi_data.pt'.")
    data = torch.load("processed_ppi_data.pt")
    print(f"x shape: {data.x.shape}")  # 应输出 torch.Size([15939, 1])
    print(f"edge_index shape: {data.edge_index.shape}")  # 应输出 torch.Size([2, 401966])
    print(f"y shape: {data.y.shape}")  # 应输出 torch.Size([15939])
    print(
        f"train_mask shape: {data.train_mask.shape}, sum: {data.train_mask.sum().item()}")  # 应输出 torch.Size([15939]), sum 是训练集大小
    print(
        f"test_mask shape: {data.test_mask.shape}, sum: {data.test_mask.sum().item()}")  # 应输出 torch.Size([15939]), sum 是测试集大小

    # 检查掩码不重叠
    assert (data.train_mask & data.test_mask).sum() == 0, "Train and test masks overlap!"

    print(data)