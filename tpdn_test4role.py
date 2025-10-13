# Tensor Product Decomposition Network for Number Sequences

import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


def set_seed(seed):
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


class Instance(object):
    def __init__(self, filler, role, rep):
        self.filler = filler
        self.role = role
        self.rep = rep


def get_roles(role_scheme, seq_length):
    """
    Generate role assignments based on the specified scheme.

    Args:
        role_scheme: One of 'l2r', 'r2l', 'bow', 'bidi'
        seq_length: Length of the sequence

    Returns:
        List of role identifiers
    """
    if role_scheme == 'l2r':
        # Left-to-right positional encoding
        return [i for i in range(seq_length)]

    elif role_scheme == 'r2l':
        # Right-to-left positional encoding
        return [seq_length - i - 1 for i in range(seq_length)]

    elif role_scheme == 'bow':
        # Bag of words - same role for all positions
        return [0] * seq_length

    elif role_scheme == 'bidi':
        # Bidirectional - combine l2r and r2l
        return ['%d-%d' % (i, seq_length - i - 1) for i in range(seq_length)]

    else:
        raise ValueError(f"Unknown role scheme: {role_scheme}")


# Global role vocabulary
id2role = {0: '[UNK]'}
role2id = {'[UNK]': 0}


def load_number_sequences(filename, device, role_scheme='l2r', normalize=True):
    """
    Load number sequences from text file and create training instances.
    Each sequence is treated as a list of fillers with specified role scheme.

    Args:
        filename: Path to the text file with sequences
        device: PyTorch device (CPU/GPU)
        role_scheme: One of 'l2r', 'r2l', 'bow', 'bidi'
        normalize: Whether to normalize the sequences
    """
    print(f"Loading sequences from {filename}...")
    print(f"Role scheme: {role_scheme}")
    sequences = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                nums = [float(x) for x in line.strip().split()]
                sequences.append(nums)
    except FileNotFoundError:
        print(f"ERROR: File {filename} not found.")
        sys.exit(1)

    if not sequences:
        print("ERROR: No sequences loaded.")
        sys.exit(1)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Sample sequences: {sequences[:3]}")

    # Normalize sequences if requested
    if normalize:
        sequences = np.array(sequences)
        mean = sequences.mean()
        std = sequences.std()
        sequences = (sequences - mean) / std
        print(f"Normalized: mean={mean:.2f}, std={std:.2f}")
    else:
        sequences = np.array(sequences)

    # Create instances
    data = []
    seq_length = len(sequences[0])

    for seq in sequences:
        # Fillers: the actual numbers (as discrete tokens 0-100)
        filler_ids = [int(min(max(round(x), 0), 100)) if not normalize else int(min(max(round(x * std + mean), 0), 100))
                      for x in seq]

        # Roles: based on specified scheme
        roles = get_roles(role_scheme, seq_length)

        # Convert roles to IDs (build vocabulary on the fly)
        role_ids = []
        for role in roles:
            role_str = str(role)
            if role_str not in role2id:
                role2id[role_str] = len(id2role)
                id2role[role2id[role_str]] = role_str
            role_ids.append(role2id[role_str])

        # Representation: the actual sequence values as a vector
        representation = seq.tolist()

        # Convert to tensors
        filler_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)
        role_t = Variable(torch.LongTensor(role_ids), requires_grad=False).unsqueeze(0)
        rep_t = Variable(torch.FloatTensor(representation), requires_grad=False).unsqueeze(0)

        # Move to device
        filler_t = filler_t.to(device)
        role_t = role_t.to(device)
        rep_t = rep_t.to(device)

        data.append(Instance(filler_t, role_t, rep_t))

    print(f"Role vocabulary size: {len(id2role)}")
    print(f"Sample roles for first sequence: {roles}")

    return data


class SumFlattenedOuterProduct(nn.Module):
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()

    def forward(self, input1, input2):
        # Compute outer product and flatten
        outer_product = torch.bmm(input1.transpose(1, 2), input2)
        flattened_outer_product = outer_product.view(outer_product.size()[0], -1).unsqueeze(0)
        return flattened_outer_product


class TensorProductEncoder(nn.Module):
    def __init__(self, num_roles, num_fillers, role_dim, filler_dim, final_layer_width):
        super(TensorProductEncoder, self).__init__()
        self.role_embed = nn.Embedding(num_roles, role_dim)
        self.filler_embed = nn.Embedding(num_fillers, filler_dim)
        self.sum_layer = SumFlattenedOuterProduct()
        self.last_layer = nn.Linear(filler_dim * role_dim, final_layer_width)





    def forward(self, fillers, roles):

        # Forward pass with activation capture
        filler_embed = self.filler_embed(fillers)
        role_embed = self.role_embed(roles)



        # Tensor product
        tensor_product = self.sum_layer(filler_embed, role_embed)

        # Final projection
        output = self.last_layer(tensor_product)

        return output


def batchify(data, batch_size):
    """Group data into batches."""
    batches = []
    for bi in range(0, len(data), batch_size):
        batch_data = data[bi:min(bi + batch_size, len(data))]

        first_item = batch_data[0]
        cur_filler = first_item.filler
        cur_role = first_item.role
        cur_rep = first_item.rep

        for inst in batch_data[1:]:
            cur_filler = torch.cat((cur_filler, inst.filler), 0)
            cur_role = torch.cat((cur_role, inst.role), 0)
            cur_rep = torch.cat((cur_rep, inst.rep), 0)

        batches.append((cur_filler, cur_role, cur_rep.unsqueeze(0)))

    return batches


def train_model(model, train_batches, optimizer, criterion, batch_size):
    """Train the model for one epoch."""
    model.train()
    random.shuffle(train_batches)
    epoch_losses = []

    for bi in tqdm(range(0, len(train_batches), batch_size), desc="Training"):
        batch = train_batches[bi:min(bi + batch_size, len(train_batches))]
        if not batch:
            continue

        optimizer.zero_grad()
        cur_loss = 0

        for filler_in, role_in, target_out in batch:
            enc_out = model(filler_in, role_in)
            cur_loss += criterion(enc_out, target_out)

        cur_loss.backward()
        optimizer.step()
        epoch_losses.append(cur_loss.item())

    return sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')


def evaluate(model, data):
    """Evaluate model on a dataset."""
    model.eval()
    total_mse = 0

    with torch.no_grad():
        for instance in data:
            pred = model(instance.filler, instance.role)
            total_mse += torch.mean(torch.pow(pred - instance.rep, 2))

    return float(total_mse / len(data))


def run_tpdn(train_data, valid_data, test_data, device, role_scheme='l2r'):
    """Train and evaluate the TPDN model."""
    if not train_data:
        print("ERROR: No training data loaded.")
        return None

    # Model parameters
    num_fillers = 101  # 0-100
    num_roles = len(id2role)  # Use the role vocabulary size
    role_dim = 8
    filler_dim = 16
    rep_size = train_data[0].rep.size(1)

    print(f"\nModel Configuration:")
    print(f"  Role scheme: {role_scheme}")
    print(f"  Number of fillers: {num_fillers}")
    print(f"  Number of roles: {num_roles}")
    print(f"  Role dimension: {role_dim}")
    print(f"  Filler dimension: {filler_dim}")
    print(f"  Representation size: {rep_size}")

    # Create model
    model = TensorProductEncoder(num_roles, num_fillers, role_dim, filler_dim, rep_size)
    model.to(device)

    # Training configuration
    batch_size = 32
    learning_rate = 0.001
    n_epochs = 50
    patience = 10
    best_loss = float('inf')
    patience_counter = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Prepare batches
    train_batches = batchify(train_data, batch_size)

    # Track losses
    train_losses = []
    valid_losses = []

    # Initial evaluation
    print("\nInitial evaluation:")
    initial_valid = evaluate(model, valid_data)
    initial_test = evaluate(model, test_data)
    print(f"  Validation MSE: {initial_valid:.6f}")
    print(f"  Test MSE: {initial_test:.6f}")

    # Training loop
    print(f"\nTraining for up to {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # Train
        avg_train_loss = train_model(model, train_batches, optimizer, criterion, batch_size)
        train_losses.append(avg_train_loss)

        # Validate
        valid_loss = evaluate(model, valid_data)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}: Train={avg_train_loss:.6f}, Valid={valid_loss:.6f}")

        # Early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_tpdn_model.pt")
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("best_tpdn_model.pt"))

    # Final evaluation
    final_valid = evaluate(model, valid_data)
    final_test = evaluate(model, test_data)

    print(f"\nFinal Results:")
    print(f"  Validation MSE: {final_valid:.6f}")
    print(f"  Test MSE: {final_test:.6f}")

    # Plot losses
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title(f'TPDN Training Progress ({role_scheme.upper()} Role Scheme)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tpdn_number_sequences_{role_scheme}_loss.png', dpi=150)
    print(f"\nLoss plot saved as 'tpdn_number_sequences_{role_scheme}_loss.png'")
    plt.close()

    # Save model outputs with intermediate activations
    print("\n" + "=" * 60)
    print("SAVING MODEL OUTPUTS")
    print("=" * 60)

    # Save training data outputs (first 100 instances)
    save_model_outputs(model, train_data,
                       f'model_outputs_train_{role_scheme}.json',
                       role_scheme, id2role, max_instances=100)

    # Save validation data outputs (all instances)
    save_model_outputs(model, valid_data,
                       f'model_outputs_valid_{role_scheme}.json',
                       role_scheme, id2role, max_instances=None)

    # Save test data outputs (all instances)
    save_model_outputs(model, test_data,
                       f'model_outputs_test_{role_scheme}.json',
                       role_scheme, id2role, max_instances=None)

    return model


def save_model_outputs(model, data, filename, role_scheme, id2role, max_instances=None):
    """
    Save model outputs and intermediate activations to JSON file.

    Args:
        model: Trained TensorProductEncoder model
        data: List of Instance objects
        filename: Output JSON filename
        role_scheme: Role scheme used
        id2role: Dictionary mapping role IDs to role names
        max_instances: Maximum number of instances to save (None = all)
    """
    print(f"\nGenerating model outputs for {filename}...")
    model.eval()
    outputs = []

    num_to_save = len(data) if max_instances is None else min(max_instances, len(data))

    with torch.no_grad():
        for i in tqdm(range(num_to_save), desc="Saving outputs"):
            instance = data[i]

            # Forward pass
            model_output = model(instance.filler, instance.role).squeeze(0)
            target = instance.rep.squeeze(0)


            # Extract data
            filler_ids = instance.filler.squeeze(0).cpu().numpy().tolist()
            role_ids = instance.role.squeeze(0).cpu().numpy().tolist()
            role_names = [id2role.get(rid, f'[UNK-{rid}]') for rid in role_ids]

            # Calculate loss for this instance
            instance_loss = torch.mean(torch.pow(model_output - target, 2)).item()



            # Create entry
            entry = {
                'instance_id': i,
                'filler_ids': filler_ids,
                'filler_values': filler_ids,  # For numbers, ID = value
                'role_ids': role_ids,
                'role_names': role_names,
                'role_scheme': role_scheme,
                'target_output': target.cpu().numpy().tolist(),
                'model_output': model_output.cpu().numpy().tolist(),
                'instance_mse': float(instance_loss),
            }
            outputs.append(entry)

    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved {num_to_save} model outputs to {filename}")

    # Print summary statistics
    mse_values = [entry['instance_mse'] for entry in outputs]
    print(f"  Average MSE: {np.mean(mse_values):.6f}")
    print(f"  Std MSE: {np.std(mse_values):.6f}")
    print(f"  Min MSE: {np.min(mse_values):.6f}")
    print(f"  Max MSE: {np.max(mse_values):.6f}")


def main():
    # Set random seed
    device = set_seed(42)

    # Choose role scheme: 'l2r', 'r2l', 'bow', or 'bidi'
    role_scheme = 'l2r'  # Change this to test different role schemes

    # Load data
    print("=" * 60)
    all_data = load_number_sequences('random_sequences.txt', device,
                                     role_scheme=role_scheme, normalize=True)

    # Split into train/valid/test
    n_total = len(all_data)
    n_train = int(0.7 * n_total)
    n_valid = int(0.15 * n_total)

    train_data = all_data[:n_train]
    valid_data = all_data[n_train:n_train + n_valid]
    test_data = all_data[n_train + n_valid:]

    print(f"\nData split:")
    print(f"  Training: {len(train_data)} sequences")
    print(f"  Validation: {len(valid_data)} sequences")
    print(f"  Test: {len(test_data)} sequences")
    print("=" * 60)

    # Train model
    model = run_tpdn(train_data, valid_data, test_data, device, role_scheme)

    if model is not None:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model trained with '{role_scheme}' role scheme")
        print("=" * 60)


if __name__ == "__main__":
    main()
