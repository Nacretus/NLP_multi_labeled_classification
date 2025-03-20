import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#########################################
# Step 1: Data Preparation and Exploration
#########################################

def load_and_prepare_data(filepath):
    """Load and prepare the dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print("\nDistribution of labels:")
    for col in ['toxic', 'very toxic', 'not toxic', 'obscenity/profanity', 'insults', 'threatening', 'identity-based negativity']:
        print(f"{col}: {df[col].sum()} positive samples ({df[col].sum() / len(df) * 100:.2f}%)")

    # Check for overlap between exclusive categories
    exclusive_overlap = df[['toxic', 'very toxic', 'not toxic']].sum(axis=1)
    print("\nOverlap in exclusive categories:")
    print(exclusive_overlap.value_counts())

    # Fix any overlaps to make categories truly exclusive
    # Priority: very toxic > toxic > not toxic
    df_fixed = df.copy()
    df_fixed.loc[df_fixed['very toxic'] == 1, 'toxic'] = 0  # If very toxic, not just toxic
    df_fixed.loc[(df_fixed['very toxic'] == 1) | (df_fixed['toxic'] == 1), 'not toxic'] = 0  # If toxic or very toxic, not "not toxic"

    # Create a toxicity level column for exclusive classification
    df_fixed['toxicity_level'] = 0  # Default: not toxic
    df_fixed.loc[df_fixed['toxic'] == 1, 'toxicity_level'] = 1  # Toxic
    df_fixed.loc[df_fixed['very toxic'] == 1, 'toxicity_level'] = 2  # Very toxic

    return df_fixed

def stratified_split(df, text_column='Comment'):
    """Split data into train, validation, and test sets with stratification"""
    X = df[text_column].values
    y_level = df['toxicity_level'].values
    y_categories = df[['obscenity/profanity', 'insults', 'threatening', 'identity-based negativity']].values

    # First split into train and temp (val+test)
    X_train, X_temp, y_level_train, y_level_temp, y_cat_train, y_cat_temp = train_test_split(
        X, y_level, y_categories, test_size=0.4, random_state=42, stratify=y_level
    )

    # Split temp into validation and test
    X_val, X_test, y_level_val, y_level_test, y_cat_val, y_cat_test = train_test_split(
        X_temp, y_level_temp, y_cat_temp, test_size=0.5, random_state=42, stratify=y_level_temp
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return (X_train, y_level_train, y_cat_train,
            X_val, y_level_val, y_cat_val,
            X_test, y_level_test, y_cat_test)

#########################################
# Step 2: Text Preprocessing
#########################################

def clean_text(text):
    """Clean and normalize text"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()
        return text
    return ""

class Vocab:
    """Simple vocabulary class for tokenization"""
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = {}

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count words
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)

        # Add top words to vocab
        for word, _ in sorted_words[:self.max_size-2]:  # -2 for PAD and UNK
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        return [self.word2idx.get(word, 1) for word in text.split()]  # 1 is <UNK>

    def save(self, filepath):
        """Save vocabulary to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def load_fasttext_embeddings(vocab, embedding_file, embedding_dim=300):
    """Load pre-trained FastText embeddings for vocabulary"""
    print(f"Loading FastText embeddings from {embedding_file}...")

    # Initialize embedding matrix
    embedding_matrix = np.zeros((len(vocab.word2idx), embedding_dim))

    # Load embeddings
    found_words = 0
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            values = line.strip().split(' ')
            word = values[0]

            if word in vocab.word2idx:
                vectors = np.asarray(values[1:], dtype='float32')
                embedding_matrix[vocab.word2idx[word]] = vectors
                found_words += 1

    print(f"Found embeddings for {found_words}/{len(vocab.word2idx)} words")
    return torch.FloatTensor(embedding_matrix)

def pad_sequences(sequences, max_len):
    """Pad sequences to the same length"""
    padded_seqs = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seqs.append(seq[:max_len])
        else:
            padded_seqs.append(seq + [0] * (max_len - len(seq)))
    return padded_seqs

#########################################
# Step 3: PyTorch Dataset and DataLoader
#########################################

class ToxicityDataset(Dataset):
    """Dataset for toxicity classification"""
    def __init__(self, texts, toxicity_levels, categories, vocab, max_len=200):
        self.texts = [clean_text(text) for text in texts]
        self.sequences = [vocab.text_to_sequence(text) for text in self.texts]
        self.padded_seqs = torch.LongTensor(pad_sequences(self.sequences, max_len))
        self.toxicity_levels = torch.LongTensor(toxicity_levels)
        self.categories = torch.FloatTensor(categories)

    def __len__(self):
        return len(self.padded_seqs)

    def __getitem__(self, idx):
        return {
            'text': self.padded_seqs[idx],
            'toxicity_level': self.toxicity_levels[idx],
            'categories': self.categories[idx]
        }

#########################################
# Step 4: Model Architecture
#########################################

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class ToxicityClassifier(nn.Module):
    """Hybrid TextCNN + BiLSTM model for toxicity classification"""
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, max_len,
                 filter_sizes=[3, 4, 5, 6, 7], num_filters=256):
        super(ToxicityClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize with pretrained embeddings
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                     kernel_size=fs) for fs in filter_sizes
        ])

        # BiLSTM layers
        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)

        # Dense layers with increased depth (added 2 more as requested)
        self.fc1 = nn.Linear(num_filters * len(filter_sizes) + 2 * 256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)

        # Output layers
        self.toxicity_out = nn.Linear(64, 3)  # 3 classes: not toxic, toxic, very toxic
        self.category_out = nn.Linear(64, 4)  # 4 binary categories

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, max_len)

        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, max_len, embedding_dim)

        # CNN layers
        embedded_cnn = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded_cnn))  # (batch_size, num_filters, max_len - filter_size + 1)
            pool_out = F.max_pool1d(conv_out, conv_out.shape[2])  # (batch_size, num_filters, 1)
            conv_outputs.append(pool_out.squeeze(2))  # (batch_size, num_filters)

        cnn_features = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))

        # BiLSTM layers
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch_size, max_len, 2*hidden_size)

        # Get the last hidden state from both directions
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 2*hidden_size)

        # Combine CNN and LSTM features
        combined = torch.cat([cnn_features, lstm_features], dim=1)

        # Dense layers
        x = F.relu(self.fc1(self.dropout(combined)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = F.relu(self.fc4(self.dropout(x)))

        # Output layers
        toxicity_level = self.toxicity_out(x)  # (batch_size, 3)
        categories = torch.sigmoid(self.category_out(x))  # (batch_size, 4)

        return toxicity_level, categories

#########################################
# Step 5: Training Loop
#########################################

def train_model(model, train_loader, val_loader, focal_loss, bce_loss,
                optimizer, scheduler, num_epochs=20, save_dir='models'):
    """Train the model"""
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_toxicity_acc': [], 'val_toxicity_acc': [],
        'train_category_acc': [], 'val_category_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_toxicity_correct = 0
        train_category_correct = 0
        train_samples = 0

        for batch in tqdm(train_loader, desc="Training"):
            texts = batch['text'].to(device)
            toxicity_levels = batch['toxicity_level'].to(device)
            categories = batch['categories'].to(device)

            # Forward pass
            toxicity_preds, category_preds = model(texts)

            # Calculate losses
            toxicity_loss = focal_loss(toxicity_preds, toxicity_levels)
            category_loss = bce_loss(category_preds, categories)

            # Combined loss (you can adjust the weights)
            loss = toxicity_loss + category_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * texts.size(0)

            _, predicted_toxicity = torch.max(toxicity_preds, 1)
            train_toxicity_correct += (predicted_toxicity == toxicity_levels).sum().item()

            predicted_categories = (category_preds > 0.5).float()
            train_category_correct += (predicted_categories == categories).sum().item()
            train_samples += texts.size(0)

        train_loss /= train_samples
        train_toxicity_acc = train_toxicity_correct / train_samples
        train_category_acc = train_category_correct / (train_samples * 4)  # 4 categories

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_toxicity_correct = 0
        val_category_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                texts = batch['text'].to(device)
                toxicity_levels = batch['toxicity_level'].to(device)
                categories = batch['categories'].to(device)

                # Forward pass
                toxicity_preds, category_preds = model(texts)

                # Calculate losses
                toxicity_loss = focal_loss(toxicity_preds, toxicity_levels)
                category_loss = bce_loss(category_preds, categories)

                # Combined loss
                loss = toxicity_loss + category_loss

                # Update metrics
                val_loss += loss.item() * texts.size(0)

                _, predicted_toxicity = torch.max(toxicity_preds, 1)
                val_toxicity_correct += (predicted_toxicity == toxicity_levels).sum().item()

                predicted_categories = (category_preds > 0.5).float()
                val_category_correct += (predicted_categories == categories).sum().item()
                val_samples += texts.size(0)

        val_loss /= val_samples
        val_toxicity_acc = val_toxicity_correct / val_samples
        val_category_acc = val_category_correct / (val_samples * 4)  # 4 categories

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Toxicity Acc: {train_toxicity_acc:.4f} | "
              f"Train Category Acc: {train_category_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Toxicity Acc: {val_toxicity_acc:.4f} | "
              f"Val Category Acc: {val_category_acc:.4f}")

        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_toxicity_acc'].append(train_toxicity_acc)
        history['val_toxicity_acc'].append(val_toxicity_acc)
        history['train_category_acc'].append(train_category_acc)
        history['val_category_acc'].append(val_category_acc)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print("Saved best model checkpoint!")

    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'final_model.pt'))

    return history

#########################################
# Step 6: Evaluation
#########################################

def evaluate_model(model, test_loader, focal_loss, bce_loss):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    test_toxicity_correct = 0
    test_samples = 0

    # For AUC-ROC and confusion matrix
    all_toxicity_labels = []
    all_toxicity_preds = []
    all_category_labels = []
    all_category_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            texts = batch['text'].to(device)
            toxicity_levels = batch['toxicity_level'].to(device)
            categories = batch['categories'].to(device)

            # Forward pass
            toxicity_preds, category_preds = model(texts)

            # Calculate losses
            toxicity_loss = focal_loss(toxicity_preds, toxicity_levels)
            category_loss = bce_loss(category_preds, categories)

            # Combined loss
            loss = toxicity_loss + category_loss

            # Update metrics
            test_loss += loss.item() * texts.size(0)

            _, predicted_toxicity = torch.max(toxicity_preds, 1)
            test_toxicity_correct += (predicted_toxicity == toxicity_levels).sum().item()
            test_samples += texts.size(0)

            # Store predictions and labels for metrics
            all_toxicity_labels.extend(toxicity_levels.cpu().numpy())
            all_toxicity_preds.extend(F.softmax(toxicity_preds, dim=1).cpu().numpy())
            all_category_labels.extend(categories.cpu().numpy())
            all_category_preds.extend(category_preds.cpu().numpy())

    test_loss /= test_samples
    test_toxicity_acc = test_toxicity_correct / test_samples

    print(f"Test Loss: {test_loss:.4f} | Test Toxicity Acc: {test_toxicity_acc:.4f}")

    # Convert to numpy arrays
    all_toxicity_labels = np.array(all_toxicity_labels)
    all_toxicity_preds = np.array(all_toxicity_preds)
    all_category_labels = np.array(all_category_labels)
    all_category_preds = np.array(all_category_preds)

    # Calculate AUC-ROC for toxicity level (one-vs-rest approach)
    auc_toxicity = roc_auc_score(
        np.eye(3)[all_toxicity_labels],  # Convert to one-hot
        all_toxicity_preds,
        multi_class='ovr'
    )
    print(f"AUC-ROC for toxicity level (multi-class): {auc_toxicity:.4f}")

    # Calculate AUC-ROC for each category
    category_names = ['obscenity/profanity', 'insults', 'threatening', 'identity-based negativity']
    for i, category in enumerate(category_names):
        auc = roc_auc_score(all_category_labels[:, i], all_category_preds[:, i])
        print(f"AUC-ROC for {category}: {auc:.4f}")

    # Confusion matrix for toxicity level
    all_toxicity_pred_classes = np.argmax(all_toxicity_preds, axis=1)
    cm = confusion_matrix(all_toxicity_labels, all_toxicity_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Toxic', 'Toxic', 'Very Toxic'],
                yticklabels=['Not Toxic', 'Toxic', 'Very Toxic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Toxicity Level')
    plt.savefig('toxicity_level_confusion_matrix.png')
    plt.close()

    # Classification report for toxicity level
    print("\nClassification Report for Toxicity Level:")
    print(classification_report(all_toxicity_labels, all_toxicity_pred_classes,
                                target_names=['Not Toxic', 'Toxic', 'Very Toxic']))

    # Classification report for categories
    all_category_pred_classes = (all_category_preds > 0.5).astype(int)
    print("\nClassification Report for Categories:")
    print(classification_report(all_category_labels, all_category_pred_classes,
                                target_names=category_names))

    return {
        'test_loss': test_loss,
        'test_toxicity_acc': test_toxicity_acc,
        'toxicity_auc': auc_toxicity,
        'all_toxicity_labels': all_toxicity_labels,
        'all_toxicity_preds': all_toxicity_preds,
        'all_category_labels': all_category_labels,
        'all_category_preds': all_category_preds
    }

#########################################
# Step 7: Main Execution
#########################################

def main():
    # Configuration
    DATA_FILE = 'combine please work .csv'
    FASTTEXT_FILE = 'fasttext_vectors_dim_300_new_.txt'
    MAX_LEN = 200
    MAX_VOCAB_SIZE = 50000
    EMBEDDING_DIM = 300
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Load and prepare data
    df = load_and_prepare_data(DATA_FILE)

    # Split data
    X_train, y_level_train, y_cat_train, X_val, y_level_val, y_cat_val, X_test, y_level_test, y_cat_test = stratified_split(df)

    # Clean texts
    X_train_clean = [clean_text(x) for x in X_train]
    X_val_clean = [clean_text(x) for x in X_val]
    X_test_clean = [clean_text(x) for x in X_test]

    # Build vocabulary
    vocab = Vocab(max_size=MAX_VOCAB_SIZE)
    vocab.build_vocab(X_train_clean)

    # Save vocabulary
    vocab.save('vocab.pkl')

    # Load FastText embeddings
    embedding_matrix = load_fasttext_embeddings(vocab, FASTTEXT_FILE, EMBEDDING_DIM)

    # Create datasets
    train_dataset = ToxicityDataset(X_train, y_level_train, y_cat_train, vocab, MAX_LEN)
    val_dataset = ToxicityDataset(X_val, y_level_val, y_cat_val, vocab, MAX_LEN)
    test_dataset = ToxicityDataset(X_test, y_level_test, y_cat_test, vocab, MAX_LEN)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = ToxicityClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        max_len=MAX_LEN,
        filter_sizes=[3, 4, 5, 6, 7],  # Increased filter sizes as requested
        num_filters=256
    ).to(device)

    # Define loss functions
    focal_loss = FocalLoss(gamma=2, alpha=0.25)
    bce_loss = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6, verbose=True
    )

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        focal_loss=focal_loss,
        bce_loss=bce_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS
    )

    # Load best model
    checkpoint = torch.load('models/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate model
    results = evaluate_model(model, test_loader, focal_loss, bce_loss)

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_toxicity_acc'], label='Train Toxicity Accuracy')
    plt.plot(history['val_toxicity_acc'], label='Validation Toxicity Accuracy')
    plt.title('Toxicity Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main()