import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from collections import Counter

# ============================================================================
# 1. TOKENIZER (convert text to numbers)
# ============================================================================
class SimpleTokenizer:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
    
    def build_vocab(self, text, min_freq=1):
        """Build vocabulary from text"""
        words = text.split()
        word_freq = Counter(words)
        
        # Add special tokens
        self.token2id['<PAD>'] = 0
        self.token2id['<EOS>'] = 1
        idx = 2
        
        # Add words by frequency
        for word, freq in word_freq.most_common():
            if freq >= min_freq:
                self.token2id[word] = idx
                idx += 1
        
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        print(f"Vocab size: {self.vocab_size}")
    
    def encode(self, text):
        """Convert text to token IDs"""
        words = text.split()
        tokens = [self.token2id.get(w, self.token2id['<PAD>']) for w in words]
        return tokens + [self.token2id['<EOS>']]
    
    def decode(self, tokens):
        """Convert token IDs back to text"""
        return ' '.join([self.id2token.get(t, '<UNK>') for t in tokens if t != self.token2id['<EOS>']])


# ============================================================================
# 2. ATTENTION HEAD (single attention mechanism)
# ============================================================================
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        
        # Mask to prevent attending to future tokens
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)) == 1)
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = Q @ K.transpose(-2, -1) * self.scale  # (B, T, T)
        
        # Mask future tokens (causal attention)
        scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
        
        # Softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = weights @ V
        return output


# ============================================================================
# 3. MULTI-HEAD ATTENTION (multiple attention heads in parallel)
# ============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, self.head_dim, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Run all heads in parallel
        head_outputs = [head(x) for head in self.heads]
        # Concatenate head outputs
        x = torch.cat(head_outputs, dim=-1)
        # Project back
        x = self.proj(x)
        x = self.dropout(x)
        return x


# ============================================================================
# 4. FEEDFORWARD NETWORK
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# 5. TRANSFORMER BLOCK (attention + feedforward)
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, block_size)
        self.feedforward = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention with residual connection and layer norm
        x = x + self.attention(self.ln1(x))
        # Feedforward with residual connection and layer norm
        x = x + self.feedforward(self.ln2(x))
        return x


# ============================================================================
# 6. FULL TRANSFORMER MODEL
# ============================================================================
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        
        # Embedding
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Output layer
        x = self.ln_final(x)
        logits = self.head(x)
        
        return logits
    
    def generate(self, prompt_tokens, max_len=50, temperature=0.1):
        """Generate text from prompt"""
        context = prompt_tokens[:self.block_size]
        
        for _ in range(max_len):
            # Get predictions
            logits = self(torch.tensor([context], device=next(self.parameters()).device))
            logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            context.append(next_token)
            if len(context) > self.block_size:
                context = context[-self.block_size:]
        
        return context


# ============================================================================
# 7. DATASET
# ============================================================================
class TextDataset:
    def __init__(self, text, tokenizer, block_size=32):
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x), torch.tensor(y)


# ============================================================================
# 8. TRAINING
# ============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load training data from file
    try:
        with open('solar-system.txt', 'r', encoding='utf-8') as f:
            training_text = f.read()
    except FileNotFoundError:
        print("Error: solar-system.txt not found in current directory")
        print("Make sure the file is in the same folder as this script")
        return
    
    # Setup
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_text)
    
    dataset = TextDataset(training_text, tokenizer, block_size=32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        block_size=32
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Training...")
    for epoch in range(20):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'tiny_llm_weights.pt')
    torch.save(tokenizer.token2id, 'tokenizer_vocab.pt')
    print("\n✓ Model saved: tiny_llm_weights.pt")
    print("✓ Tokenizer saved: tokenizer_vocab.pt")

    # Generate text
    print("\n" + "="*50)
    print("GENERATION")
    print("="*50)
    prompt = "What is sun? a star or planet?"
    prompt_tokens = tokenizer.encode(prompt)[:-1]  # Remove <EOS>
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt_tokens, max_len=30, temperature=0.7)
    
    generated_text = tokenizer.decode(generated)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


def load_and_generate(prompt, max_len=50):
    """Load trained model and generate from prompt"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    token2id = torch.load('v2/tokenizer_vocab.pt')
    tokenizer = SimpleTokenizer()
    tokenizer.token2id = token2id
    tokenizer.id2token = {v: k for k, v in token2id.items()}
    tokenizer.vocab_size = len(token2id)

    # Load model
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        block_size=64
    ).to(device)
    model.load_state_dict(torch.load('v2/tiny_llm_weights.pt', map_location=device))
    model.eval()

    # Generate
    prompt_tokens = tokenizer.encode(prompt)[:-1]
    with torch.no_grad():
        generated = model.generate(prompt_tokens, max_len=max_len, temperature=0.7)

    return tokenizer.decode(generated)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # Usage: python tiny_llm.py generate "your prompt here"
        prompt = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else "the quick brown"
        result = load_and_generate(prompt, max_len=100)
        print(f"Generated: {result}")
    else:
        train()