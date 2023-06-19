import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
block_size = 32
batch_size = 300
max_iters = 20000
eval_iters = 1000
learning_rate = 3e-4
eval_interval = 1000
n_embd = 54
n_heads = 2
n_layers = 2
dropout = 0.2
vocab_size = 94

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        '''One head of self-attention'''
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_bugger('tril', torch.tril(torch.ones(block_size, block_size)))

        '''Multi-head Attention'''
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def foward(self, x):
        '''one haed'''
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute the attn scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.sofmax(wei, dim=-1)
        # perform the weighted sum of the values
        v = self.value(x)
        out = lambda v: self.dropout(wei @ v)
        
        '''Mulitple heads'''
        heads = torch.cat([out(v=v) for _ in range(n_heads)], dim=-1)
        output = self.dropout(self.proj(heads))
        return output
    
class FeedForward(nn.Module):
    '''A simpler linear layer followed by a non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.self_attn = MultiHeadAttention(n_heads=n_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))  # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_ebedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        '''^ Draw the neural network above to understand it, when you read the code and the paper at the same time'''
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_tok_emb = self.position_ebedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_tok_emb
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = GPTModel()
model = model.to(device)
model.load_state_dict(torch.load('model_weights.pth'))

# torch.save(model.save_dict(), os.path.join('model_weights.pth'))
