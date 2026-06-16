
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import pandas as pd

from transformer_blocks import Block

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

word_count = len(text.split())
print("Jumlah kata corpus:", word_count)

block_size = 16
embedding_dim = 64
n_heads = 4
n_layers = 2
lr = 2e-3
epochs = 700
batch_size = 32
max_new_tokens = 80

class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

def train_experiment(name, model_type, vocab_size):
    print("\n" + "="*70)
    print(f"EKSPERIMEN TOKENISASI: {name.upper()}")
    print("="*70)

    prefix = f"tokenizer_{name}"

    spm.SentencePieceTrainer.Train(
        input="corpus.txt",
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        hard_vocab_limit=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")

    ids = sp.encode(text, out_type=int)
    data = torch.tensor(ids, dtype=torch.long)

    actual_vocab = sp.get_piece_size()
    total_tokens = len(ids)

    print("Vocab size:", actual_vocab)
    print("Total token:", total_tokens)

    model = TinyGPT(actual_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_history = []

    for step in range(epochs + 1):
        xb, yb = get_batch(data)
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            loss_value = loss.item()
            loss_history.append((step, loss_value))
            print(f"Step {step}, loss={loss_value:.4f}")

    prompts = [
        "Pariwisata Indonesia",
        "Wisatawan dapat",
        "Desa wisata"
    ]

    generated_outputs = []
    for prompt in prompts:
        context = torch.tensor([sp.encode(prompt)], dtype=torch.long).to(device)
        out = model.generate(context, max_new_tokens=max_new_tokens)
        generated_text = sp.decode(out[0].tolist())
        generated_outputs.append((prompt, generated_text))

        print("\nPrompt:", prompt)
        print("Generated text:")
        print(generated_text)

    final_loss = loss_history[-1][1]

    return {
        "tokenisasi": name,
        "model_type": model_type,
        "vocab_size": actual_vocab,
        "total_tokens": total_tokens,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "generated_outputs": generated_outputs
    }

experiments = [
    ("bpe", "bpe", 80),
    ("unigram", "unigram", 80),
    ("char", "char", 80),
    ("word", "word", 300)
]

results = []

for name, model_type, vocab_size in experiments:
    result = train_experiment(name, model_type, vocab_size)
    results.append(result)

summary = pd.DataFrame([
    {
        "Tokenisasi": r["tokenisasi"],
        "Model Type": r["model_type"],
        "Vocab Size": r["vocab_size"],
        "Total Token": r["total_tokens"],
        "Final Loss": round(r["final_loss"], 4)
    }
    for r in results
])

print("\n\nRINGKASAN HASIL EKSPERIMEN")
print(summary)

summary.to_csv("ringkasan_hasil.csv", index=False)

with open("hasil_eksperimen.txt", "w", encoding="utf-8") as f:
    f.write("HASIL EKSPERIMEN TINY GPT - CORPUS PARIWISATA INDONESIA\n")
    f.write("Link corpus: https://github.com/alifionm/tiny-gpt-pariwisata/blob/main/corpus.txt\n\n")
    f.write(f"Jumlah kata corpus: {word_count}\n\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n")

    for r in results:
        f.write("="*80 + "\n")
        f.write(f"TOKENISASI: {r['tokenisasi'].upper()}\n")
        f.write(f"Model type: {r['model_type']}\n")
        f.write(f"Vocab size: {r['vocab_size']}\n")
        f.write(f"Total token: {r['total_tokens']}\n")
        f.write(f"Final loss: {r['final_loss']:.4f}\n\n")
        f.write("Loss history:\n")
        for step, loss_value in r["loss_history"]:
            f.write(f"Step {step}, loss={loss_value:.4f}\n")

        f.write("\nGenerated outputs:\n")
        for prompt, output in r["generated_outputs"]:
            f.write(f"\nPrompt: {prompt}\n")
            f.write(output + "\n")

print("\nFile hasil sudah dibuat:")
print("- hasil_eksperimen.txt")
print("- ringkasan_hasil.csv")
