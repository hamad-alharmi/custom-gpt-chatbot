import os, json, torch, tiktoken, numpy as np
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig

BATCH_SIZE    = 16
BLOCK_SIZE    = 256
MAX_ITERS     = 5000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT    = "checkpoint.pt"
enc           = tiktoken.get_encoding("gpt2")

class ConversationDataset(Dataset):
    def __init__(self, path, block_size):
        self.block_size = block_size
        tokens = []
        if path.endswith(".jsonl") and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    for turn in obj.get("conversation", []):
                        tokens.extend(enc.encode(f"{turn['role'].upper()}: {turn['content'].strip()}\n"))
                    tokens.append(enc.encode("<|endoftext|>")[0])
        else:
            with open(path, "r", encoding="utf-8") as f:
                tokens = enc.encode(f.read())
        self.data = torch.tensor(tokens, dtype=torch.long)
        print(f"Dataset: {len(self.data):,} tokens | device: {DEVICE}")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

def train():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/conversations.jsonl"):
        starter = [
            {"conversation": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]},
            {"conversation": [{"role": "user", "content": "What can you do?"}, {"role": "assistant", "content": "I can chat, answer questions, and assist with tasks."}]},
            {"conversation": [{"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}]},
        ]
        with open("data/conversations.jsonl", "w") as f:
            for item in starter:
                f.write(json.dumps(item) + "\n")
        print("Starter data written. Add more to data/conversations.jsonl for better results.")

    ds = ConversationDataset("data/conversations.jsonl", BLOCK_SIZE)
    n  = int(0.9 * len(ds))
    train_loader = DataLoader(torch.utils.data.Subset(ds, range(n)),      batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(ds, range(n, len(ds))), batch_size=BATCH_SIZE)

    config = GPTConfig()
    config.block_size = BLOCK_SIZE
    model  = GPT(config).to(DEVICE)
    print(f"Params: {model.param_count()/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    best_val  = float("inf")
    step      = 0

    for _ in range(99999):
        model.train()
        for x, y in train_loader:
            if step >= MAX_ITERS: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step()

            if step % EVAL_INTERVAL == 0:
                model.eval()
                vl = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        _, l = model(vx.to(DEVICE), vy.to(DEVICE))
                        vl.append(l.item())
                        if len(vl) >= 20: break
                val_loss = float(np.mean(vl))
                print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"step": step, "model": model.state_dict(), "config": config.__dict__, "val_loss": val_loss}, CHECKPOINT)
                    print(f"  saved checkpoint (val={val_loss:.4f})")
                model.train()
            step += 1
        if step >= MAX_ITERS: break
    print("Done.")

if __name__ == "__main__":
    train()
