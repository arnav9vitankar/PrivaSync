# === IMPORTS ===
import asyncio
import base64
import io
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import websockets

# === LOGGING === 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# === CONSTANTS ===
WS_URL = 'ws://localhost:8000/ws/train'
CHUNK_SIZE = 1024 * 1024
TIMEOUT = 120
CONFIG_FILE = 'config.json'

# Locations searched for the global model file (in order)
MODEL_FILES = [os.path.expanduser('~/Downloads/global_model.pt'), os.path.expanduser('~/Downloads/global_model_initial.pt'), 'global_model.pt', 'global_model_initial.pt']

# Configuration dataclasses for model and training parameters
@dataclass
class ModelConfig:
    vocab_size: int = 5000
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 64

@dataclass
class TrainConfig:
    epochs_per_round: int = 3
    batch_size: int = 32
    learning_rate: float = 3e-4
    grad_clip: float = 1.0

# === SPECIAL TOKENS ===
PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
SPECIAL = [PAD, UNK, BOS, EOS]

class WordTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inverse_vocab = {i: w for w, i in self.vocab.items()}

    def _words(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts, max_size=5000):
        counts = Counter()
        for text in texts:
            counts.update(self._words(text))
        words = [w for w, _ in counts.most_common(max_size - len(SPECIAL))]
        self.vocab = {w: i for i, w in enumerate(SPECIAL + words)}
        self.inverse_vocab = {i: w for w, i in self.vocab.items()}

    def encode(self, text):
        unk = self.vocab[UNK]
        return [self.vocab.get(w, unk) for w in self._words(text)]

    def decode(self, ids):
        return ' '.join(self.inverse_vocab.get(i, UNK) for i in ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))

# === DATASET ===
class LocalTextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.ids = tokenizer.encode(text)
        if len(self.ids) <= seq_len:
            raise ValueError(f'Not enough text: got {len(self.ids)} tokens but need > {seq_len}. Provide more training text.')
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, i):
        return (torch.tensor(self.ids[i:i+self.seq_len], dtype=torch.long),
                torch.tensor(self.ids[i+1:i+self.seq_len+1], dtype=torch.long))

# === MODEL ===
class GRULanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
        drop = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.gru = nn.GRU(cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers, dropout=drop, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        return self.head(self.dropout(x)), hidden

# === SERIALIZATION ===
def pack_weights(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()

def unpack_weights(data):
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)

# === TRAINING ===  
def device_for_training():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def train(model, dataset, cfg):
    device = device_for_training()
    log.info(f'Training device: {device}')
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(cfg.epochs_per_round):
        total = 0.0
        batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item()
            batches += 1
        log.info(f'Epoch {epoch + 1}/{cfg.epochs_per_round} - avg loss: {total / max(batches, 1):.4f}')
    model.cpu()
    return pack_weights(model.state_dict())

# === WEBSOCKET UPLOAD ===
async def upload(token, data, round_no):
    chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
    log.info(f'Connecting to {WS_URL}, {len(chunks)} chunk(s)')
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({'type': 'auth', 'token': token}))
        for i, chunk in enumerate(chunks):
            await ws.send(json.dumps({'type': 'weights_chunk', 'round': round_no, 'chunk_index': i, 'total_chunks': len(chunks), 'data': chunk}))
            log.info(f'Sent chunk {i + 1}/{len(chunks)}')
            await asyncio.sleep(0.05)
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT)
            except asyncio.TimeoutError:
                log.error(f'No response from server after {TIMEOUT}s')
                return False
            msg = json.loads(raw)
            kind = msg.get('type')
            if kind == 'round_complete':
                return True
            if kind == 'rejected':
                log.warning(f"Server rejected submission: {msg.get('reason', 'no reason given')}")
                return False
            log.info(f'Server: {msg}')

# === CONFIG AND MODEL LOADING HELPERS ===
def read_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"'{CONFIG_FILE}' not found. Download the training bundle from the PrivaSync dashboard first.")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    os.remove(CONFIG_FILE)
    return cfg

def find_model():
    for path in MODEL_FILES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('Global model file not found. Searched:\n' + '\n'.join('  ' + p for p in MODEL_FILES))

def get_tokenizer(vocab_path, text, cfg):
    if vocab_path and os.path.exists(vocab_path):
        return WordTokenizer.load(vocab_path)
    log.warning('Building a local vocabulary for testing')
    tok = WordTokenizer()
    tok.build_vocab([text], cfg.vocab_size)
    return tok

# === MAIN PIPELINE ===
async def main():
    try:
        cfg = read_config()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.error(f'Config error: {e}')
        return

    token = cfg.get('jwt', '')
    text = cfg.get('data', '')
    round_no = cfg.get('round', 1)
    vocab_path = cfg.get('vocab_path')

    if not token or not text:
        log.error("Missing 'jwt' or 'data' in config.json")
        return

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    try:
        model_path = find_model()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    model = GRULanguageModel(model_cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    tokenizer = get_tokenizer(vocab_path, text, model_cfg)
    try:
        dataset = LocalTextDataset(text, tokenizer, model_cfg.seq_len)
    except ValueError as e:
        log.error(f'Dataset error: {e}')
        return

    try:
        weights = train(model, dataset, train_cfg)
    except Exception as e:
        log.error(f'Training failed: {e}')
        return

    encoded = base64.b64encode(weights).decode('utf-8')
    try:
        ok = await upload(token, encoded, round_no)
    except Exception as e:
        log.error(f'WebSocket error: {e}')
        return

    torch.save(unpack_weights(weights), 'local_model.pt')
    if ok:
        log.info('Federated learning round completed successfully.')
    else:
        log.warning('Weights uploaded but server did not confirm the round.')

# === ENTRY POINT ===
if __name__ == '__main__':
    asyncio.run(main())