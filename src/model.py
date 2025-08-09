
import json
from pathlib import Path
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

__all__ = [
    "PairDataset",
    "SiameseModel",
]

class PairDataset(Dataset):
    def __init__(self, pair_source):
        # allow str, Path, or file-like
        if isinstance(pair_source, (str, os.PathLike, Path)):
            path = Path(pair_source)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        elif hasattr(pair_source, "read"):
            data = json.load(pair_source)
        else:
            # already a list[dict]
            data = pair_source
        self.text1   = [d["text1"] for d in data]
        self.text2   = [d["text2"] for d in data]
        self.labels  = torch.tensor([d["label"] for d in data], dtype=torch.float32)
        # authors may or may not exist in older files
        self.author1 = [d.get("author1") for d in data]
        self.author2 = [d.get("author2") for d in data]
        self.has_authors = any(a is not None for a in self.author1)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        if self.has_authors:
            return self.text1[idx], self.text2[idx], self.labels[idx], self.author1[idx], self.author2[idx]
        else:
            return self.text1[idx], self.text2[idx], self.labels[idx]
    
    @classmethod
    def from_list(cls, items):
        tmp = Path("__tmp_pairs.json")         
        tmp.write_text(json.dumps(items, ensure_ascii=False))
        return cls(tmp)

# Siamese model
class SiameseModel(nn.Module):
    # Sentence-BERT encoder → projection MLP → similarity MLP
    def __init__(self,
                 encoder_name="all-MiniLM-L6-v2",
                 proj_dim=256,
                 mlp_hidden=512,
                 dropout=0.2,
                 init_temp=10.0,
                 device=None):
        super().__init__()

        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )

        self.encoder = SentenceTransformer(encoder_name, device=device)
        emb_dim = self.encoder.get_sentence_embedding_dimension()

        # projection head
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim, bias=False)
        )

        # MLP scorer
        self.scorer = nn.Sequential(
            nn.Linear(proj_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1, bias=False)
        )

        self.temp = nn.Parameter(torch.tensor([init_temp]))

    # tokenise, run through the HF model with gradients, then mean-pool token vectors to sentence embedding
    def _embed(self, sentences):
        device = next(self.parameters()).device
        tok = self.encoder.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)                                # move all tensors

        # Hugging-Face model inside the first ST module
        hf_model = self.encoder._first_module().auto_model.to(device)

        out = hf_model(**tok, return_dict=True)     # gradients ON

        mask = tok["attention_mask"].unsqueeze(-1)
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb                                  # [B, hidden_dim]

    # forward 
    def forward(self, s1, s2):
        e1 = self._embed(list(s1))
        e2 = self._embed(list(s2))

        z1 = F.normalize(self.proj(e1), p=2, dim=-1)
        z2 = F.normalize(self.proj(e2), p=2, dim=-1)

        feats  = torch.cat([z1 * z2, (z1 - z2).abs()], dim=-1)
        return self.scorer(feats).squeeze(-1) * self.temp


    def encode(self, texts, batch_size=64):
        with torch.inference_mode():
            e = self.encoder.encode(texts, batch_size=batch_size, convert_to_tensor=True)
            z = F.normalize(self.proj(e), p=2, dim=-1)
            return z

    @property
    def embedding_dim(self):
        """Return                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         the dimensionality of sentence embeddings."""
        return self.encoder.get_sentence_embedding_dimension()
    
    # freeze method
    def freeze_encoder_layers(self, n_layers):
        bert = self.encoder._first_module()          
        for name, param in bert.auto_model.named_parameters():
            if name.startswith("encoder.layer."):
                layer_idx = int(name.split(".")[2])
                if layer_idx < n_layers:
                    param.requires_grad = False
