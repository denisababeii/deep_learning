import torch

from monroe.model.ckpt import load_ckpt
from monroe.eval.embed import embed_smiles

checkpoint = "monroe/Monroe/checkpoint"

encoder = load_ckpt(checkpoint)

print("Monroe checkpoint loaded successfully")
print("Hidden dimension:", encoder.hidden_dim)

embeddings = embed_smiles(
    ["CCO", "c1ccccc1"],
    encoder,
    n_workers=1
)

for smiles, embedding in embeddings.items():
    print(smiles, embedding.shape)

print("Monroe test completed successfully!")
