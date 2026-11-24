import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from models.readouts import build_readout


class MonroeDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs = graphs
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index].clone()

        graph.y = self.targets[index].unsqueeze(0)

        return graph


class MonroeModel(nn.Module):

    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        from monroe.model.ckpt import load_ckpt

        self.cfg = cfg
        self.graph_dir = Path(data_dir) / "monroe_graph_cache"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.graph_cache = {}

        self.encoder = load_ckpt(str(cfg.checkpoint)).to(device)

        self.encoder.requires_grad_(not cfg.freeze_encoder)

        self.readout = build_readout(
            readout,
            self.encoder.hidden_dim,
            n_targets
        ).to(device)

        self._device = torch.device(device)

    @property
    def device(self):
        return self._device

    def _cache_path(self, smiles):
        key = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
        return self.graph_dir / f"{key}.npz"

    def _get_graphs(self, frame):
        from monroe.eval.embed import featurize_smiles, to_pyg

        smiles_list = frame["SMILES"].astype(str).tolist()

        missing = list(dict.fromkeys(
            smiles for smiles in smiles_list
            if smiles not in self.graph_cache
            and not self._cache_path(smiles).exists()
        ))

        if missing:

            built = featurize_smiles(
                missing,
                n_workers=int(self.cfg.num_workers)
            )

            for smiles, graph in built:
                np.savez_compressed(
                    self._cache_path(smiles),
                    node_float=graph["node_float"],
                    node_codes=graph["node_codes"],
                    edge_index=graph["edge_index"],
                    edge_codes=graph["edge_codes"],
                    pos_rdkit=graph["pos_rdkit"],
                )

            failed = [
                smiles for smiles in missing
                if not self._cache_path(smiles).exists()
            ]

            if failed:
                raise RuntimeError(
                    f"Monroe could not do {len(failed)} molecules. "
                )

        graphs = []

        for smiles in smiles_list:

            if smiles not in self.graph_cache:

                with np.load(self._cache_path(smiles)) as saved:
                    graph_dict = {
                        name: saved[name].copy()
                        for name in [
                            "node_float",
                            "node_codes",
                            "edge_index",
                            "edge_codes",
                            "pos_rdkit",
                        ]
                    }

                self.graph_cache[smiles] = to_pyg(graph_dict)

            graphs.append(self.graph_cache[smiles])

        return graphs

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        from torch_geometric.loader import DataLoader

        graphs = self._get_graphs(frame)

        y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

        dataset = MonroeDataset(graphs, y)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def param_groups(self):

        groups = [
            {
                "params": list(self.readout.parameters()),
                "lr": self.cfg.readout_lr
            }
        ]
        if not self.cfg.freeze_encoder:
            groups.append(
                {
                    "params": list(self.encoder.parameters()),
                    "lr": self.cfg.encoder_lr
                }
            )

        return groups

    def train(self, mode=True):

        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()

        return self

    def forward(self, batch):

        batch = batch.to(self.device)

        if self.cfg.freeze_encoder:
            with torch.no_grad():
                graph_embedding, _ = self.encoder(batch)
        else:
            graph_embedding, _ = self.encoder(batch)

        return self.readout(graph_embedding)
