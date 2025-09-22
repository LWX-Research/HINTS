import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import *
from model_util import QuaternionLinear, qsvd_reconstruction

class HINTS(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super(HINTS, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.gcn_dim =32
        self.dim = in_channels + self.gcn_dim
        self.nodes_num = 96
        self.modalities = ['t1', 't1ce', 't2']
        self.prototypes = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(num_prototypes, self.dim)) 
            for k in ['t1', 't1ce', 't2', 'share']
        })
        self.mlp_dict = nn.ModuleDict({k: self._build_mlp() for k in ['t1', 't1ce', 't2', 'share']})
        self.modal_encoders = nn.ModuleDict({
            modal: nn.ModuleList([
                GCNConv(in_channels, self.gcn_dim//2),
                GCNConv(self.gcn_dim//2, self.gcn_dim)
            ])for modal in self.modalities
        })
        self.bn_layers = nn.ModuleDict({
            modal: nn.ModuleDict({
                '1': nn.BatchNorm1d(self.gcn_dim//2),
                '2': nn.BatchNorm1d(self.gcn_dim)
            }) for modal in self.modalities
        })
        self.pred_head = nn.Sequential(
            nn.Linear(self.dim*2, 32),  
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(), 
        )                
        self.q_linear = nn.Sequential(
            QuaternionLinear(self.dim*2, 256),
            QuaternionLinear(256, self.dim*2),
        )
        self.linear_q = nn.Sequential(
            nn.Linear(self.dim*2, 256),  
            nn.ReLU(),
            nn.Linear(256, self.dim*2),
        )

    def _build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, 1)
        )
    
    def HPA(self, features, mod, moto=0.9):
        shared_query = self.prototypes['share'].unsqueeze(0)  # [1, N_shared, D]
        modality_query = self.prototypes[mod].unsqueeze(0)    # [1, N_mod, D]
        key = value = features  # [B, S, D]

        shared_query = shared_query.expand(key.size(0), -1, -1)    # [B, N_shared, D]
        modality_query = modality_query.expand(key.size(0), -1, -1)  # [B, N_mod, D]
        
        shared_attn_scores = torch.bmm(shared_query, key.transpose(1,2)) / (shared_query.size(-1) ** 0.5)
        shared_attn_weights = F.softmax(shared_attn_scores, dim=-1)
        updated_shared = torch.einsum('bnm,bmd->bnd', shared_attn_weights, value)
        updated_shared = shared_query + F.layer_norm(updated_shared, updated_shared.shape[-1:])
        
        modality_attn_scores = torch.bmm(modality_query, key.transpose(1,2)) / (modality_query.size(-1) ** 0.5)
        modality_attn_weights = F.softmax(modality_attn_scores, dim=-1)
        updated_modality = torch.einsum('bnm,bmd->bnd', modality_attn_weights, value)
        updated_modality = modality_query + F.layer_norm(updated_modality, updated_modality.shape[-1:])
        
        if not hasattr(self, 'shared_updates'):
            self.shared_updates = {}
        self.shared_updates[mod] = updated_shared.mean(0)
        
        if self.training:
            setattr(self, f'updated_{mod}', updated_modality.mean(0))
        
        def generate_feature(updated_prototypes, mlp):
            weights = mlp(updated_prototypes).squeeze(-1)  # [B, N]
            weights = F.softmax(weights, dim=-1)
            return torch.einsum('bn,bnd->bd', weights, updated_prototypes)  # [B, D]
        
        shared_feature = generate_feature(updated_shared, self.mlp_dict['share'])
        modality_feature = generate_feature(updated_modality, self.mlp_dict[mod])
        
        return shared_feature, modality_feature

    def update_all_prototypes(self, moto=0.9):
        if not self.training or not hasattr(self, 'shared_updates'):
            return
            
        if len(self.shared_updates) == 3:  
            shared_mean = torch.stack(list(self.shared_updates.values())).mean(0)
            self.prototypes['share'] = self.prototypes['share'] * moto + (1-moto) * shared_mean
            
            for mod in self.modalities:
                if hasattr(self, f'updated_{mod}'):
                    updated_mod = getattr(self, f'updated_{mod}')
                    self.prototypes[mod] = self.prototypes[mod] * moto + (1-moto) * updated_mod
                    delattr(self, f'updated_{mod}')
            
            self.shared_updates = {}

    def encode_modality(self, 
                        x: torch.Tensor, 
                        edge_index: torch.Tensor, 
                        modality: str) -> torch.Tensor:
            x = self.modal_encoders[modality][0](x, edge_index)
            x = self.bn_layers[modality]['1'](x)
            x = F.relu(x)
            
            x = self.modal_encoders[modality][1](x, edge_index)
            x = self.bn_layers[modality]['2'](x)
            x = x.view(batch_size, self.nodes_num, -1)
            return F.relu(x)

    def SQC(self, features):
        query = torch.mean(features, dim=-1)
        query = self.linear_q(query)  
        features=self.q_linear(features.unsqueeze(1))
        features=qsvd_reconstruction(features).squeeze(1)
        key=value= features

        attn = torch.einsum('bd,bdn->bn', query, key) / (query.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)      
        
        output = torch.einsum('bn,bdn->bd', attn, value)
        return output  # shape: [B, D]
    
    
    def forward(self, data):
        encoded = {}
        
        for mod in self.modalities:
            encoded_feat = self.encode_modality(
                getattr(data, f'{mod}_x'),
                getattr(data, f'{mod}_edge_index'),
                mod
            )
            
            original_view = getattr(data, f'{mod}_x').view(-1, self.nodes_num, in_channels)
            
            encoded[mod] = torch.cat([original_view, encoded_feat], dim=2)

        results = [self.HPA(encoded[mod], mod) for mod in self.modalities]
        t1_shared, t1_mod = results[0]
        t1c_shared, t1c_mod = results[1]
        t2_shared, t2_mod = results[2]
     
        share_feature = torch.stack([t1_shared, t1c_shared, t2_shared], dim=-1)  
        distinct = torch.stack([t1_mod, t1c_mod, t2_mod], dim=-1)   

        features = torch.cat([
            share_feature, 
            distinct
            ],dim=1)
        
        share_quaternion=self.SQC(
            torch.cat([
                torch.zeros_like(features[:,:,0].unsqueeze(-1)),
                features
                ],dim=-1)
        )
       
        pred = self.pred_head(share_quaternion)
        
        self.update_all_prototypes()
        
        return pred, share_feature, distinct

    def initialize_prototypes(self, data_loader, num_batches=1):
            from sklearn.cluster import KMeans
            import numpy as np
            
            device = next(self.parameters()).device
            
            all_features = {mod: [] for mod in self.modalities}
            shared_features = []
            
            self.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    batch = batch.to(device)
                    for mod in self.modalities:
                        original_feat = getattr(batch, f'{mod}_x').view(-1, self.nodes_num, in_channels)    
                        encoded_feat = self.encode_modality(
                            getattr(batch, f'{mod}_x'),
                            getattr(batch, f'{mod}_edge_index'),
                            mod
                        )
                        combined_feat = torch.cat([original_feat, encoded_feat], dim=2)
                        all_features[mod].append(combined_feat.reshape(-1, self.dim).cpu().numpy())  # [N, D]
                        shared_features.append(combined_feat.reshape(-1, self.dim).cpu().numpy())  # [N, D]
            
            for mod in self.modalities:
                all_features[mod] = np.concatenate(all_features[mod], axis=0)
                
            shared_features = np.concatenate(shared_features, axis=0)  # [3*N, D]
            for mod in self.modalities:
                kmeans = KMeans(
                    n_clusters=num_prototypes, 
                    random_state=seed,
                    n_init=20, 
                    max_iter=300,
                    init='k-means++' 
                )
                kmeans.fit(all_features[mod])
                self.prototypes[mod].data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
            
            kmeans = KMeans(
                n_clusters=num_prototypes, 
                random_state=seed,
                n_init=20,  
                max_iter=300,
                init='k-means++'
            )
            kmeans.fit(shared_features)
            self.prototypes['share'].data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
            self.train()
