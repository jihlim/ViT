from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F 


class FlattenPatchesLinearProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_patch: int,
        patch_size: int,
    ):
        super(FlattenPatchesLinearProjection, self).__init__()
        
        self.patch_size = patch_size
        self.fc = nn.Linear(d_patch, d_model)
        
    def _forward_impl(self, x):
        # Convert Images into Patches
        x = x.unfold(2, self.patch_size, self.patch_size)                       # [b, c, h//patch_size, w, patch_size]
        x = x.unfold(3, self.patch_size, self.patch_size)                       # [b, c, h//patch_size, w//patch_size, patch_size, patch_size]
        x = x.flatten(2, 3).flatten(3)                                          # [b, c, num_patches, (patch_size ** 2)]
        x = x.transpose(1, 2).flatten(2)                                        # [b, num_patches, (patch_size ** 2) * c]
        
        # Linear Projection
        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class HybridArchitecture(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_patch: int,
        patch_size: int,
    ):
        super(HybridArchitecture, self).__init__()
        
        self.conv = nn.Conv2d()
    
    def _forward_impl(self, x):
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class PositionalEncoding(nn.Module):
    def __init__(
      self,
      d_patch: int,
      num_patches: int,
    ):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros((num_patches + 1, d_patch), dtype=torch.int64)
        pos = torch.arange(0, num_patches+1, dtype=torch.int64)
        dim = torch.arange(0, d_patch, dtype=torch.int64)
        
        dim_even = dim[::2]
        dim_odd = dim[1::2]
        i_even = dim_even
        i_odd = (dim_odd // 2) * 2
        pe[pos, :] = pos[:, None]
        
        pe = pe.to(device=pe.device, dtype=torch.float32)
        
        pe[:, dim_even] = torch.sin(pe[:, dim_even] / (10000 ** (i_even/d_patch)))
        pe[:, dim_odd] = torch.cos(pe[:, dim_odd] / (10000 ** (i_odd/d_patch)))
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def _forward_impl(self, x):
        x = x + self.pe
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        num_patches: int,
        d_patch: int,
        d_query = None,
        d_key = None,
        d_value = None,
    ):
        super(MultiHeadAttention, self).__init__()
        if d_query == None:
            d_query = d_model
        if d_key == None:
            d_key = d_model
        if d_value == None:
            d_value = d_model

        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.qkv_weight = nn.Linear(d_patch, 3*d_model)
        self.multi_head_w_o = nn.Linear(d_model, d_model)
        
    def attention(self, q_k_v, sqrt_d_k):
        q, k, v = q_k_v.chunk(3, dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        attention_weight = torch.bmm(q, k.transpose(1,2))
        attention_weight = attention_weight / sqrt_d_k
        attention_weight = self.softmax(attention_weight)
        out = torch.bmm(attention_weight, v)
        return out
        
    def _forward_impl(
        self, 
        query,
        key,
        value,
    ):
        if key == None:
            key = query
        if value == None:
            value = query
        is_self_attention = torch.equal(query, key) and torch.equal(query, value)
        
        q_k_v = self.qkv_weight(query)
        sqrt_d_k = self.d_head ** 0.5
        
        # Mix q, k, v for Multi-Head Attention
        b, l, c = q_k_v.shape
        q_k_v = q_k_v.contiguous().view(b, l, 3 * self.d_head, self.num_head)           # [b, l, 3 * d_head, num_head]
        q_k_v = q_k_v.transpose(2, 3)                                                   # [b, l, num_head, 3 * d_head]

        # Divide q, k, v for Parallel Computation
        q_k_v = q_k_v.transpose(1, 2)                                                   # [b, num_head, l, 3 * d_head]
        q_k_v = q_k_v.contiguous().view(-1, l, 3 * self.d_head)                         # [b * num_head, l, 3 * d_head]
        head_out = self.attention(q_k_v, sqrt_d_k)                                      # [b * num_head, 1, d_head]
        head_out = head_out.contiguous().view(b, self.num_head, l, self.d_head)         # [b, num_head, l, d_head]
        head_out = head_out.transpose(1, 2)                                             # [b, l, num_head, d_head]
        head_out = head_out.flatten(start_dim=2)                                        # [b, l, d_model]
        out = self.multi_head_w_o(head_out)
        return out
    
    def forward(self, query, key=None, value=None):
        return self._forward_impl(query, key, value)


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 3072,
    ):
        super(FFN, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def _forward_impl(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        out = self.fc2(x)
        return out
    
    def forward(self, x):
        return self._forward_impl(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        d_model: int, 
        num_classes: int,
        is_pretrain: bool = True,
    ):
        super(MLPHead, self).__init__()
        
        self.is_pretrain = is_pretrain
        self.hidden_layer = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
    def _forward_impl(self, x):
        if self.is_pretrain:
            x = self.hidden_layer(x)
            x = torch.tanh(x)
        out = self.fc(x)
        return out
    
    def forward(self, x):
        return self._forward_impl(x)


class ViTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 3072,
        num_head: int = 12,
        num_patches: int = 196,
        d_patch: int = 768,
        is_hybrid_arch: bool = False,
    ):
        super(ViTEncoderLayer, self).__init__()
        
        self.d_model = d_model
        
        self.shortcut1 = nn.Identity()
        self.ln1 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_head=num_head, num_patches=num_patches, d_patch=d_patch)
        self.shortcut2 = nn.Identity()
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        
    def _forward_impl(self, x):
        identity = self.shortcut1(x)
        x = self.ln1(x)
        x = self.multi_head_attention(x)
        x = x + identity
        identity = self.shortcut2(x)
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + identity
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class ViTEncoder(nn.Module):
    def __init__(
      self,
      layer,
      num_layers: int = 12,
    ):
        super(ViTEncoder, self).__init__()
        self.encoder = self._make_encoder(layer, num_layers)
    
    def _make_encoder(
        self, 
        layer, 
        num_layers
    ):
        layer_list = []
        for _ in range(num_layers):
            layer_list.append(layer)
        return nn.Sequential(*layer_list)
    
    def _forward_impl(self, x):
        x = self.encoder(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class ViT(nn.Module):
    r"""
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et at. ICLR 2021)
    paper: https://openreview.net/pdf?id=YicbFdNTTy
    """
    def __init__(
        self, 
        num_layers: int = 12,
        d_model: int = 768,
        d_ff: int = 3072,
        num_head: int = 12,
        image_size: Union[tuple[int], int] = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        is_hybrid_arch: bool = False,
        is_greyscale: bool = False,
        is_pretrain: bool = True,
    ):
        super(ViT, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.patch_size = patch_size
        if isinstance(image_size, int):
            num_patches = (image_size // patch_size) ** 2
        elif isinstance(image_size, tuple) and len(image_size) == 2:
            h, w = image_size
            num_patches = (h * w) // (patch_size ** 2)
        else:
            raise TypeError(f"image_size should be tuple(int) or int, but you have {type(image_size)} of {len(image_size)} elements")
        d_patch = (patch_size ** 2) * 3
        if is_greyscale:
            d_patch = patch_size ** 2
        
        self.class_embedding = nn.Parameter(torch.zeros((1, 1, d_model), dtype=torch.float32))
        
        self.linear_projection = nn.Linear(d_patch, d_model)
        self.pos_enc = PositionalEncoding(d_patch=d_patch, num_patches=num_patches)
        encoder_layer = ViTEncoderLayer(d_model=d_model, d_ff=d_ff, num_head=num_head)
        self.encoder = ViTEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.mlp_head = MLPHead(d_model, num_classes, is_pretrain)
    
    def convert_to_patch(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size)                       # [b, c, h//patch_size, w, patch_size]
        x = x.unfold(3, self.patch_size, self.patch_size)                       # [b, c, h//patch_size, w//patch_size, patch_size, patch_size]
        x = x.flatten(2, 3).flatten(3)                                          # [b, c, num_patches, (patch_size ** 2)]
        x = x.transpose(1, 2).flatten(2)                                        # [b, num_patches, (patch_size ** 2) * c]
        return x
    
    def _forward_impl(self, x):
        x = self.convert_to_patch(x)
        patch_embedding = self.linear_projection(x)
        b, l, c = x.shape
        class_embedding = self.class_embedding.repeat(b, 1, 1)
        x = torch.cat((class_embedding, patch_embedding), dim=1)
        patch_pos_embedding = self.pos_enc(x)
        enc_out = self.encoder(patch_pos_embedding)
        image_representation = enc_out[:, 0]
        y = self.ln(image_representation)
        out = self.mlp_head(y)
        return out
    
    def forward(self, x):
        return self._forward_impl(x)
    
def vit_base(num_classes):
    return ViT(num_classes=num_classes)

def vit_large(num_classes):
    return ViT(num_layers=24, d_model=1024, d_ff=4096, num_head=16, num_classes=num_classes)

def vit_huge(num_classes):
    return ViT(num_layers=32, d_model=1280, d_ff=5120, num_head=16, num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    model = ViT(num_classes=10).to(device)
    output = model(x)
    print(output.shape)