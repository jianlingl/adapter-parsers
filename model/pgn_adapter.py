from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn import Parameter, ParameterList, init
from transformers.models.bert.modeling_bert import BertOutput, BertSelfOutput
from transformers import XLMRobertaModel
from typing import Optional


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.matmul(x, w)
    if b is not None:
        if b.dim() != 1:
            b = b.unsqueeze(-2)
        y = y + b
    return y


class Adapter(nn.Module):

    def __init__(
        self, in_feats: int = 768, adapter_size: int = 64, bias: bool = True,
        train_layer_norm: bool = True, dynamic_weights: bool = False
    ):
        super(Adapter, self).__init__()

        self.in_feats = in_feats
        self.adapter_size = adapter_size
        self.bias = bias

        self.weight_down = None
        self.weight_up = None
        self.bias_down = None
        self.bias_up = None
        self.act_fn = nn.GELU()
        self.train_layer_norm = train_layer_norm
        self.dynamic_weights = dynamic_weights

        if not dynamic_weights:
            self.weight_down = nn.Parameter(torch.Tensor(in_feats, adapter_size))
            self.weight_up = nn.Parameter(torch.Tensor(adapter_size, in_feats))

            if bias:
                self.bias_down = nn.Parameter(torch.zeros(adapter_size))
                self.bias_up = nn.Parameter(torch.zeros(in_feats))

            self.reset_parameters()

    def forward(self, hidden_states: torch.Tensor):
        x = batched_linear(hidden_states, self.weight_down, self.bias_down)
        x = self.act_fn(x)
        x = batched_linear(x, self.weight_up, self.bias_up)
        x = x + hidden_states
        return x

    def reset_parameters(self) -> None:
        init.normal_(self.weight_down, std=1e-3)
        init.normal_(self.weight_up, std=1e-3)

    def update_weight(self, weight_name: str, weight: torch.Tensor) -> None:
        object.__setattr__(self, weight_name, weight)


class AdapterBertLayer(nn.Module):
    """
    替代 BertOutput 和 BertSelfOutput
    """
    def __init__(self, base: Union[BertOutput, BertSelfOutput], adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter
        for param in base.LayerNorm.parameters():
            param.requires_grad = adapter.train_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterXLMRobertaModel(nn.Module):
	def __init__(
        self,
        model_name_or_path: str,
        adapter_num: int = 12,
        external_param: Union[bool, List[bool]] = False
    ):
		super().__init__()
		self.xlmr = XLMRobertaModel.from_pretrained(model_name_or_path)
		
		# for param in self.xlmr.parameters():
		# 	param.requires_grad = False
		
		# self.adapters_groups = self.insert_adapters(adapter_num, external_param)
	
	def insert_adapters(self, adapters_num: int, external_param: bool) -> nn.ModuleList:
		
		adapters_groups = nn.ModuleList()
		for i in range(adapters_num):
			
			adapter_a = Adapter(dynamic_weights=external_param)
			adapter_f = Adapter(dynamic_weights=external_param)

			layer = self.xlmr.encoder.layer[i]
			layer.output = AdapterBertLayer(layer.output, adapter_a)
			layer.attention.output = AdapterBertLayer(layer.attention.output, adapter_f)
			
			adapters_groups.append(nn.ModuleList([adapter_a, adapter_f]))
		
		return adapters_groups
	
	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
		return self.xlmr(input_ids, attention_mask, **kwargs)  # (sequence_output, pooled_output) + encoder_outputs[1:]


class PGNAdapterXLMRobertaModel(AdapterXLMRobertaModel):
	def __init__(self,
				 model_name_or_path: str,
				 lan_dim: int = 8,
				 lan_num: int = 3,
				 adapter_size: int = 128,
				 pgn_layers: int = 12,
				 share_param: bool = False):
		super().__init__(model_name_or_path, adapter_size, pgn_layers, True)
		
		# self.embedding = nn.Embedding(lan_num, lan_dim)
		self.embedding = nn.Embedding(3, lan_dim)
		self.pgns_groups = self.pgns_groups_init(lan_dim)
		
	def pgns_groups_init(self, pgn_emb_dim: int) -> nn.ModuleList:
		pgns_groups = nn.ModuleList()
		for adapters_group in self.adapters_groups:
			pgns_group = nn.ModuleList()
			for adapter in adapters_group:
				weights_dict = {
					'weight_down': nn.Parameter(init.normal_(torch.Tensor(
						adapter.in_feats, adapter.adapter_size, pgn_emb_dim), std=1e-3)),
					'weight_up': nn.Parameter(init.normal_(torch.Tensor(
						adapter.adapter_size, adapter.in_feats, pgn_emb_dim), std=1e-3))
				}
				if adapter.bias:
					weights_dict['bias_down'] = nn.Parameter(torch.zeros(adapter.adapter_size, pgn_emb_dim))
					weights_dict['bias_up'] = nn.Parameter(torch.zeros(adapter.in_feats, pgn_emb_dim))
				pgns_group.append(nn.ParameterDict(weights_dict))
			assert len(pgns_group) == len(adapters_group)
			pgns_groups.append(pgns_group)
		assert len(pgns_groups) == len(self.adapters_groups)
		return pgns_groups

	def forward(self, lan_ids: torch.LongTensor, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
		self.parameters_generation(lan_ids)
		res = self.xlmr(input_ids, **kwargs)
		self.flush_parameters()
		return res
	
	def parameters_generation(self, pgn_ids: torch.LongTensor) -> None:
		e = torch.matmul(pgn_ids, self.embedding.weight)  # [batch, dim]
		
		for adapters_g, pgns_g in zip(self.adapters_groups, self.pgns_groups):
			for adapter, pgn in zip(adapters_g, pgns_g):
				for weight_name, pgn_weight in pgn.items():
					ALPHA = 'abcdefg'
					dims = ALPHA[:pgn_weight.dim() - 1]
					adapter_weight = torch.einsum(f'{dims}k,nk->n{dims}', pgn_weight, e)
					adapter.update_weight(weight_name, adapter_weight)
	
	def flush_parameters(self):
		for adapters_g, pgns_g in zip(self.adapters_groups, self.pgns_groups):
			for adapter, pgn in zip(adapters_g, pgns_g):
				for weight_name, pgn_weight in pgn.items():
					adapter.update_weight(weight_name, None)
	
	# def forward(self, lan_id, input_ids: torch.Tensor, mask: torch.Tensor = None, **kwargs):
	# 	self.set_lan(lan_id)
	# 	return self.xlmr(input_ids, mask, **kwargs)  # (sequence_output, pooled_output) + encoder_outputs[1:]
