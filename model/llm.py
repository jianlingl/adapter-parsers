import sys
sys.path.append('..')
# sys.path.append('/data/ljl/adapter-parsers/')
from config import LLMParam
import torch
import torch.nn as nn
import torch.nn.functional as F
from lang2vec.lang2vec import get_features
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from typing import Optional
from time import time

get_trainable_parameters_size = lambda model: sum([param.numel() for _, param in model.named_parameters() if param.requires_grad])


class BatchLoraLinear(nn.Module):
    def __init__(self, batch_weight: Optional[torch.Tensor] = None):
        super(BatchLoraLinear, self).__init__()
        self.weight = batch_weight
    
    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.weight)


class LLM(nn.Module):
    def __init__(self, LMpara: LLMParam):
        super(LLM, self).__init__()
        llm_path = LMpara.plm
        self.llm = AutoModel.from_pretrained(llm_path, revision='master', device_map='cuda:0', torch_dtype=LMpara.lm_dtype)
        self.use_adapter = LMpara.use_adapter
        self.use_lang_emb = LMpara.use_lang_emb

        if not self.use_adapter and 'bloom' in llm_path:
            assert False, "better not run this setting, waste of time!"
            for param in self.llm.parameters():
                param.requires_grad = False

        if self.use_adapter:
            lora_r, lora_alpha, lora_dropout = LMpara.lora_r, LMpara.lora_alpha, LMpara.lora_dropout
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=['query', 'key', 'value', 'query_key_value']
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.lora_size, self.llm_size = self.llm.get_nb_trainable_parameters()

        if self.use_lang_emb:
            assert self.use_adapter, "Language embedding requires adapter"
            lang_hidden_dim, lang_dim = LMpara.lang_hidden_dim, LMpara.lang_dim
            self.lang_MLP = Lang_MLP(lang_hidden_dim, lang_dim)

            lora_num, lora_A, lora_B = self.get_lora_modules(lora_r)
            self.lora_init_replace()
            self.PGN = PGN(lang_dim, lora_num, lora_r, lora_A, lora_B, self.lora_A_keys, self.lora_B_keys)

    def get_lora_modules(self, lora_r):
        lora_A_pattern, lora_B_pattern = {}, {}
        for name, module in self.llm.named_modules():
            if 'lora' in name and isinstance(module, nn.Linear):
                if 'lora_A' in name: lora_A_pattern[name.replace('.', '_dot_')] = module.weight.size()
                if 'lora_B' in name: lora_B_pattern[name.replace('.', '_dot_')] = module.weight.size()
        
        assert len(lora_A_pattern) == len(lora_B_pattern), "Mismatch in lora_A and lora_B"
        lora_num = len(lora_A_pattern)
        r1, lora_A = list(lora_A_pattern.values())[0][0], list(lora_A_pattern.values())[0][1]
        r2, lora_B = list(lora_B_pattern.values())[0][1], list(lora_B_pattern.values())[0][0]
        assert lora_r == r1 == r2, "Mismatch in lora_r"
        self.lora_A_keys, self.lora_B_keys = list(lora_A_pattern.keys()), list(lora_B_pattern.keys())
        return lora_num, lora_A, lora_B

    def lora_init_replace(self):
        def recur_replace(module: nn.Module, name: str, lora_A_keys: list, lora_B_keys: list):
            for sub_name, sub_module in module.named_children():
                full_name = f"{name}_dot_{sub_name}" if name else sub_name
                if full_name in lora_A_keys and isinstance(sub_module, nn.Linear):
                    setattr(module, sub_name, BatchLoraLinear(None))
                elif full_name in lora_B_keys and isinstance(sub_module, nn.Linear):
                    setattr(module, sub_name, BatchLoraLinear(None))
                else:
                    recur_replace(sub_module, full_name, lora_A_keys, lora_B_keys)
        self.llm.print_trainable_parameters()
        recur_replace(self.llm, '', self.lora_A_keys, self.lora_B_keys)

    def lora_weight_replace(self, batch_W_A: nn.ParameterDict, batch_W_B: nn.ParameterDict):
        def recur_replace(module: nn.Module, name: str, b_W_A: nn.ParameterDict, b_W_B: nn.ParameterDict):
            for sub_name, sub_module in module.named_children():
                full_name = f"{name}_dot_{sub_name}" if name else sub_name
                if full_name in b_W_A.keys() and isinstance(sub_module, BatchLoraLinear):
                    sub_module.weight = b_W_A[full_name]
                    # setattr(sub_module, 'weight', b_W_A[full_name])
                elif full_name in b_W_B.keys() and isinstance(sub_module, BatchLoraLinear):
                    sub_module.weight = b_W_B[full_name]
                    # setattr(sub_module, 'weight', b_W_B[full_name])
                else:
                    recur_replace(sub_module, full_name, b_W_A, b_W_B)
        recur_replace(self.llm, '', batch_W_A, batch_W_B)
        cur_trainable_size = get_trainable_parameters_size(self.llm)
        batch_size = list(batch_W_A.values())[0].size(0)
        assert cur_trainable_size == self.lora_size * batch_size, f"Mismatch in trainable parameters{cur_trainable_size, self.lora_size, batch_size}"

    def lora_weight_flush(self):
        def recur_replace(module: nn.Module, name: str):
            for sub_name, sub_module in module.named_children():
                full_name = f"{name}_dot_{sub_name}" if name else sub_name
                if isinstance(sub_module, BatchLoraLinear):
                    sub_module.weight = None
                    # setattr(sub_module, 'weight', None)
                else:
                    recur_replace(sub_module, full_name)
        recur_replace(self.llm, '')

    def print_trainable_parameters(self):
        print(f"Trainable parameters: {get_trainable_parameters_size(self.llm)}")
        if self.use_adapter:
            print("use adapter:")
            self.llm.print_trainable_parameters()
        elif self.use_lang_emb:
            print(f"use lang emb: adapter size {self.PGN.param_size}")
    
    def forward(self, batch_input_ids, batch_attention_mask, batch_langs=None):
        print(batch_langs)
        if self.use_lang_emb:
            # time_start = time()
            lang_embs = self.lang_MLP(batch_langs)
            # lang_MLP_time = time()
            # print(f"lang_MLP time: {lang_MLP_time - time_start}")
            batch_W_A, batch_W_B = self.PGN(lang_embs)
            # PGN_time = time()
            # print(f'PGN time: {PGN_time - lang_MLP_time}')
            self.lora_weight_replace(batch_W_A, batch_W_B)
            # weight_replace_time = time()
            # print(f'weight replace time: {weight_replace_time - PGN_time}')
            assert self.use_adapter, "Adapter is required"

        repre = self.llm(batch_input_ids, batch_attention_mask).last_hidden_state
        # repre_time = time()
        # print(f'repre time: {repre_time - weight_replace_time}')
        if self.use_lang_emb: 
            self.lora_weight_flush()
            # flush_time = time()
            # print(f'flush time: {flush_time - repre_time}')
        return repre


class Lang_MLP(nn.Module):
    # lang2vec,用knn包含最多语言的特征
    def __init__(self, hidden_dim, lang_dim):
        super(Lang_MLP, self).__init__()
        self.feats_set = ["syntax_knn", "phonology_knn", "inventory_knn"]
        self.input_dim = 79 + 20 + 89
        self.langs_1hot = self.get_langs_1hot()
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, lang_dim)

    def get_langs_1hot(self):
        possible_lans = ['eng', 'deu', 'fra', 'heb', 'hun', 'jpn', 'kor', 'swe', 'zho', 'akk', 'kaz', 'kmr', 'mar', 'san', 'tam', 'yor']
        one_hots = [(get_features(possible_lans, feat, minimal=True)) for feat in self.feats_set]
        tmp_dict = {lan: one_hots[0][lan] + one_hots[1][lan] + one_hots[2][lan] for lan in possible_lans}
        assert all(len(item) == self.input_dim for item in tmp_dict.values()), "Mismatch in input dimensions"
        return tmp_dict
    
    def get_lang_1hot(self, batch_langs):
        return torch.tensor([self.langs_1hot[lan] for lan in batch_langs], dtype=torch.float32).to('cuda')

    def forward(self, batch_langs):
        lang_1hot = self.get_lang_1hot(batch_langs)
        lang_embs = F.relu(self.fc1(lang_1hot))
        lang_embs = self.fc2(lang_embs)
        return lang_embs


class PGN(nn.Module):
    def __init__(self, lang_dim, lora_num, lora_r, lora_A, lora_B, lora_A_keys, lora_B_keys) -> None:
        super(PGN, self).__init__()
        self.lang_dim = lang_dim
        self.lora_num = lora_num
        self.lora_r = lora_r
        self.lora_A = lora_A
        self.lora_B = lora_B

        assert len(lora_A_keys) == len(lora_B_keys) == lora_num, "Mismatch in lora_A and lora_B"

        self.paramsA = nn.ParameterDict({key_A: nn.Parameter(torch.randn(lang_dim, lora_A, lora_r)) for key_A in lora_A_keys})
        self.paramsB = nn.ParameterDict({key_B: nn.Parameter(torch.randn(lang_dim, lora_r, lora_B)) for key_B in lora_B_keys})

        self.param_size = sum(p.numel() for p in self.paramsA.values()) + sum(p.numel() for p in self.paramsB.values())

    def forward(self, batch_lang_embs):
        batch_size = batch_lang_embs.size(0)

        batch_W_A = nn.ParameterDict({
            key_A: torch.einsum('bi, birk->brk', batch_lang_embs, self.paramsA[key_A].expand(batch_size, -1, -1, -1)) 
            for key_A in self.paramsA.keys()})
        batch_W_B = nn.ParameterDict({
            key_B: torch.einsum('bi, bikr->bkr', batch_lang_embs, self.paramsB[key_B].expand(batch_size, -1, -1, -1)) 
            for key_B in self.paramsB.keys()})
        return batch_W_A, batch_W_B


# if __name__ == '__main__':
    # a = PGN(8, 24, 8, 768, 768, ['A' + str(i) for i in range(24)], ['B' + str(i) for i in range(24)])
    # print(a.param_size)
    # print(a.paramsA['A0'].shape)
    # print(a.paramsB['B0'].shape)