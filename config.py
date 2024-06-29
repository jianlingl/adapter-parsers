import argparse
import torch


class LLMParam():
    def __init__(self) -> None:
        self.plm = '/data/hfmodel/bert-base-multilingual-cased'
        # self.plm = '/data/hfmodel/xlm-roberta-large'
        # self.plm = '/data/hfmodel/bloom-7b1'
        self.lm_dtype = torch.float32 if 'bloom' not in self.plm else torch.bfloat16
        self.use_adapter = False
        self.use_lang_emb = False
        self.lang_hidden_dim = 64
        self.lang_dim = 8
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1


class Hparam():
    def __init__(self, args) -> None:
        if isinstance(args, argparse.Namespace):
            # treebank
            self.label_vocab = None
            self.tag_vocab = None
            # train
            self.LMpara = LLMParam()
            self.max_epoch = 80
            self.batch_size = 32
            self.big_batch_size = 32
            self.seed = 6666 if args.seed is None else args.seed
            self.checks_per_epoch = 4
            # 优化器参数
            self.learning_rate = 0.00005
            self.learning_rate_warmup_steps = 160 # decrease if the dataset is small
            self.step_decay_factor = 0.5
            self.epoch_decay_patience = 3
            self.clip_grad_norm = 1000.0
            self.early_stop_patience = 20
            # encoder and score
            self.d_pretrained = 768
            self.d_word = 512
            self.use_tag = False
            self.d_tag = 256
            self.use_position = True
            self.max_snt_len = 160
            self.d_position = 512
            self.encoder_max_len = 512
            # partitioned transformer encoder
            self.num_encoder_layer = 2
            self.num_heads = 8
            self.d_model = 1024
            self.d_ff = 2048
            self.d_qkv = 64
            
            self.morpho_emb_dropout = 0.2
            self.relu_dropout = 0.1
            self.residual_dropout = 0.2
            self.attention_dropout = 0.2

            self.pred_tag = True
            self.d_tag_hidden = 256
            self.tag_loss_scale = 5.0
            self.d_label_hidden = 256

            # dev test pred
            self.evalb_dir = "EVALB"
        
        elif isinstance(args, dict):
            for k, v in args.items():
                setattr(self, k, v)

        else:
            assert False, "Error in input or load hparam"


iso_code = {'en': 'eng', 'de': 'deu', 'fr': 'fra', 'he': 'heb', 'hu': 'hun', 'ja': 'jpn', 'ko': 'kor', 'sv': 'swe', 'zh': 'zho', 'akk': 'akk', 'kk': 'kaz', 'ku': 'kmr', 'mr': 'mar', 'sa': 'san', 'ta': 'tam', 'yo': 'yor'}
