import argparse


class Hparam():
    def __init__(self, args: argparse.Namespace or dict) -> None:
        if isinstance(args, argparse.Namespace):
            # treebank
            self.train_path = args.train_path
            self.dev_path = args.dev_path
            self.test_path = args.test_path
            self.model_path_base = args.model_path_base
            self.device = int(args.device)
            self.label_vocab = None
            self.tag_vocab = None

            # train
            self.max_epoch = 60
            self.batch_size = 4
            self.big_batch_size = 32
            self.numpy_seed = 8888 if args.numpy_seed is None else args.numpy_seed
            self.pretrain_model_path = 'pretrain_model/xlm-roberta-base'
            self.checks_per_epoch = 4

            # 优化器参数
            self.learning_rate = 0.00005
            self.learning_rate_warmup_steps = 250
            self.step_decay_factor = 0.5
            self.epoch_decay_patience = 3
            self.clip_grad_norm = 1000.0
            self.early_stop_patience = 20 


            # encoder and score
            self.d_pretrained = 768
            self.d_word = 512
            self.use_tag = True
            self.d_tag = 128
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

            self.d_label_hidden = 128

            # dev test pred
            self.evalb_dir = "EVALB"
        
        elif isinstance(args, dict):
            for k, v in args.items():
                setattr(self, k, v)
        
        else:
            assert False, "Error in input or load hparam"
