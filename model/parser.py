# 编码 + 解码 + 损失函数
from typing import List, Optional, Dict
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import XLMRobertaModel, AutoModel

from model.pgn_adapter import AdapterXLMRobertaModel, PGNAdapterXLMRobertaModel
from model.tokenization import Retokenizer
from model.encoder import ConcatTag, ConcatPosition, encoderStackLayer, PartitionedTransformerEncoderLayer, FeatureDropout
from model.decoder import ChartDecoder
from model.treebank import Tree
from model.llm import LLM
from config import Hparam


class Parser(nn.Module):
    def __init__(self, hparam: Hparam) -> None:
        super().__init__()
        # 预训练
        self.hparam = hparam
        self.tokenizer = Retokenizer(hparam.LMpara.plm)
        self.llm = LLM(hparam.LMpara)

        # 词表示
        self.tag_vocab = hparam.tag_vocab
        self.use_tag, self.use_position = hparam.use_tag, hparam.use_position
        self.word_feature_dropout = FeatureDropout(hparam.morpho_emb_dropout)

        if hparam.use_tag:
            self.concat_tag = ConcatTag(len(self.tag_vocab), hparam.d_tag, hparam.d_pretrained+hparam.d_tag)
            self.c_project = nn.Linear(hparam.d_pretrained+hparam.d_tag, hparam.d_word, bias=False)
        else:
            self.c_project = nn.Linear(hparam.d_pretrained, hparam.d_word, bias=False)

        if hparam.use_position:
            self.concat_position = ConcatPosition(hparam.max_snt_len, hparam.d_model)        

        self.encoder = encoderStackLayer(PartitionedTransformerEncoderLayer(
            hparam.d_model, 
            n_head=hparam.num_heads, 
            d_qkv=hparam.d_qkv, 
            d_ff=hparam.d_ff, 
            ff_dropout=hparam.relu_dropout, 
            residual_dropout=hparam.residual_dropout, attention_dropout=hparam.attention_dropout, 
        ), num_layers=hparam.num_encoder_layer)

        self.span_linear_cat = nn.Linear(hparam.d_word*4, hparam.d_word*2, bias=True)

        self.decoder = ChartDecoder(hparam.label_vocab)

        # 打分函数
        self.score_label = nn.Sequential(
            # nn.Linear(hparam.d_word*2, hparam.d_word, bias=True),
            nn.Linear(hparam.d_model, hparam.d_label_hidden),
            nn.LayerNorm(hparam.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparam.d_label_hidden, len(hparam.label_vocab)-1),
            # *伪标签也预测，不强制赋值为0
            # nn.Linear(hparam.d_label_hidden, len(hparam.label_vocab)),
        )

        assert not (hparam.use_tag and hparam.pred_tag), "can not be both TRUE"
        self.pred_tag = False
        self.score_tag = None
        if hparam.pred_tag:
            self.pred_tag = True
            self.score_tag = nn.Sequential(
                nn.Linear(hparam.d_model, hparam.d_tag_hidden),
                nn.LayerNorm(hparam.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparam.d_tag_hidden, len(hparam.tag_vocab)),
            )
            self.tag_loss_scale = hparam.tag_loss_scale
        # todo zls 
        # self.score_label 后面加sigmoid ->稳定, logits = logits + Haming_augment必须乘个概率

    def forward(self, tokenized):
        input_ids, attention_mask, words_index, word_attention_mask = tokenized["input_ids"].to('cuda'), tokenized["attention_mask"].to('cuda'), tokenized["words_index"].to('cuda'), tokenized["word_attention_mask"].to('cuda')
        if self.use_tag:
            tag_ids = tokenized["tag_ids"].to('cuda')
        # span表示 # 在头尾多加一维
        # words_index = torch.cat([words_index, torch.zeros(words_index.size()[0], 1, dtype=torch.long).to('cuda')], dim=-1)
        # words_index[torch.arange(word_attention_mask.size()[0]), word_attention_mask.sum(-1)] = word_attention_mask.sum(-1)
        # words_index = torch.cat([torch.zeros(words_index.size()[0], 1, dtype=torch.long).to('cuda'), words_index], dim=-1)
        # word_attention_mask = torch.cat([torch.ones(word_attention_mask.size()[0], 2, dtype=torch.bool).to('cuda'), word_attention_mask], dim=-1)
        # tag_ids = torch.cat([torch.zeros(tag_ids.size()[0], 1, dtype=int).to('cuda'), tag_ids], dim=-1)
        # tag_ids = torch.cat([tag_ids, torch.zeros(tag_ids.size()[0], 1, dtype=int).to('cuda')], dim=-1)

        features = self.llm(input_ids, attention_mask, tokenized['langs'])

        # 花式索引头词
        batch_index = torch.arange(features.size()[0]).unsqueeze(-1)
        word_repre = features[batch_index, words_index, :]
        word_repre.masked_fill_(~word_attention_mask[:, :, None], 0)

        # 获取更全面词表示
        if self.use_tag:
            word_repre = self.concat_tag(word_repre, tag_ids)
        word_repre = self.c_project(self.word_feature_dropout(word_repre).to(torch.float))

        if self.use_position:
            word_repre = self.concat_position(word_repre)

        # 编码器优化词表示
        # word_repre = self.encoder(word_repre, src_key_padding_mask=~word_attention_mask)
        word_repre = self.encoder(word_repre, mask=word_attention_mask)
        # word_repre = self.w_linear_after_encoder(word_repre)

        # span表示 # 奇偶位 -> 前向和后向的表示
        # features = torch.cat([
        #         features[..., 0::2],
        #         features[..., 1::2],
        #     ], dim=-1,
        # )

        tag_scores = self.score_tag(word_repre) if self.pred_tag else None
        
        span_features = self.span_features(word_repre)

        # 给span打分
        logits_4labels = self.score_label(span_features)
        # return logits_4labels

        logits_4stars = logits_4labels.new_zeros(logits_4labels.size()[:3] + (1, ))
        logits = torch.cat((logits_4stars, logits_4labels), dim=-1)

        return logits, tag_scores

    def compute_loss(self, tokenized):
        logits, tag_scores = self.forward(tokenized)
        gold_event, len_list = tokenized["gold_event"].to('cuda'), tokenized["word_attention_mask"].sum(-1).tolist()

        Haming_augment = 1. - gold_event
        Haming_augment[torch.arange(logits.size()[0]), 0, torch.tensor(len_list) - 1, 0] -= 1e9  # 规避根节点标签为*,通过给Haming_augment赋值一个极小值，使得其空标签的得分特别低，永远不可能预测空标签

        logits = logits + Haming_augment
        # todo * r 概率

        pred_event, _ = self.decoder.CKY(logits.data.cpu().numpy(), len_list)
        pred_score = (torch.tensor(pred_event).to('cuda') * logits).sum([1,2,3])
        gold_score = (gold_event * logits).sum([1,2,3])

        # dist = torch_struct.TreeCRF(logits, lengths=torch.LongTensor(len_list).to('cuda'))
        # pred_score = dist.max
        
        loss = F.relu(pred_score - gold_score).mean()

        if tag_scores is None:
            return loss
        else:
            tag_gold = tokenized["tag_ids"].to('cuda')
            tag_loss = self.tag_loss_scale * F.cross_entropy(
                tag_scores.view(-1, tag_scores.size(-1)), 
                tag_gold.view(-1), 
                reduction='mean',
                ignore_index=0)
            return loss + tag_loss

    def span_features(self, features_out):
        # span_i_j concatination
        words_num = features_out.size()[1]
        features_i = features_out.unsqueeze(2).expand(-1, -1, words_num, -1)
        features_j = features_out.unsqueeze(1).expand(-1, words_num, -1, -1)
        features = self.span_linear_cat(torch.cat((features_i, features_j), dim=-1))
        return features

        # kitaev's span representation
        # fencepost_annotations = torch.cat([
        #         features_out[:, :-1, :features_out.shape[-1]//2],
        #         features_out[:, 1:, features_out.shape[-1]//2:],
        #     ], dim=-1,
        # )
        # span_features = (
        #     torch.unsqueeze(fencepost_annotations, 1) - torch.unsqueeze(fencepost_annotations, 2)
        # )[:, :-1, 1:]  # 分别在第一维和第二维加一个维度  8x1xNx1024-8xNx1x1024
        # return span_features

    def encode_and_collate(self, trees: List[Tree]):
        batched_words = [list(tree.leaves()) for tree in trees]
        batched_tags = [list(tree.pos()) for tree in trees]
        batched_langs = [tree.lang for tree in trees]

        batched_tokenized = self.tokenizer(batched_words)
        batched_tokenized["langs"] = batched_langs
        batched_tokenized["tag_ids"] = torch.zeros(batched_tokenized["words_index"].size(), dtype=int)
        for i, tags in enumerate(batched_tags):
            for j, t in enumerate(tags):
                batched_tokenized["tag_ids"][i, j] = self.tag_vocab[t]

        batched_tokenized["gold_event"] = self.decoder.chart_from_tree(trees)

        return batched_tokenized

    def parse(self, trees: List[Tree]):
        self.eval()
        assert self.training == False, f"training is {self.training}"

        parsed_tree_list = []

        all_batched_words = [list(tree.leaves()) for tree in trees]
        for batch_start in range(0, len(all_batched_words), self.hparam.batch_size):

            batch_end = batch_start + self.hparam.batch_size
            batched_words = all_batched_words[batch_start : batch_end]
            batched_tags = [list(tree.pos()) for tree in trees[batch_start : batch_end]]
            batched_langs = [tree.lang for tree in trees[batch_start : batch_end]]

            batched_tokenized = self.tokenizer(batched_words)
            len_list = batched_tokenized["word_attention_mask"].sum(-1).tolist()
            batched_tokenized["langs"] = batched_langs

            if self.use_tag:
                batched_tokenized["tag_ids"] = torch.zeros(batched_tokenized["words_index"].size(), dtype=int)
                for i, tags in enumerate(batched_tags):
                    for j, t in enumerate(tags):
                        batched_tokenized["tag_ids"][i, j] = self.tag_vocab[t]

            with torch.no_grad():
                label_logits, tag_logits = self.forward(batched_tokenized)

                # TT 根节点的* 强行赋极小值
                label_logits[torch.arange(label_logits.size(0)), 0, torch.tensor(len_list) - 1, 0] -= 1.e9

                parsed_trees = self.decoder.tree_from_chart(label_logits, len_list, batched_words, batched_tags)
                parsed_tree_list += parsed_trees

        return parsed_tree_list

    def parse_1by1(self, trees: List[Tree]):
        self.eval()
        assert self.training == False, f"training is {self.training}"

        logits_list, parserd_tree_list = [], []
        for tree in tqdm(trees):
            batched_words = [list(tree.leaves())]
            batched_tags = [list(tree.pos())]
            assert len(batched_words) == len(batched_tags)
            batched_tokenized = self.tokenizer(batched_words)
            len_list = batched_tokenized["word_attention_mask"].sum(-1).tolist()

            batched_tokenized["tag_ids"] = torch.zeros(batched_tokenized["words_index"].size(), dtype=int)
            for i, tags in enumerate(batched_tags):
                for j, t in enumerate(tags):
                    batched_tokenized["tag_ids"][i, j] = self.tag_vocab[t]

            with torch.no_grad():
                label_logits, tag_logits = self.forward(batched_tokenized)
                # TT 根节点的* 强行赋极小值
                # logits[torch.arange(logits.size(0)), 0, torch.tensor(len_list) - 1, 0] -= 1.e9

                parserd_tree = self.decoder.tree_from_chart(label_logits, len_list, batched_words, batched_tags)
                logits_list += label_logits.tolist()
                parserd_tree_list += parserd_tree

        return parserd_tree_list, logits_list

    @classmethod
    def from_trained(cls, model_path):
        data = torch.load(model_path, map_location='cpu')
        hparam = Hparam(data["hparam"])
        state_dict = data["state_dict"]
        parser = cls(hparam)
        parser.load_state_dict(state_dict)
        return parser.to('cuda')
