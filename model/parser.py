# 编码 + 解码 + 损失函数
from selectors import SelectorKey
from typing import List
from torch import nn
import torch.nn.functional as F
import torch
import torch_struct
from transformers import XLMRobertaModel

from model.pgn_adapter import AdapterXLMRobertaModel
from model.tokenization import Tokenizer, Retokenizer
from model.encoder import ConcatTag, ConcatPosition, encoderStackLayer, PartitionedTransformerEncoderLayer, FeatureDropout
from model.decoder import ChartDecoder
# from config import Hparam
from config_partitioned_transformer import Hparam
from model.treebank import Tree

class Parser(nn.Module):
    def __init__(self, hparam: Hparam, cuda_device) -> None:
        super().__init__()
        # 预训练
        self.hparam = hparam
        self.tokenizer = Retokenizer(hparam.pretrain_model_path)
        self.pretrain_model = XLMRobertaModel.from_pretrained(hparam.pretrain_model_path)
        # self.pretrain_model = AdapterXLMRobertaModel(hparam.pretrain_model_path)
        
        # 词表示
        self.tag_vocab = hparam.tag_vocab
        self.use_tag, self.use_position = hparam.use_tag, hparam.use_position
        self.word_feature_dropout = FeatureDropout(hparam.morpho_emb_dropout)

        if hparam.use_tag:
            self.concat_tag = ConcatTag(len(self.tag_vocab), hparam.d_tag)
            self.c_project = nn.Linear(hparam.d_pretrained+hparam.d_tag, hparam.d_word)
        else:
            self.c_project = nn.Linear(hparam.d_pretrained, hparam.d_word)

        if hparam.use_position:
            self.concat_position = ConcatPosition(hparam.max_snt_len, hparam.d_word)        
        
        self.encoder = encoderStackLayer(PartitionedTransformerEncoderLayer(
            hparam.d_model, 
            n_head=hparam.num_heads, 
            d_qkv=hparam.d_qkv, 
            d_ff=hparam.d_ff, 
            ff_dropout=hparam.relu_dropout, 
            residual_dropout=hparam.residual_dropout, attention_dropout=hparam.attention_dropout, 
        ), num_layers=hparam.num_encoder_layer)

        self.w_linear_after_encoder = nn.Linear(hparam.d_model, hparam.d_word)

        self.span_linear = nn.Linear(hparam.d_word*2, hparam.d_word, bias=True)

        # self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hparam.d_word, hparam.num_heads, dim_feedforward=1024, batch_first=True), 2)

        self.decoder = ChartDecoder(hparam.label_vocab)
        
        
        # 打分函数
        self.score_label = nn.Sequential(
            nn.Linear(hparam.d_word, hparam.d_label_hidden),
            nn.LayerNorm(hparam.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparam.d_label_hidden, len(hparam.label_vocab)-1),
        )

        # todo zls 
        # self.score_label 后面加sigmoid ->稳定, logits = logits + Haming_augment必须乘个概率

        self.device = cuda_device

    def forward(self, tokenized):
        input_ids, attention_mask, words_index, tag_ids, word_attention_mask = tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device), tokenized["words_index"].to(self.device), tokenized["tag_ids"].to(self.device), tokenized["word_attention_mask"].to(self.device)

        # # span表示 # 在头尾多加一维
        # words_index = torch.cat([words_index, torch.zeros(words_index.size()[0], 1, dtype=torch.long).to(self.device)], dim=-1)
        # words_index[torch.arange(word_attention_mask.size()[0]), word_attention_mask.sum(-1)] = word_attention_mask.sum(-1)
        # words_index = torch.cat([torch.zeros(words_index.size()[0], 1, dtype=torch.long).to(self.device), words_index], dim=-1)
        # word_attention_mask = torch.cat([torch.ones(word_attention_mask.size()[0], 2, dtype=torch.bool).to(self.device), word_attention_mask], dim=-1)
        # tag_ids = torch.cat([torch.zeros(tag_ids.size()[0], 1, dtype=int).to(self.device), tag_ids], dim=-1)
        # tag_ids = torch.cat([tag_ids, torch.zeros(tag_ids.size()[0], 1, dtype=int).to(self.device)], dim=-1)

        # 预训练输出
        features = self.pretrain_model(input_ids, attention_mask).last_hidden_state

        # # span表示 # 奇偶位 -> 前向和后向的表示
        # features = torch.cat([
        #         features[..., 0::2],
        #         features[..., 1::2],
        #     ], dim=-1,
        # )

        # 取头表示词
        features_out = []
        for i, w_index in enumerate(words_index):
            features_out.append(features[i, w_index, :].unsqueeze(0))
        word_repre = torch.cat(features_out, dim=0)

        # 获取更全面词表示
        if self.use_tag:
            word_repre = self.concat_tag(word_repre, tag_ids)
        word_repre = self.c_project(self.word_feature_dropout(word_repre))

        if self.use_position:
            word_repre = self.concat_position(word_repre)

        # 编码器优化词表示
        # word_repre = self.encoder(word_repre, src_key_padding_mask=~word_attention_mask)
        word_repre = self.encoder(word_repre, mask=word_attention_mask)
        word_repre = self.w_linear_after_encoder(word_repre)

        span_features = self.span_features(word_repre)

        # 给span打分
        logits_4labels = self.score_label(span_features)
        logits_4stars = logits_4labels.new_zeros(logits_4labels.size()[:3] + (1, ))
        logits = torch.cat((logits_4stars, logits_4labels), dim=-1)

        return logits

    def compute_loss(self, tokenized):
        logits = self.forward(tokenized)
        gold_event, len_list = tokenized["gold_event"].to(self.device), tokenized["word_attention_mask"].sum(-1).tolist()

        Haming_augment = 1. - gold_event
        Haming_augment[torch.arange(logits.size()[0]), 0, torch.tensor(len_list) - 1, 0] -= 1e9  # 规避根节点标签为*,通过给Haming_augment赋值一个极小值，使得其空标签的得分特别低，永远不可能预测空标签

        logits = logits + Haming_augment
        # todo * r 概率

        pred_event, _ = self.decoder.CKY(logits.data.cpu().numpy(), len_list)
        pred_score = (torch.tensor(pred_event).to(self.device) * logits).sum([1,2,3])
        gold_score = (gold_event * logits).sum([1,2,3])

        # dist = torch_struct.TreeCRF(logits, lengths=torch.LongTensor(len_list).to(self.device))
        # pred_score = dist.max
        
        loss = F.relu(pred_score - gold_score).mean()
        return loss

    def span_features(self, features_out):
        words_num = features_out.size()[1]
        features_i = features_out.unsqueeze(2).expand(-1, -1, words_num, -1)
        features_j = features_out.unsqueeze(1).expand(-1, words_num, -1, -1)
        features = self.span_linear(torch.cat((features_i, features_j), dim=-1))
        return features

        # fencepost_annotations = torch.cat([
        #         features_out[:, :-1, :768//2],
        #         features_out[:, 1:, 768//2:],
        #     ], dim=-1,
        # )

        # # Note that the bias added to the final layer norm is useless because
        # # this subtraction gets rid of it
        # span_features = (
        #     torch.unsqueeze(fencepost_annotations, 1) - torch.unsqueeze(fencepost_annotations, 2)
        # )[:, :-1, 1:]  # 分别在第一维和第二维加一个维度  8x1xNx1024-8xNx1x1024
        # return span_features

    def encode_and_collate(self, trees: List[Tree]):
        batched_words = [list(tree.leaves()) for tree in trees]
        batched_tags = [list(tree.pos()) for tree in trees]

        batched_tokenized = self.tokenizer(batched_words)

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

            batched_tokenized = self.tokenizer(batched_words)
            len_list = batched_tokenized["word_attention_mask"].sum(-1).tolist()

            batched_tokenized["tag_ids"] = torch.zeros(batched_tokenized["words_index"].size(), dtype=int)
            for i, tags in enumerate(batched_tags):
                for j, t in enumerate(tags):
                    batched_tokenized["tag_ids"][i, j] = self.tag_vocab[t]

            with torch.no_grad():
                logits = self.forward(batched_tokenized)
                parsed_trees = self.decoder.tree_from_chart(logits, len_list, batched_words, batched_tags)
                parsed_tree_list += parsed_trees

        return parsed_tree_list

    @classmethod
    def from_trained(cls, model_path, cuda_device):
        data = torch.load(model_path)
        hparam = Hparam(data["hparam"])
        state_dict = data["state_dict"]
        parser = cls(hparam, cuda_device)
        parser.load_state_dict(state_dict)
        return parser.to(cuda_device)
