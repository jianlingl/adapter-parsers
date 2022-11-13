from operator import le
from re import L
import torch
import numpy as np
from  typing import List

from model.treebank import Tree


class ChartDecoder():

    def __init__(self, label_vocab, force_root_constituent=True) -> None:
        self.label_vocab = label_vocab
        self.label_from_index = {i: label for label, i in label_vocab.items()}
        self.force_root_constituent = force_root_constituent
        self.num_labels = len(label_vocab)

    def chart_from_tree(self, batch_trees: List[Tree]): # 这里是二叉化的树 onehot + pad 注：b*l*l*num_labels
        num_words = max([len(list(tree.leaves())) for tree in batch_trees])
        batch_size = len(batch_trees)
        chart = torch.full((batch_size, num_words, num_words, self.num_labels), 0, dtype=torch.long)

        for i, tree in enumerate(batch_trees):
            spans = tree.get_labeled_spans(strip_top=True)  # 去掉TOP
            for start, end, label in spans:
                if label in self.label_vocab:
                    chart[i, start, end-1, self.label_vocab[label]] = 1
                else:
                    assert False
        return chart

    def build_tree(self, tree_split_table, best_label_table, leaf_list, tag_list, left, right):
        k = tree_split_table[left, right]
        label = self.label_from_index[best_label_table[left, right]]

        if k != -1:
            treeL = self.build_tree(tree_split_table, best_label_table, leaf_list, tag_list, left, k)
            treeR = self.build_tree(tree_split_table, best_label_table, leaf_list, tag_list, k+1, right)
            tree = Tree(label, [treeL, treeR], left, right + 1)
        else:
            leaf_tag = Tree(tag_list[left], leaf_list[left], left, right + 1)
            tree = Tree(label, [leaf_tag], left, right + 1)
        return tree
    
    def tree_from_chart(self, logits :torch.Tensor, len_list: List[int], leaves: List[List[str]], tags: List[List[str]]):
        pred_events, tree_split_tables = self.CKY(logits.data.cpu().numpy(), len_list)

        best_label_tables = np.argmax(pred_events, axis=-1)
        # 3 遍历最大得分对应的切分 tree_split_tables 和 最好标签对应的 best_label_tables 递归获取树结构 pred_tree : Tree
        pred_tree_list = []
        for tree_split_table, best_label_table, snt_len, leaf_list, tag_list in zip(tree_split_tables, best_label_tables, len_list, leaves, tags):
            tree = self.build_tree(tree_split_table[:snt_len, :snt_len], best_label_table[:snt_len, :snt_len], leaf_list, tag_list, left=0, right=snt_len-1)
            pred_tree_list.append(tree)
        return pred_tree_list


    def CKY(self, logits, len_list):
        # 1 计算每个span对应位置的最大标签得分，存入 best_label_score_tables  (b, len, len, 1) 和 best_label_index_tables (b, len, len, 1)
        best_label_score_tables = np.max(logits, axis=-1)
        best_label_index_tables = np.argmax(logits, axis=-1)
        # 2 计算每个span对应位置的最大总得分，存入 tree_score_tables (b, len, len, 1) 并且保存最大得分对应的切分 tree_split_tables (b, len, len, 1)
        tree_score_tables = np.zeros(best_label_score_tables.shape)
        tree_split_tables = np.full(best_label_score_tables.shape, -1)
        pred_events = np.zeros(logits.shape)
        
        for B, snt_len in enumerate(len_list):
            self.single_CKY(tree_score_tables[B], tree_split_tables[B], best_label_score_tables[B], snt_len)
            self.find_tree(tree_split_tables[B], best_label_index_tables[B], pred_events[B], 0, snt_len-1)          

        return pred_events, tree_split_tables

    def single_CKY(self, tree_score, tree_split, best_label_score, snt_len):
        tree_score[np.arange(snt_len), np.arange(snt_len)] = np.diagonal(best_label_score, axis1=0)[:snt_len]

        for i in range(1, snt_len):
            for j in range(i, snt_len):
                span_start, span_end = j-i, j
                max_score, best_k = self.get_best_k_score(tree_score, span_start, span_end)
                tree_score[span_start, span_end] = max_score + best_label_score[span_start, span_end]
                tree_split[span_start, span_end] = best_k

    def get_best_k_score(self, best_label_score, i, j):
        max_score, best_k = best_label_score[i][i] + best_label_score[i+1][j], i
        for k in range(i, j):
            l_score = best_label_score[i][k] + best_label_score[k+1][j]
            if l_score > max_score:
                max_score, best_k = l_score, k
        return max_score, best_k

    def find_tree(self, tree_split, best_label_index, pred_event, span_s, span_e):
        pred_event[span_s, span_e, best_label_index[span_s, span_e]] = 1.

        k = tree_split[span_s, span_e]
        if k == -1:
            return
        else:
            self.find_tree(tree_split, best_label_index, pred_event, span_s, k)
            self.find_tree(tree_split, best_label_index, pred_event, k+1, span_e)
