import json
import torch
from typing import List
from model.treebank import load_treebank, Tree
from model.parser import Parser
from model.eval import evalb, eval_ups_lps

device = 3
label_vocab = {'*': 0, 'AJP': 1, 'AJP::AVP': 2, 'AJP::NP': 3, 'AVP': 4, 'AVP::AJP': 5, 'CONJP': 6, 'NP': 7, 'NP::AJP': 8, 'NP::AVP': 9, 'NP::PP': 10, 'NP::S': 11, 'NP::S::VP': 12, 'NP::X': 13, 'PP': 14, 'PP::NP': 15, 'S': 16, 'S::AJP': 17, 'S::AVP': 18, 'S::NP': 19, 'S::PP': 20, 'S::VP': 21, 'S::VP::AVP': 22, 'S::X': 23, 'VP': 24, 'VP::AVP': 25, 'VP::NP': 26, 'VP::PP': 27, 'VP::S': 28, 'VP::S::AJP': 29, 'VP::S::VP': 30, 'X': 31, 'X::AJP': 32, 'X::AVP': 33, 'X::NP': 34, 'X::PP': 35, 'X::S': 36, 'X::VP': 37}
label_set = {'NP', 'VP', 'AJP', 'AVP', 'PP', 'S','CONJP', 'COP', 'X'}

def get_upper_triangle_scores(bank_path, model_path, scores_path):
    gold_trees = load_treebank(bank_path, binarize=True, del_top=True, prob_join_label=False)

    parser = Parser.from_trained(model_path, device)
    assert label_vocab == parser.hparam.label_vocab

    # test first
    pred_trees, logits_list = parser.parse_1by1(trees=gold_trees)
    fscore = evalb("EVALB", gold_trees, pred_trees)
    fscore_lps, fscore_ups = eval_ups_lps(gold_trees, pred_trees, label_scores=True, label_set=label_set, len_scores=True)

    # write trees and logits
    write_tensor(scores_path, gold_trees, logits_list)

def write_tensor(path, g_trees: List[Tree], scores):
    assert len(g_trees) == len(scores)
    with open(path, 'w', encoding='utf-8') as F:
        for g_t, score in zip(g_trees, scores):
            json.dump({"g_tree": g_t.linearize(), "score": score}, F)
            F.write('\n')

def load_tensor(path):
    scores_list = []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        scores_list.append(json.loads(line))
    return scores_list

if __name__ == "__main__":
    model_path = "log/models/parser_4en_score_finetune_predict*.pt"
    trainbank_path = "/home/ljl/0_max_margin/data/universal/en/En.u1.train"
    scores_path = "/home/ljl/0_max_margin/data/scores/score_finetune_predict*_newtrans.scores"
    get_upper_triangle_scores(trainbank_path, model_path, scores_path)

    # load_tensor(scores_path)
