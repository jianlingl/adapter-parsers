import os
import re
import subprocess
import math
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

from model.treebank import Tree, write_tree, load_single_treebank, load_tree_from_str


class FScore(object):
    def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match
        self.tagging_accuracy = tagging_accuracy

    def __str__(self):
        return (
            f"("
            f"Recall={self.recall:.2f}, "
            f"Precision={self.precision:.2f}, "
            f"FScore={self.fscore:.2f}, "
            f"CompleteMatch={self.complete_match:.2f}"
        ) + (
            f", TaggingAccuracy={self.tagging_accuracy:.2f})"
            if self.tagging_accuracy < 100
            else ")"
        )


class Metric():
    def __init__(self, span_correct, gold_span_total, pred_span_total) -> None:
        self.span_correct = span_correct
        self.gold_span_total = gold_span_total
        self.pred_span_total = pred_span_total
        self.score = self.get_R_P_F()

    def get_R_P_F(self):
        R = self.division(self.span_correct, self.gold_span_total)
        P = self.division(self.span_correct, self.pred_span_total)
        F = self.division((2 * R * P), (R + P))
        return (R * 100, P * 100, F * 100)

    def division(self, x, y):
        return x/y if y > 0 else 0.

    def to_string(self):
        return "R %.2f P %.2f F %.2f gold-correct-pred %d-%d-%d" % (*self.score, self.gold_span_total, self.span_correct,self.pred_span_total)


def eval_ups_lps(gold_trees: List[Tree], pred_trees: List[Tree], del_PUNCT=True, punct_tag=["PUNCT"], label_scores=False, label_set=[], len_scores=False, threshold=2, max_span_len=20):
    # g_trees = tree_del_PUNCT(gold_trees, punct_tag, del_PUNCT=del_PUNCT)
    # p_trees = tree_del_PUNCT(pred_trees, punct_tag, del_PUNCT=del_PUNCT)
    g_trees, p_trees = pair_tree_del_PUNCT(gold_trees, pred_trees, punct_tag, del_PUNCT=del_PUNCT)

    g_span_triples, p_span_triples = debinarize_and_get_span_triples(g_trees), debinarize_and_get_span_triples(p_trees)

    unlabel_span_correct, label_span_correct, gold_span_total, pred_span_total = statistic_compute(g_span_triples, p_span_triples, len_set=None, label=None)

    score_lps = Metric(label_span_correct, gold_span_total, pred_span_total)
    score_ups = Metric(unlabel_span_correct, gold_span_total, pred_span_total)
    print("lps \t", score_lps.to_string())
    print("ups \t", score_ups.to_string())

    if label_scores:
        _ = score_4_each_label(g_span_triples, p_span_triples, labels_set=label_set)
    if len_scores:
        _ = score_4_each_length(g_span_triples, p_span_triples, threshold=threshold, max_span_len=max_span_len)

    return score_ups.score, score_lps.score


def score_4_each_label(gold_triples_list, pred_triples_list, labels_set):
    label_scores_dict: Dict[str, Metric] = {}
    for label in labels_set:
        _, label_span_correct, gold_span_total, pred_span_total = statistic_compute(gold_triples_list, pred_triples_list, len_set=None, label=label)
        score_lps = Metric(label_span_correct, gold_span_total, pred_span_total)
        label_scores_dict[label] = score_lps.score
        print(label, ": ", score_lps.to_string())
    
    return label_scores_dict


def score_4_each_length(gold_triples_list, pred_triples_list, threshold=2, max_span_len=20):
    def get_len_4scores(max_span_len, threshold):
        if threshold <= 1:
            return [(len, len) for len in range(1, max_span_len + 1)]
        else:
            return [(len, len + threshold - 1) for len in range(1, max_span_len, threshold)]

    len_set_list = get_len_4scores(max_span_len, threshold)
    len_score_dict = {}
    for (s, e) in len_set_list:
        k = '【 ' + str(s) + ' - ' + str(e) + ' 】'
        unlabel_span_correct, label_span_correct, gold_span_total, pred_span_total = statistic_compute(gold_triples_list, pred_triples_list, len_set=(s, e), label=None)
        score_lps = Metric(label_span_correct, gold_span_total, pred_span_total)
        score_ups = Metric(unlabel_span_correct, gold_span_total, pred_span_total)
        
        len_score_dict[k] = score_lps.score + score_ups.score
        print(k, ": ", score_lps.to_string(), score_ups.to_string())

    return len_score_dict


def statistic_compute(gold_triples_list, pred_triples_list, len_set: Optional[Tuple] = None, label: Optional[str] = None):
    unlabel_span_correct, label_span_correct, gold_span_total, pred_span_total = 0, 0, 0, 0

    for g_triples, p_triples in zip(gold_triples_list, pred_triples_list):
        gold_dict, unlabel_Gspan_count_dict = get_set_dict_4snt(g_triples, len_set, label)
        pred_dict, unlabel_Pspan_count_dict = get_set_dict_4snt(p_triples, len_set, label)

        gold_span_total += len(g_triples)
        pred_span_total += len(p_triples)

        for k, v in gold_dict.items():
            for _ in range(v):
                if k in pred_dict and pred_dict[k] > 0:
                    pred_dict[k] -= 1
                    label_span_correct += 1

                if not label:
                    if k[:-1] in unlabel_Pspan_count_dict and unlabel_Pspan_count_dict[k[:-1]] > 0:
                        unlabel_Pspan_count_dict[k[:-1]] -= 1
                        unlabel_span_correct += 1

    return unlabel_span_correct, label_span_correct, gold_span_total, pred_span_total


def get_set_dict_4snt(span_triples, len_set: Optional[Tuple] = None, label: Optional[str] = None) -> List[Dict[Any, int]]:
    # 针对特定标签计算LPS的时候，只返回该标签对应的短语集合 span_set
    # 针对特定长度或者直接计算LPS，UPS的时候，返回有标签的短语集合 span_set 和无标签计数字典 unlabel_span_count
    span_dict, unlabel_span_count = dict(), dict()
    for s, e, l in span_triples:
        if len_set:
            if len_set[0] <= (e - s) <= len_set[1]:
                span_dict[(s, e, l)] = span_dict.get((s, e, l), 0) + 1
                unlabel_span_count[(s, e)] = unlabel_span_count.get((s, e), 0) + 1
        if label:
            if l == label:
                span_dict[(s, e, l)] = span_dict.get((s, e, l), 0) + 1
        if not len_set and not label:
            span_dict[(s, e, l)] = span_dict.get((s, e, l), 0) + 1
            unlabel_span_count[(s, e)] = unlabel_span_count.get((s, e), 0) + 1

    return span_dict, unlabel_span_count


def debinarize_and_get_span_triples(trees: List[Tree], debinarize=True):
    span_triples_list = []
    for tree in trees:
        if debinarize:
            tree.debinarize()
        span_triples_list.append(list(tree.get_labeled_spans()))
    return span_triples_list


def evalb(evalb_dir, gold_trees, predicted_trees, folder=None):
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)

    if os.path.exists(evalb_program_path):
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)

    observe_res = True

    if not observe_res:
        temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
        gold_path = os.path.join(temp_dir.name, "gold.txt")
        predicted_path = os.path.join(temp_dir.name, "predicted.txt")
        output_path = os.path.join(temp_dir.name, "output.txt")

    else:
        temp_dir = "log/evalb/"
        if folder is not None:
            temp_dir += folder
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        gold_path = os.path.join(temp_dir, "gold.txt")
        predicted_path = os.path.join(temp_dir, "predicted.txt")
        output_path = os.path.join(temp_dir, "output.txt")


    write_tree(gold_trees, gold_path)
    write_tree(predicted_trees, predicted_path)
    
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
            match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.complete_match = float(match.group(1))
            match = re.match(r"Tagging accuracy\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.tagging_accuracy = float(match.group(1))
                break

    success = (
        not math.isnan(fscore.fscore) or fscore.recall == 0.0 or fscore.precision == 0.0
    )

    if not observe_res:
        if success:
            temp_dir.cleanup()
        else:
            print("Error reading EVALB results.")
            print("Gold path: {}".format(gold_path))
            print("Predicted path: {}".format(predicted_path))
            print("Output path: {}".format(output_path))        

    print("fscore {} ".format(fscore)) 
    return fscore


def tree_del_PUNCT(trees: List[Tree], punct_tags: List[str], del_PUNCT=True):
    if del_PUNCT:
        trees_no_punct = []
        for t in trees:
            # # del on the tree
            tree = load_tree_from_str(t.linearize(), del_top=False)
            tree.del_PUNCT(punct_tags)
            tree = load_tree_from_str(tree.linearize(), del_top=False)
            trees_no_punct.append(tree)
        return trees_no_punct
    else:
        return trees

def pair_tree_del_PUNCT(gold_trees: List[Tree], pred_trees: List[Tree], punct_tags: List[str], del_PUNCT=True):
    if del_PUNCT:
        gold_trees_no_punct, pred_trees_no_punct = [], []
        for g, p in zip(gold_trees, pred_trees):
            # # del on the tree
            gold_tree = load_tree_from_str(g.linearize(), del_top=False)
            pred_tree = load_tree_from_str(p.linearize(), del_top=False)

            gold_tree.del_PUNCT(punct_tags)
            pred_tree.del_PUNCT(punct_tags)

            try:
                assert list(gold_tree.leaves()) == list(pred_tree.leaves()), "the leaves of gold and pred are not equal"
                if list(gold_tree.leaves()) == [] or list(pred_tree.leaves()) == []:
                    pass
                gold_tree = load_tree_from_str(gold_tree.linearize(), del_top=False)
                gold_trees_no_punct.append(gold_tree)
                pred_tree = load_tree_from_str(pred_tree.linearize(), del_top=False)
                pred_trees_no_punct.append(pred_tree)

            except:
                print(g.linearize())

        return gold_trees_no_punct, pred_trees_no_punct
    else:
        return gold_trees, pred_trees


def statistic_label_proportions_4testset(test_treebank_path):
    test_bank = load_single_treebank(test_treebank_path, sort=False, binarize=False, prob_join_label=False)
    test_trees = tree_del_PUNCT(test_bank, punct_tags=["PUNCT"],del_PUNCT=True)
    span_triples_list = debinarize_and_get_span_triples(test_trees, debinarize=False)
    label_num_dict = {}
    for snt_span in span_triples_list:
        for _, _, ll in snt_span:
            label_num_dict[ll] = label_num_dict.get(ll, 0) + 1
    
    labels_count = sum(label_num_dict.values())
    # assert labels_count == len(span_triples_list)

    for label, count in label_num_dict.items():
        label_num_dict[label] = round(count/labels_count,3)
    label_num_dict = sorted(label_num_dict.items(), key=lambda item: item[1], reverse=True)
    return label_num_dict


def phrase_length_statistic(span_triples_list):
    max_span_len, span_len_accum, span_count = 0, 0, 0
    for triples in span_triples_list:
        span_count += len(triples)
        span_len_list = [e-s for s, e, _ in triples]
        max_span_len = max(span_len_list + [max_span_len])
        span_len_accum += sum(span_len_list)
    
    print("sum of phrases: ", span_count)
    avg_len = span_len_accum /sum(span_count)
    return max_span_len, avg_len


def sent_len_statistic(treebank: List[Tree]):
    snt_len_list = []
    for tree in tqdm(treebank):
        snt_len_list.append(len(list(tree.leaves())))

    print("avg", sum(snt_len_list)/len(snt_len_list))
    print("max", max(snt_len_list))

    snt_len_list = sorted(snt_len_list, reverse=True)
    print(snt_len_list)


def toks_count_statistic(treebank: List[Tree]):
    tok_count = 0
    for tree in treebank:
        tok_count += len(list(tree.leaves()))

    print("all toks conunt: ", tok_count)
    print('avg sent toks len: ', tok_count / len(treebank))


if __name__ == "__main__":
    # treebank = load_treebank("data/universal/fr/Fr.u1.train", sort=False, binarize=False, prob_join_label=False)
    # sent_len_statistic(treebank)

    # ground = load_treebank("data/universal/zh_from_groundtruth24", sort=False, binarize=True)
    # predicted = load_treebank("data/universal/zh_from_chatGPT", sort=False, binarize=True)
    # fscore = evalb("/home/ljl/0_max_margin/EVALB/", ground, predicted)
    folder = '/home/ljl/parse_use_GPT/test5/'
    for path in os.listdir(folder):
        label_statis = statistic_label_proportions_4testset(folder+path)
        print(path)
        print(label_statis)
