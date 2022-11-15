import math
from model.data_unscape import ptb_unescape
from typing import List


class Tree:
    def __init__(self, label, children, left, right, prob_join_label=False) -> None:
        if prob_join_label:
            prob, label = label.split('|||')
            self.prob, self.label = float(prob), label
        else:
            self.prob, self.label = None, label
        self.word = None if not isinstance(children, str) else children
        self.children = children if not isinstance(children, str) else None
        self.left = left
        self.right = right
        self.is_leaf = False if self.word is None else True

    def leaves(self):
        if self.is_leaf:
            yield self.word
        else:
            for child in self.children:
                yield from child.leaves()

    def del_PUNCT(self, puct: str):
        if self.is_leaf:
            flag = True if self.label == puct else False
        else:
            new_children = []
            for child in self.children:
                flag = child.del_PUNCT(puct)
                if not flag:
                    new_children.append(child)
            self.children = new_children

            flag = True if len(self.children) == 0 else False

        return flag

    def pos(self):
        if self.is_leaf:
            yield self.label
        else:
            for child in self.children:
                yield from child.pos()

    def binarize(self):
        if not self.is_leaf:
            while len(self.children) == 1 and not self.children[0].is_leaf:
                self.label += '::' + self.children[0].label
                self.children = self.children[0].children

            if len(self.children) > 2:
                left_child = self.children[0]
                right_child = Tree('*', self.children[1:], self.children[1].left, self.children[-1].right)
                self.children = [left_child, right_child]

            # if len(self.children) == 2:
            #     left_child, right_child = self.children[0], self.children[1]
            #     if left_child.is_leaf:
            #         left_child = Tree('*', [left_child], left_child.left, left_child.right)
            #     if right_child.is_leaf:
            #         right_child = Tree('*', [right_child], right_child.left, right_child.right)
            #     self.children = [left_child, right_child]

            for child in self.children:
                child.binarize()

    def debinarize(self):
        if not self.is_leaf:
            while '::' in self.label:
                label_list = self.label.split('::')
                label_this, label_tail = '::'.join(label_list[:-1]), label_list[-1]
                self.label = label_this
                self.children = [Tree(label_tail, self.children, self.left, self.right)]

            for child in self.children:
                child.debinarize()

            child_list = []
            for child in self.children:
                if '*' in child.label:
                    child_list.extend(child.children)
                else:
                    child_list.append(child)
            self.children = child_list

    def span_labels(self):
        if self.is_leaf:
            res = []
        else:
            res = [self.label]
            for child in self.children:
                res += child.span_labels()
        return res

    def get_labeled_spans0(self, strip_top=True):
        if not self.is_leaf:
            if strip_top:
                if self.label != 'TOP':
                    yield (self.left, self.right, self.label)
            else:
                yield (self.left, self.right, self.label)

        if not self.is_leaf:
            for child in self.children:
                yield from child.get_labeled_spans0(strip_top)

    def get_labeled_spans(self, strip_top=True, prob=False):
        if self.is_leaf:
            res = []
        else:
            if strip_top and self.label == 'TOP':
                res = []
                for child in self.children:
                    res += child.get_labeled_spans(strip_top, prob)
            else:
                if prob:
                    res = [(self.left, self.right, self.label, self.prob)]
                else:
                    res = [(self.left, self.right, self.label)]
                for child in self.children:
                    res += child.get_labeled_spans(strip_top, prob)
        return res

    def cal_span_prob(self):
        if not self.is_leaf:
            for child in self.children:
                child.cal_span_prob()

            prob = 1.0
            for child in self.children:
                assert child.prob is not None and child.prob != -1, "the child prob has not been updated"
                prob *= child.prob
            self.prob = math.pow(prob, 1/len(self.children))

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)


def build_label_vocab(trees: List[Tree]):
    labels = []
    for tree in trees:
        labels += tree.span_labels()
    label_set = sorted(set(labels))
    label_set.remove('*')
    label_set = ['*'] + label_set
    return {label: i for i, label in enumerate(label_set)}


def build_tag_vocab(trees: List[Tree]):
    tags = []
    for tree in trees:
        tags += list(tree.pos())
    tag_set = [''] + sorted(set(tags))
    return {tag: i for i, tag in enumerate(tag_set)}


def build_tree(tokens, idx, span_left_idx):
    idx += 1
    label = tokens[idx]
    idx += 1
    assert idx < len(tokens)

    # 若直接是）说明当前短语只有标签
    assert tokens[idx] != ')', "empty label here"
    
    if tokens[idx] == '(':
        children = []
        span_right_idx = span_left_idx
        while idx < len(tokens) and tokens[idx] == '(':
            child, idx, span_right_idx = build_tree(tokens, idx, span_right_idx)
            children.append(child)
        tree = Tree(label, children, span_left_idx, span_right_idx)
    
    else:
        word = tokens[idx]
        # 对特殊字符进行处理
        word = ptb_unescape([word])[0] 
        assert len(word) != 0, "after ptb none return!"
        idx += 1
        span_right_idx = span_left_idx + 1
        tree = Tree(label, word, span_left_idx, span_right_idx)
        assert tree.is_leaf
    
    assert tokens[idx] == ')'
    return tree, idx+1, span_right_idx


def write_tree(tree_list: List[Tree], path):
    with open(path, 'w', encoding='utf-8') as W:
        for tree in tree_list:
            tree.debinarize()
            W.write(Tree("TOP", [tree], tree.left, tree.right).linearize() + '\n')


def load_treebank(path, binarize=True, max_snt_len: int=150, top_exist: bool=True):
    trees = []
    for bracket_line in open(path, 'r', encoding='utf-8'):
        t = load_tree_from_str(bracket_line, top_exist=top_exist)
        if len(list(t.leaves())) >= max_snt_len:
            continue

        # 二叉化
        if binarize:
            origin_t_str = t.linearize()
            t.binarize()
            trees.append(t)

            # Check the binarization and debinarization
            t_ = load_tree_from_str(t.linearize(), top_exist=False)
            t_.debinarize()
            assert t_.linearize() == origin_t_str, "debinarization can not reverse to original tree"

        else:
            trees.append(t)
    
    print(path)
    print(len(trees))
    return trees


def load_tree_from_str(bracket_line: str, top_exist: bool=True):
    assert bracket_line.count('(') == bracket_line.count(')')

    tokens = bracket_line.replace('(', ' ( ').replace(')',' ) ').split()
    idx, span_left_idx = 0, 0
    tree, _, _ = build_tree(tokens, idx, span_left_idx)

    if top_exist:
        # 处理根节点TOP
        if len(tree.children) == 1 and tree.label == 'TOP':
            tree = tree.children[0]
        else:
            tree.label = "S"
    
    return tree


# def guess_lan(path):
    if 'zh' in path or 'zh_from' in path or 'Zh' in path:
        lan = 'zh'
    elif 'wsj' in path or 'en_from' in path or 'En' in path:
        lan = 'en'
    elif 'german' in path or 'de_from' in path or 'De' in path:
        lan = 'de'
    elif 'french' in path or 'fr_from' in path or 'Fr' in path:
        lan = 'fr'
    elif 'korean' in path or 'ko_from' in path or 'Ko' in path:
        lan = 'ko'
    elif 'he' in path or 'he_from' in path or 'He' in path:
        lan = 'he'
    elif 'hu' in path or 'hu_from' in path or 'Hu' in path:
        lan = 'hu'
    elif 'ja' in path or 'ja_from' in path or 'Ja' in path:
        lan = 'ja'
    elif 'sv' in path or 'sv_from' in path or 'Sv' in path:
        lan = 'sv'
    else:
        lan = 'None'
    return lan


# if __name__ == "__main__":
#     path_ = [["data/universal/en/En.u1.dev", "data/universal/en_dev_raw"],
#             ["data/universal/en/En.u1.train", "data/universal/en_train_raw"],
#             ["data/universal/en/En.u1.test", "data/universal/en_test_raw"]]
    
#     for p_in, p_out in path_:
#         with open(p_out, 'w', encoding="utf-8") as W:
#             tree_list = load_treebank(p_in, binarize=False, max_snt_len=1000)
#             for tree in tree_list:
#                 W.write(' '.join(list(tree.leaves())) + '\n')
