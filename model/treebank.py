import sys, os
sys.path.append('..')
from config import iso_code
from model.data_unscape import ptb_unescape
from typing import List


class Tree:
    def __init__(self, label, children, left, right, lang=None) -> None:
        self.label = label
        self.word = None if not isinstance(children, str) else children
        self.children = children if not isinstance(children, str) else None
        self.left = left
        self.right = right
        self.is_leaf = False if self.word is None else True
        self.head = None
        self.lang = lang

    def leaves(self):
        if self.is_leaf:
            yield self.word
        else:
            for child in self.children:
                yield from child.leaves()

    def del_PUNCT(self, pucts: List[str]):
        if self.is_leaf:
            flag = True if self.label in pucts else False
        else:
            new_children = []
            for child in self.children:
                flag = child.del_PUNCT(pucts)
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

    def get_labeled_spans(self, strip_top=True):
        if self.is_leaf:
            res = []
        else:
            if strip_top and self.label in ['TOP', 'ROOT', 'root']:
                res = []
            else:
                res = [(self.left, self.right, self.label)]

            for child in self.children:
                res += child.get_labeled_spans(strip_top)
        return res

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)

    def add_lang(self, lang: str):
        self.lang = lang


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

def load_single_treebank(path, sort=False, binarize=True, too_long=False, del_top: bool=True, add_lang=False):
    trees = []
    for bracket_line in open(path, 'r', encoding='utf-8'):
        t = load_tree_from_str(bracket_line, del_top=del_top)
        if too_long and too_long(' '.join(list(t.leaves()))):
            continue

        # 二叉化
        if binarize:
            origin_t_str = t.linearize()
            t.binarize()
            trees.append(t)

            # Check the binarization and debinarization
            t_ = load_tree_from_str(t.linearize(), del_top=False)
            t_.debinarize()
            assert t_.linearize() == origin_t_str, "debinarization can not reverse to original tree"

        else:
            trees.append(t)
    
    if add_lang:
        lang = path.split('/')[-1].split('.')[0]
        for t in trees:
            t.add_lang(iso_code[lang])

    print(f"{len(trees)} in {path}")
    if sort:
        return sorted(trees, key=lambda x: len(list(x.leaves())))
    else:
        return trees


def load_tree_from_str(bracket_line: str, del_top: bool=True):
    assert bracket_line.count('(') == bracket_line.count(')')

    tokens = bracket_line.replace('(', ' ( ').replace(')',' ) ').split()
    idx, span_left_idx = 0, 0
    tree, _, _ = build_tree(tokens, idx, span_left_idx)

    if del_top:
        if len(tree.children) == 1 and tree.label in ['TOP', 'ROOT', 'root']:
            tree = tree.children[0]
        else:
            tree.label = "S"
    
    return tree


def load_multi_treebanks(folder, pattern, binarize=True, too_long=False, del_top: bool=True, add_lang=True):
    assert os.path.isdir(folder) and pattern is not None, "error in data path"
    langs = ['en', 'de', 'fr', 'he', 'ko', 'sv', 'zh', 'akk', 'kk', 'kmr', 'mr', 'sa', 'ta', 'yo']
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if pattern in f and f.split('.')[0] in langs]
    
    trees = []
    for path in paths:
        if os.path.exists(path):
            trees += load_single_treebank(path, sort=False, binarize=binarize, too_long=too_long, del_top=del_top, add_lang=add_lang)
    return trees


def load_treebank(path, pattern=None, sort=False, binarize=True, too_long=False, del_top: bool=True, add_lang=False):
    if os.path.isdir(path):
        all_trees = load_multi_treebanks(path, pattern, binarize=binarize, too_long=too_long, del_top=del_top, add_lang=add_lang)
        print(f"Total {len(all_trees)} {pattern} trees in {path}")
        return all_trees
    else:
        return load_single_treebank(path, sort=sort, binarize=binarize, too_long=too_long, del_top=del_top, add_lang=add_lang)


if __name__ == "__main__":
    path_ = [["/home/gpm/projects/CrossDomain-ConstituencyParsing-Origin/data/origin/WSJ/dev.txt", "data/en_dev_binary.txt"],
            # ["data/universal/en/En.u1.train", "data/universal/en_train_raw"],
            # ["data/universal/en/En.u1.test", "data/universal/en_test_raw"]
            ]
    
    for p_in, p_out in path_:
        with open(p_out, 'w', encoding="utf-8") as W:
            tree_list = load_single_treebank(p_in, sort=False, binarize=True, del_top=False)
            for tree in tree_list:
                W.write(tree.linearize() + '\n')
