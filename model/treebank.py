from copy import deepcopy
from model.data_unscape import ptb_unescape
from typing import List


class Tree:
    def __init__(self, label, children, left, right) -> None:
        
        self.label = label
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
        flag = False

        if self.is_leaf:
            if self.label == puct:
                flag = True
            else:
                flag = False
        else:
            new_children = []
            for child in self.children:
                flag = child.del_PUNCT(puct)
                if not flag:
                    new_children.append(child)
            self.children = new_children

            if len(self.children) == 0:
                flag = True
            else:
                flag = False

        return flag

    def pos(self):
        if self.is_leaf:
            yield self.label
        else:
            for child in self.children:
                yield from child.pos()
    
    def binarization(self):
        if not self.is_leaf:
            if len(self.children) == 1 and not self.children[0].is_leaf:
                self.label = self.label + '::' + self.children[0].label
                self.children = self.children[0].children
            
            if len(self.children) > 2:
                multi_child = self.children[1:]

                self.children = self.children[:1]
                new_treelet = Tree('*', multi_child, multi_child[0].left, multi_child[-1].right)
                self.children.append(new_treelet)

            for child in self.children:
                child.binarization()

    def debinarization(self):
        if not self.is_leaf:
            if '::' in self.label:
                label_list = self.label.split('::')
                label_this_node = label_list[0]
                label_next_node = '::'.join(label_list[1:])

                self.label = label_this_node
                extra_child = Tree(label_next_node, self.children, self.left, self.right)
                self.children = [extra_child]

            # 深度优先后续遍历
            for child in self.children:
                child.debinarization()

            new_children = []
            for child in self.children:
                if child.label == '*':
                    new_children.extend(child.children)
                else:
                    new_children.append(child)
            self.children = new_children
    
    def span_labels(self):
        if not self.is_leaf:
            yield self.label
        if not self.is_leaf:
            for child in self.children:
                yield from child.span_labels()
    
    def get_labeled_spans(self, strip_top=True):
        if not self.is_leaf:
            if strip_top:
                if self.label != 'TOP':
                    yield (self.left, self.right, self.label)
            else:
                yield (self.left, self.right, self.label)

        if not self.is_leaf:
            for child in self.children:
                yield from child.get_labeled_spans()
        
    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)


def build_label_vocab(trees: List[Tree]):
    labels = []
    for tree in trees:
        labels += list(tree.span_labels())
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
            tree.debinarization()
            W.write(Tree("TOP", [tree], tree.left, tree.right).linearize() + '\n')


def load_treebank(path, binarize=True, max_snt_len: int=150, top_exist: bool=True):
    trees = []
    for bracket_line in open(path, 'r', encoding='utf-8'):
        tree = load_tree_from_str(bracket_line, top_exist=top_exist)
        if len(list(tree.leaves())) >= max_snt_len:
            continue

        # 二叉化
        if binarize:
            binarizedTree = deepcopy(tree)
            binarizedTree.binarization()
            debinarizedTree = deepcopy(binarizedTree)
            debinarizedTree.debinarization()
            assert debinarizedTree.linearize() == tree.linearize(), "debinarization can not reverse to original tree"
            trees.append(binarizedTree)
        else:
            trees.append(tree)
    
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
