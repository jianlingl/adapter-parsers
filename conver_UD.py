from model.treebank import load_treebank, Tree
from model.data_unscape import ptb_unescape
from typing import List


class DepDict:
    def __init__(self, dep_info) -> None:
        self.dep_info = dep_info
        self.head_dict: dict = {}
        self.convert_info2dict()
    
    def convert_info2dict(self) -> None:
        for id, info in self.dep_info.items():
            word, head, label = info['word'], info['head'], info['label']
            if not self.head_dict.get(head, False):
                self.head_dict[head] = [{'id': id, 'word': word, 'label': label}]
            else:
                self.head_dict[head].append({'id': id, 'word': word, 'label': label})
        
        assert sum([len(depinfo) for _, depinfo in self.head_dict.items()]) == len(self.dep_info), ""
    
    def edges(self):
        tmp = dict()
        for id, info in self.dep_info.items():
            tmp[id] = info['head']
        assert sorted(tmp) == [i for i in range(1, len(self.dep_info)+1)]
        return tmp

    def words(self) -> list:
        return [info['word'] for _, info in self.dep_info.items()]


def load_dep_dicts(dep_path):
    trees, snts = [], []
    info_dict, snt = {}, []
    for line in open(dep_path, 'r', encoding='utf-8'):
        if not line.startswith('#') and line.strip() != '':
            info = line.split('\t')
            id, word, coarstag, finetag, head, label = info[0], info[1], info[3], info[4], info[6], info[7]
            if id != '_' and coarstag != '_' and head != '_' and label != '_':
                info_dict[int(id)] = {'word': word, 'coarstag': coarstag, 'finetag': finetag, 'head': int(head), 'label': label}
                snt.append(word)
        elif len(info_dict) > 0:
            trees.append(DepDict(info_dict))
            snts.append(snt)
            info_dict, snt = dict(), []
        else:
            assert False, ""
    return trees

def find_head(self, grammar_head:dict, deps:dict):
    if self.is_leaf:
        self.head = self.right
    elif not self.is_leaf and len(self.children) == 1:
        assert self.children[0].is_leaf, ""
        self.head = self.right
    elif not self.is_leaf:

        for child in self.children:
            child.find_head(grammar_head, deps)

        # 每个短语都有一个head
        production = [kid.label for kid in self.children]
        gramm = f"{self.label}->{' '.join(production)}"
        # todo  stop here for those no exist in grammars
        
        head_idx = grammar_head.get(gramm, None)
        if head_idx is None: head_idx = assign_head(production)
        head_phrase = self.children[head_idx]
        self.head = self.children[head_idx].head

        assert self.head is not None, ""

        for i, kid in enumerate(self.children): 
            if kid != head_phrase:
                assert deps.get(kid.head, "no exist") == "no exist", ""
                deps[kid.head] = self.head
            else:
                assert i == head_idx, ""
    else:
        assert False

Tree.find_head = find_head

def load_grammars(path, grammars:dict) -> dict:
    with open(path, 'r', encoding='utf-8')as F:
        for line in F:
            gramm, cnt = line.strip().split(" : ")
            assert gramm.count('*') == 1, "grammar errors!"
            grammars[gramm] = grammars.get(gramm, 0) + int(cnt)
    return grammars


def process_grammars(grammars:dict) -> dict:
    all_cnt = sum(grammars.values())
    grammars = {gramm: round(cnt/all_cnt, 2) for gramm, cnt in grammars.items()}
    
    def rm_head(grammars: list):
        new_gramms = dict()
        duplicate_cnt = 0
        for gramm, prob in grammars:
            assert gramm.count('*') == 1, ""
            production = gramm.split('->')[1].split()
            for i, phrase in enumerate(production):
                if '*' in phrase:
                    head = i
                    break
            new_gramm = gramm.replace('*', '')
            
            if new_gramms.get(new_gramm, 'no exist') == 'no exist':
                new_gramms[new_gramm] = head
            else:
                duplicate_cnt += 1
                print(gramm, prob, head)

        return new_gramms, duplicate_cnt
    
    grammars_list = sorted(grammars.items(), key=lambda x: x[1], reverse=True)
    new_grammars, duplicate_cnt = rm_head(grammars_list)
    print(f"{round(duplicate_cnt/len(grammars_list), 2)} grammars are duplicate and are deleted")
    return new_grammars


def assign_head(production: list) -> int:
    # ('NP', 'VP', 'AJP', 'AVP', 'PP', 'S', 'CONJP', 'COP', 'X')
    # ('ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X', 'NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ')
    head_priorities = ['VP', 'VERB', 'NP', 'PROPN', 'NOUN', 'PRON', 'AJP', 'ADJ', 'S', 'CONJP', 'SCONJ', 'COP', 'CCONJ', 'AVP', 'ADV', 'PP']
    head_idx = None
    for head in head_priorities:
        if head in production:
            head_idx = production.index(head)
            break
        else:
            pass
    if head_idx is None:
        head_idx = 0
    return head_idx


def eval_rels(deps: dict, gold_deps: dict):
    assert sorted(deps) == sorted(gold_deps), ""
    correct_ = 0
    for tail, pred_head in deps.items():
        gold_head = gold_deps[tail]
        if pred_head == gold_head: correct_ += 1
    return correct_, len(deps)


if __name__ == "__main__":
    abbr = {'en': "UD_English-EWT", 'de': "UD_German-GSD", 'fr': "UD_French-GSD", 'he': "UD_Hebrew-HTB", 
            'ko': "UD_Korean-Kaist", 'sv': "UD_Swedish-Talbanken", 'zh': "UD_Chinese-GSD"}

    grammar_folder = "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/grammar_rules/"
    pred_const_folder = "log/evalb/"
    gold_depdt_folder = "/home/ljl/0_data/ud2.13_excluCross/"

    lans = ['en', 'de', 'fr', 'he', 'ko', 'sv', 'zh']
    lans = ['zh']
    for lan in lans:
        gramm_paths = [
            # grammar_folder + lan + '.train.grammar', grammar_folder + lan + '.dev.grammar', 
                       grammar_folder + lan + '.test.grammar',]
        pred_const_path = pred_const_folder + lan + '/predicted.txt'
        UDfolder = abbr[lan]
        low_abbr = UDfolder.split('-')[-1].lower()
        gold_ud_path = gold_depdt_folder + f"{UDfolder}/{lan}_{low_abbr}-ud-test.conllu"

        print("---"*5)
        grammars = dict()
        for path in gramm_paths:
            # if 'test' not in path: 
            grammars = load_grammars(path, grammars)
        grammars = process_grammars(grammars)
        
        correct_cnt, all_cnt = [], []

        const_trees: List[Tree] = load_treebank(pred_const_path, binarize=False)
        depdt_trees: List[DepDict] = load_dep_dicts(gold_ud_path)
        reserve_depdts = []
        reserve_consts = []
        for dept_t in depdt_trees:
            words = ptb_unescape(dept_t.words())
            patience = 0
            for const_t in const_trees:
                words_ = list(const_t.leaves())
                if words_ == words: 
                    reserve_depdts.append(dept_t)
                    reserve_consts.append(const_t)
                    const_trees.remove(const_t)
                    break
                elif patience > 20:
                    break
                else:
                    patience += 1

        assert len(reserve_consts) == len(reserve_depdts), ""
        print(f"reserve for eval {len(reserve_consts)} !!")
        micro_fs = []
        for i, (const_t, depdt_t) in enumerate(zip(reserve_consts, reserve_depdts)):
            if i == 69:
                print("debug!!!")
            assert list(const_t.leaves()) == ptb_unescape(depdt_t.words()), ""
            deps = dict()
            const_t.find_head(grammars, deps)
            tail_words = [w for w in const_t.leaves()]
            lost_root = list(sorted(deps))
            assert len(set(lost_root)) == len(lost_root), ""
            sort_tails = [i for i in range(1, len(tail_words)+1)]
            root_id = list(set(sort_tails) - set(lost_root))
            assert len(root_id) == 1 and deps.get(root_id[0], 'no exist') == 'no exist', ""
            deps[root_id[0]] = 0
            assert sorted(deps) == [i for i in range(1, len(tail_words)+1)], ""
            
            gold_deps: dict = depdt_t.edges()
            correct_, all_ = eval_rels(deps, gold_deps)
            assert correct_ <= all_, ""
            correct_cnt.append(correct_)
            all_cnt.append(all_)
            micro_fs.append(round(correct_/all_, 2))

        print(f"{lan} UAS: {round(sum(correct_cnt*100)/sum(all_cnt), 2)}")