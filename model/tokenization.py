import torch
from transformers import XLMRobertaTokenizer


# class Tokenizer():
#     def __init__(self, pretrained_model_name_or_path) -> None:
#         self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
    
#     def __call__(self, words_list):
#         # 对于无法识别的符号，提前转换成unk_token
#         def clean_snt(words):
#             src_len = len(words)
#             for j, w_ in enumerate(words):
#                 if len(self.tokenizer.tokenize(w_, add_special_tokens=False)) < 1:
#                     words[j] = self.tokenizer.unk_token
#             assert len(words) == src_len
#             return words

#         snts_cleaned = []
#         for words in words_list:
#             snt_cleaned = " ".join(clean_snt(words))
#             snts_cleaned.append(snt_cleaned)

#         tokenized = self.tokenizer(
#                                     snts_cleaned,
#                                     padding='longest',
#                                     max_length=512,
#                                     truncation='longest_first',
#                                     return_attention_mask=True,
#                                     return_offsets_mapping=True,
#                                     return_tensors="pt"
#                                     )   
#         # ids, attention_mask, offsets_mapping = tokenized['input_ids'], tokenized['attention_mask'], tokenized['offsets_mapping']
           
#         return tokenized


class Retokenizer():

    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)

    def __call__(self, batched_words_list):        
        input_ids_list, words_index_list = [], []
        for words_list in batched_words_list:
            
            input_ids, head_index = [], []
            offset = 0
            for idx, word in enumerate(words_list):
                tokenized = self.tokenizer(word, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                for_del = self.tokenizer.convert_tokens_to_ids('▁')
                if for_del in tokenized:
                    tokenized.remove(for_del)  # 将所有补位的▁都删除

                if len(tokenized) < 1:
                    input_ids.append(self.tokenizer.unk_token_id)
                    head_index.append(idx + 1 + offset) # head_index 从1开始，最后计算的时候直接获取word表示，不考虑<s>和 </s>
                else:
                    input_ids += tokenized
                    head_index.append(idx + 1 + offset) # head_index 从1开始，最后计算的时候直接获取word表示，不考虑<s>和 </s>

                offset += (len(tokenized)-1)

            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
            input_ids_list.append(input_ids)
            words_index_list.append(head_index)
        
        return self.pad(input_ids_list, words_index_list)

    def pad(self, input_ids_list, words_index_list):
        tokenized = {}

        batch_size = len(input_ids_list)
        max_inputs_len = max([len(input_ids) for input_ids in input_ids_list])
        max_words_num = max([len(word_index) for word_index in words_index_list])

        # for PLM
        pad_input_ids = torch.full((batch_size, max_inputs_len), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, input_ids in enumerate(input_ids_list):
            pad_input_ids[i, :len(input_ids)] = torch.tensor(input_ids)

        attention_mask = torch.full((batch_size, max_inputs_len), 0, dtype=torch.bool)
        for i, input_ids in enumerate(input_ids_list):
            attention_mask[i, :len(input_ids)] = 1

        # for parser
        pad_words_index = torch.full((batch_size, max_words_num), max_inputs_len-1, dtype=torch.long)
        for i, words_index in enumerate(words_index_list):
            pad_words_index[i,:len(words_index)] = torch.tensor(words_index)

        word_attention_mask = torch.full((batch_size, max_words_num), 0, dtype=torch.bool)
        for i, word_index in enumerate(words_index_list):
            word_attention_mask[i,:len(word_index)] = 1
    
        tokenized["input_ids"] = pad_input_ids
        tokenized["attention_mask"] = attention_mask
        tokenized["words_index"] = pad_words_index
        tokenized["word_attention_mask"] = word_attention_mask

        return tokenized
