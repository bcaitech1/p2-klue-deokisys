import pickle as pickle
import os
import pandas as pd
import torch


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.


def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    sentence_ent = []
    for i, sentence in enumerate(dataset[1]):
        sentence_ent.append(insert_ent_tag(
            sentence, dataset[3][i], dataset[4][i], dataset[6][i], dataset[7][i]))
    out_dataset = pd.DataFrame(
        {'sentence': sentence_ent, 'entity_01': dataset[2], 'entity_02': dataset[5], 'label': label})
    # {'sentence': dataset[1], 'label': label})

    return out_dataset

# tsv 파일을 불러옵니다.


def load_data(dataset_dir):
    # 42가지의 타입들을 불러온다.
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # 데이터이름,문장,단어1,단어1시작위치,단어1끝위치,단어2,단어2시작위치,단어2끝위치,라벨
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.


def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    # entity1[sep]entity2를 저장
    special_tokens_dict = {'additional_special_tokens': [
        '[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),  # 문장
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    # 42000,42001
    # 42002,42003

    # 이순신 sep 조선 sep 이순신은 조선 중기의 무신이다. 형식으로 만든다.
    return tokenized_sentences


def insert_ent_tag(text, ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx):
    new_text = ''

    for i, ch in enumerate(text):
        if i == ent1_start_idx:
            new_text += '[E1]'
            new_text += ch

            if ent1_start_idx == ent1_end_idx:
                new_text += '[/E1]'

        elif i == ent1_end_idx:
            new_text += ch
            new_text += '[/E1]'

        elif i == ent2_start_idx:
            new_text += '[E2]'
            new_text += ch

            if ent2_start_idx == ent2_end_idx:
                new_text += '[/E2]'

        elif i == ent2_end_idx:
            new_text += ch
            new_text += '[/E2]'

        else:
            new_text += ch
    return new_text
