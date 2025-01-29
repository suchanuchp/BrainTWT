import argparse
import torch
import pandas as pd
from datasets import Dataset
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer, BertConfig, PreTrainedTokenizerFast, AutoModelForSequenceClassification

import numpy as np
from datasets import load_metric
from tqdm import tqdm

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss

from train_tokenizer import train_graph_tokenizer
from evaluate import load_labels, load_model, train_multiclass


class Temporal_Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


class BertForTemporalClassification(BertPreTrainedModel):
    '''
    Train a model only for temporal classification with CrossEntropyLoss
    '''

    def __init__(self, config):
        super().__init__(config)
        self.temporal_num_labels = config.temporal_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # temporal classification part
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        loss = temporal_loss
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# modified from BertForMlmTemporalClassification(BertPreTrainedModel)
class BertForMlmTemporalGraphClassification(BertPreTrainedModel):
    '''
        Train a model MLM for node masking and temporal graph classification -> predict graph index
        Use the temporal_weight to control the tradeoff between the two.

    '''

    def __init__(self, config):
        super().__init__(config)
        self.temporal_graph_num_labels = config.temporal_graph_num_labels  # TODO: self.temporal_graph_num_labels (# of graphs)
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_graph_num_labels)
        self.mlm = BertOnlyMLMHead(config)
        self.init_weights()
        self.temporal_weight = config.temporal_weight
        self.mlm_weight = config.mlm_weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            temporal_graph_labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # mlm part
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        # temporal classification part
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_graph_num_labels), temporal_graph_labels.view(-1))

        loss = self.mlm_weight * masked_lm_loss + self.temporal_weight * temporal_loss
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss, masked_lm_loss, temporal_loss) + outputs

        return outputs


def get_graph_tokenizer(dataset_name, walk_len, save_suffix):
    if 'tune' in save_suffix:
        graph_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f'data/{dataset_name}/models/graph_tokenizer_tune.tokenizer.json', max_len=walk_len)
    else:
        graph_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f'data/{dataset_name}/models/graph_tokenizer_{save_suffix}.tokenizer.json', max_len=walk_len)
    graph_tokenizer.unk_token = "[UNK]"
    graph_tokenizer.sep_token = "[SEP]"
    graph_tokenizer.pad_token = "[PAD]"
    graph_tokenizer.cls_token = "[CLS]"
    graph_tokenizer.mask_token = "[MASK]"
    return graph_tokenizer


def train_mlm(dataset: Dataset, graph_tokenizer: PreTrainedTokenizerFast, dataset_name: str):
    '''
    Train only masking model for node-level making
    '''
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=graph_tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="./",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_steps=0,
        save_total_limit=0,
    )

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_position_embeddings=64
    )

    model = BertForMaskedLM(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(f'data/{dataset_name}/models/masking_model')


def tokenize_function(graph_tokenizer, examples, sent_col_name):
    return graph_tokenizer(examples[sent_col_name], padding='max_length', truncation=True)


def train_mlm_temporal_model(random_walk_path: str, dataset_name: str,
                             walk_len: int, sample_num: int = None,
                             save_suffix='', opt=None):
    '''
    Train mlm and temporal model together (TM + MLM), save torch model
    :param random_walk_path: file path to load the random walks corpus (created in create_random_walks.py)
    :param dataset_name:
    :param walk_len: length of a random walk, define the length of the sequence for the model
    :param sample_num: train using a sample number
    '''
    temporal_weight = opt['temporal_weight']
    mlm_weight = opt['mlm_weight']
    epochs = opt['epochs']
    data_df = pd.read_csv(random_walk_path, index_col=None)
    label_col = 'graph_idx'
    unique_ids = data_df[label_col].unique().tolist()

    with open(f'data/{dataset_name}/models/labels_{save_suffix}', 'w') as f:
        for graph_id in unique_ids:
            f.write(f"{graph_id}\n")

    graph_idx_mapping = {value: idx for idx, value in enumerate(unique_ids)}
    data_df[label_col] = data_df[label_col].map(graph_idx_mapping)

    graph_tokenizer = get_graph_tokenizer(dataset_name, walk_len, save_suffix)

    if sample_num:
        data_df = data_df.sample(sample_num)

    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda examples: tokenize_function(graph_tokenizer, examples, 'sent'), batched=True,
                          batch_size=512)
    cols = ['input_ids', 'attention_mask']
    dataset = dataset.remove_columns(["sent", 'token_type_ids', 'Unnamed: 0'])
    dataset.set_format(type='torch', columns=cols + ['graph_idx', 'graph_label'])

    labels = dataset['input_ids']
    mask = dataset['attention_mask']
    temporal_graph_labels = dataset[label_col]
    num_classes = len(set(dataset[label_col].numpy()))

    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != graph_tokenizer.cls_token_id) * (
            input_ids != graph_tokenizer.pad_token_id) * (input_ids != graph_tokenizer.sep_token_id) * (
                       input_ids != graph_tokenizer.unk_token_id)
    selection = ((mask_arr).nonzero())
    input_ids[selection[:, 0], selection[:, 1]] = graph_tokenizer.mask_token_id

    d = Temporal_Graph_Dataset({'input_ids': input_ids, 'attention_mask': mask, 'labels': labels,
                                'temporal_graph_labels': temporal_graph_labels
                                })
    loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True)

    config = BertConfig(
        vocab_size=graph_tokenizer.vocab_size,
        hidden_size=opt['hidden_size'],
        num_hidden_layers=opt['num_hidden_layers'],
        num_attention_heads=opt['num_attention_heads'],
        max_position_embeddings=walk_len + 4,
        temporal_graph_num_labels=num_classes,
        temporal_weight=temporal_weight,
        mlm_weight=mlm_weight
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForMlmTemporalGraphClassification(config).to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=opt['lr'])
    # epochs = 5

    total_loss = []
    mlm_loss = []
    t_loss = []

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            temporal_labels = batch['temporal_graph_labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                            temporal_graph_labels=temporal_labels)
            # extract loss
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item(), mlm_loss=outputs[1].item(), t_loss=outputs[2].item())

            total_loss.append(loss.item())
            mlm_loss.append(outputs[1].item())
            t_loss.append(outputs[2].item())

            if i % 1000 == 0:
                print(f'loss={np.mean(total_loss)}, mlm_loss={np.mean(mlm_loss)}, t_loss={np.mean(t_loss)}')
                total_loss = []
                mlm_loss = []
                t_loss = []

    torch.save(model, f'data/{dataset_name}/models/mlm_and_temporal_model_{save_suffix}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--rw_path', type=str, default='data/abide/paths_walk_len_20_num_walks_30.csv')
    parser.add_argument('-d', '--dataset', type=str, default='abide')
    parser.add_argument('-w', '--walk_length', type=int, default=20)
    parser.add_argument('-s', '--save_suffix', type=str, default='')
    parser.add_argument('-n', '--sample_num', type=int, default=None)
    parser.add_argument('-t', '--temporal_weight', type=int, default=5)
    parser.add_argument('-m', '--mlm_weight', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=252)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--num_attention_heads', type=int, default=4)

    args = parser.parse_args()
    opt = vars(args)

    dataset_name = opt['dataset']
    save_suffix = opt['save_suffix']
    random_walk_path = opt['rw_path']
    walk_len = opt['walk_length']
    sample_num = opt['sample_num']

    print('training graph tokenizer...')
    train_graph_tokenizer(random_walks_file_path=random_walk_path,
                          dataset_name=dataset_name,
                          walk_len=walk_len,
                          save_suffix=save_suffix)

    print('training MLM temporal model...')
    train_mlm_temporal_model(random_walk_path=random_walk_path,
                             dataset_name=dataset_name,
                             walk_len=walk_len,
                             sample_num=sample_num,
                             save_suffix=save_suffix,
                             opt=opt)

    dataset = opt['dataset']
    save_suffix = opt['save_suffix']
    emb = load_model(f'data/{dataset}/models/mlm_and_temporal_model_{save_suffix}')
    labels = load_labels(f'data/{dataset}/models/labels_{save_suffix}')
    train_multiclass(emb, labels, n_splits=10, random_state=0)


if __name__ == '__main__':
    main()
