import os
import copy
import math
import random
import torch
import numpy as np
from fastNLP import Tester, DataSet, cache_results
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaForMaskedLM
from utils import hinge_loss
from sklearn.metrics import f1_score
from dataloader import SST2Loader, AGNewsLoader, SNLILoader
from metrics import SST2Metric, AGNewsMetric, SNLIMetric
from arguments import get_arguments

args = get_arguments()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# black box API
class LMForwardAPI:
    def __init__(self, intrinsic_dim=500, n_prompt_tokens=50, task_name='easy', loss_type='hinge',
                 inference_framework='pt', onnx_model_path=None, save_path=None, device='cuda:0'):
        self.config = RobertaConfig.from_pretrained('roberta-large')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaForMaskedLM.from_pretrained(
            'roberta-large',
            config=self.config,
            n_prompt_tokens=n_prompt_tokens,
            inference_framework=inference_framework,
            onnx_model_path=onnx_model_path,
        )
        self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        if inference_framework == 'ort':
            self.model.roberta = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if args.random_proj == 'normal':
            # calculate std for normal distribution
            embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            # embedding = embedding[1000: 2000]
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.device = device
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.save_path = save_path
        self.print_every = 50
        self.eval_every = 100
        self.loss_type = loss_type
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        if task_name == 'easy':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'medium':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'hard':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz = len(dev_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
        else:
            bsz = 32  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(50, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=32,
                                 num_workers=1, device=self.device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in train_data.items():
                train_data[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )['logits']

            loss, perf = self.calc_metric(logits, train_data['labels'])
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                for k, v in dev_data.items():
                    dev_data[k] = v.to(self.device)
                with torch.no_grad():
                    logits = self.model(
                        input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                        mask_pos=dev_data['mask_pos'],
                    )['logits']

                dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])
                # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
                    self.best_prompt = copy.deepcopy(tmp_prompt)
                # if self.save_path is not None:
                #     with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
                #         fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
                return loss, dev_loss
            return loss, None


# preparing data
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

DataLoader = {
    'easy': SST2Loader,
    'medium': AGNewsLoader,
    'hard': SNLILoader,
}


def construct_true_few_shot_data(train_data, k_shot):
    train_label_count = {}
    dev_label_count = {}
    new_train_data = DataSet()
    new_dev_data = DataSet()
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['labels']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            new_dev_data.append(train_data[index])
            dev_label_count[label] += 1

    new_train_data.set_input("input_ids", "attention_mask", "mask_pos")
    new_dev_data.set_input("input_ids", "attention_mask", "mask_pos")

    new_train_data.set_target("labels")
    new_dev_data.set_target("labels")
    return new_train_data, new_dev_data


@cache_results(f"caches/data_{args.task_name}_{args.seed}.pt", _refresh=False)
def get_data(task_name, tokenizer):
    if task_name == 'easy':
        splits = ['train', 'validation']
    else:  # for datasets without test set, we use dev set
        splits = ['train', 'test']
    data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=50).my_load(splits)
    return data_bundle


data_bundle = get_data(task_name=args.task_name, tokenizer=tokenizer)
if args.task_name == 'easy':
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('validation')
else:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')

train_data, dev_data = construct_true_few_shot_data(train_data, 16)
for ds in [train_data, dev_data, test_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)


train_data = {
    'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
    'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
    'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
    'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
}
dev_data = {
    'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
    'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
    'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
    'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
}