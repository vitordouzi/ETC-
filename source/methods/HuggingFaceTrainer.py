
from .trainers import Trainect

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
from ..utils.dataset import HugDataset
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
import shutil

f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    return f1
class HuggingFaceTrainer(Trainect):
    def __init__(self, tname, descriptor):
        super(HuggingFaceTrainer, self).__init__(tname, descriptor)
        self.model_name = descriptor['model_name']
        self.batch_size = descriptor['batch_size']
        self.max_length = descriptor['max_length']
        self.epochs     = descriptor['epochs']
        self.learning_rate = descriptor['learning_rate']
        self.device = descriptor['device'] if "device" in descriptor else ("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    def init_model(self, fold, output_dir: str = None):
        
        print('initing model...')
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=fold.nclass, max_length=self.max_length).to(self.device)
        print('initing tknz...')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=self.max_length)
        print('initing dataset...')
        fold = fold.to('hugging', tokenizer=tokenizer)
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            save_steps=len(fold['train'])//(self.batch_size*2),
            save_strategy='steps',
            save_total_limit=1,
            load_best_model_at_end=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            evaluation_strategy='steps',
            report_to="none",
            eval_steps=len(fold['train'])//(self.batch_size*2)
        )
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(fold['train'])//self.batch_size*self.epochs)
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=fold['train'],
            eval_dataset=fold['val'],
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics
        )
        return (model, tokenizer, trainer)

    def train_model(self, model, fold):
        (model, tokenizer, trainer) = model
        result = { "train_output": trainer.train() }
        if os.path.exists(self.training_args.output_dir):
            shutil.rmtree(self.training_args.output_dir)
        return result
        
    def predict(self, model, X):
        (model, tokenizer, trainer) = model
        test_dataset = HugDataset(X, [0] * len(X), tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        predictions = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.softmax(outputs.logits, dim=1).argmax(dim=-1).cpu().numpy()
                predictions.extend(preds)

        return predictions