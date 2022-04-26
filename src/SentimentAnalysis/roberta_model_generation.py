import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sklearn
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup

from src.SentimentAnalysis.roberta_config import RobertaConfig
from src.helper_functions import format_time, get_logger


# Followed guide from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
# https://colab.research.google.com/drive/1PHv-IRLPCtv7oTcIGbsgZHqrB5LPvB7S#scrollTo=EgR6MuNS8jr_

def training_epoch(model, device, optimizer, scheduler, logger, dataloader):
    t0 = time.time()

    total_loss = 0

    model.train()

    for step, batch in enumerate(dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            eta = format_time(((time.time() - t0) / step) * (len(train_dataloader) - step))
            logger.info("Step %d of %d ----- Elapsed time: %s ETA: %s", step, len(train_dataloader), elapsed, eta)
        batch_input_ids = batch["input_ids"].to(device)
        batch_mask = batch["attention_mask"].to(device)
        batch_targets = batch["targets"].to(device)
        model.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_mask, labels=batch_targets)
        loss = outputs[0]
        total_loss += loss.item()
        # Back Propagation
        loss.backward()
        # Prevent "Exploding Gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_loss = total_loss / len(train_dataloader)
    elapsed = format_time(time.time() - t0)

    loss_values.append(avg_loss)

    return avg_loss, elapsed


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def eval_epoch(model, device, dataloader):
    t0 = time.time()

    model.eval()

    eval_accuracy = 0
    num_eval_steps = 0

    for batch in dataloader:
        batch_input_ids = batch["input_ids"].to(device)
        batch_mask = batch["attention_mask"].to(device)
        batch_targets = batch["targets"].to(device)

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            target_ids = batch_targets.to('cpu').numpy()
            tmp_eval_acc = flat_accuracy(logits, target_ids)
            eval_accuracy += tmp_eval_acc
            num_eval_steps += 1

    eval_accuracy = eval_accuracy / num_eval_steps

    elapsed = format_time(time.time() - t0)

    return eval_accuracy, elapsed


def test_epoch(model, device, dataloader):
    t0 = time.time()

    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_mask = batch["attention_mask"].to(device)
            batch_targets = batch["targets"].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_mask)
            _, preds = torch.max(outputs[0].detach(), dim=1)
            predictions.extend(preds)
            targets.extend(batch_targets)

    elapsed = format_time(time.time() - t0)

    predictions = torch.stack(predictions).cpu()
    targets = torch.stack(targets).cpu()

    return predictions, targets, elapsed


class PostDataset(Dataset):
    def __init__(self, config: RobertaConfig, data_path, tokenizer):
        training_data = pd.read_csv(data_path, names=["text", "sentiment"])
        self.tokenizer = tokenizer
        self.max_len = config.max_post_len
        self.posts = training_data.iloc[:, 0].tolist()
        self.target_sentiments = training_data.iloc[:, 1].tolist()

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, index):
        post = self.posts[index]
        target_sentiment = self.target_sentiments[index]

        encoding = self.tokenizer.encode_plus(post, add_special_tokens=True,
                                              truncation=True,
                                              max_length=self.max_len,
                                              padding="max_length",
                                              return_attention_mask=True,
                                              return_tensors='pt')

        return {
            "post_text": post,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target_sentiment, dtype=torch.long)
        }


logger = get_logger(LOG_NAME="roberta_model_generation")
cf: RobertaConfig = RobertaConfig()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

roberta_tokenizer = RobertaTokenizer.from_pretrained(cf.pre_trained_model_name)

torch_dataset = PostDataset(cf, cf.training_data_path, roberta_tokenizer)
train_size = int((1 - cf.validation_split) * len(torch_dataset))
validation_size = len(torch_dataset) - train_size
training_dataset, validation_dataset = torch.utils.data.random_split(torch_dataset, [train_size, validation_size])
testing_dataset = PostDataset(cf, cf.testing_data_path, roberta_tokenizer)

train_dataloader = DataLoader(training_dataset, batch_size=cf.batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=cf.batch_size)
test_dataloader = DataLoader(testing_dataset, batch_size=cf.batch_size)

if not cf.load_model:
    model = RobertaForSequenceClassification.from_pretrained(cf.pre_trained_model_name, num_labels=cf.num_labels)

    optimizer = AdamW(model.parameters(), lr=cf.learning_rate, eps=cf.epsilon)

    model.to(device)

    total_steps = cf.num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    logger.info("train size: %d validation size: %d", train_size, validation_size)
    logger.info("num epochs: %d", cf.num_epochs)
    logger.info("num batches: %d", len(train_dataloader))
    logger.info("num steps: %d", total_steps)

    loss_values = []

    best_accuracy = 0

    for epoch in range(0, cf.num_epochs):
        logger.info("=========== Epoch %d ===========", epoch)
        logger.info("---------- Training ------------")

        avg_loss, time_elapsed = training_epoch(model, device, optimizer, scheduler, logger, train_dataloader)

        logger.info("Average training loss: %f", avg_loss)
        logger.info("Training epoch complete, took: %s", time_elapsed)

        logger.info("--------- Evaluation -----------")

        eval_acc, time_elapsed = eval_epoch(model, device, validation_dataloader)

        if eval_acc > best_accuracy:
            logger.info("Improved accuracy - saving model")
            model.save_pretrained(cf.model_save_location)
            best_accuracy = eval_acc

        logger.info("Accuracy: %f", eval_acc)
        logger.info("Evaluation complete, took: %s", time_elapsed)

logger.info("Loading model from %s" % cf.model_save_location)
model_path = Path(cf.model_save_location)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=cf.num_labels)
model.to(device)

logger.info("---------- Testing ------------")
y_pred, y_true, time_elapsed = test_epoch(model, device, test_dataloader)
class_names = ["Negative", "Positive"]
logger.info(sklearn.metrics.classification_report(y_true, y_pred))
cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
cf_plot = sklearn.metrics.ConfusionMatrixDisplay(cf_matrix).plot(values_format="d")
plt.show()
logger.info("Testing complete, took: %s", time_elapsed)
