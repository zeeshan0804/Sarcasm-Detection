import numpy as np
import os
import torch
import torch.nn.functional as F
import utils
from config import Config
from model import Model
from pytorch_pretrained_bert import BertModel
# from transformers import BertModel

from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix

if torch.cuda.is_available():
    print("Using: ", torch.cuda.get_device_name(0))
else:
    print("Using: CPU")

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# utils.build_word2id(config.train_path)
# word2id = utils.load_word2id()


bert = BertModel.from_pretrained('bert-base-uncased')

model = Model(bert).to(device)
# model = Model(word2id).to(device)


#loss_fn = F.cross_entropy -- removed due to CRF

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-3)  #学习率le-4,L2值：tweet数据集是le-3,IAC数据集是le-2


loss_train = []
loss_test = []


def evaluate(y_pred, y_true):
    print("Precision: ", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred, average="macro"))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def validation(train_loss):
    # model.eval()
    # eval_data, eval_label = utils.load_corpus(config.test_path, max_sen_len=config.max_sen_len)
    # batch_eval = utils.batch_iter(eval_data, eval_label, batch_size=1)
    # eval_acc = 0
    # eval_loss = 0
    # for x_batch, y_batch in batch_eval:
    #     x_batch = x_batch.to(device)
    #     y_batch = y_batch.to(device)
    #     # with torch.no_grad():
    #         # prediction = model(x_batch)
    #     # loss = loss_fn(prediction, y_batch)
    #     num_correct = (torch.max(prediction, 1)[1] == y_batch.data).sum()
    #     eval_loss += float(loss)
    #     eval_acc += float(num_correct)
    # eval_loss = eval_loss / len(eval_data)
    # eval_acc = 100.0 * eval_acc / len(eval_data)
    if train_loss < config.best_loss:
        config.best_loss = train_loss
        print("better model---------------------")
        print("save model---------------------")
        torch.save(model.state_dict(), config.model_save_path)
        print("current best loss: {0:.6f}".format(train_loss))
        config.hit_patient = 0
        # return eval_loss, eval_acc
    config.hit_patient += 1
    torch.save(model.state_dict(), config.model_save_path)
    return True


def train():
    train_data, train_labels = utils.load_corpus(config.train_path, max_sen_len=config.max_sen_len)
    
    for epoch in range(config.epoch):
        model.train()
        print('Epoch: {0:02}'.format(epoch + 1))
        total_epoch_loss = 0
        total_correct = 0
        total_samples = 0
        batch_train = utils.batch_iter(train_data, train_labels, config.batch_size)

        steps = 1
        for x_batch, y_batch in batch_train:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Get loss
            loss = model(x_batch, y_batch)
            
            # Get predictions for accuracy
            with torch.no_grad():
                predictions = model(x_batch)
                total_correct += (predictions == y_batch).sum().item()
                total_samples += len(y_batch)
            
            loss.backward()
            optimizer.step()
            
            if steps % 100 == 0:
                print("batch:", steps)
                current_acc = (total_correct / total_samples) * 100
                print('Training Loss: {0:.4f}'.format(loss.item()), 
                      'Training Accuracy: {0:.2f}%'.format(current_acc))
            
            steps += 1
            total_epoch_loss += loss.item()

        avg_epoch_loss = total_epoch_loss / steps
        final_acc = (total_correct / total_samples) * 100
        if config.hit_patient >= config.early_stop_patient:
          print("hit final patient-----------------------")
          print("Early Stopping")
          break
        print("\n")
        print('Epoch: {0:02}'.format(epoch + 1))
        print('Train Loss: {0:.3f}'.format(avg_epoch_loss),
              'Train Acc: {0:.3f}%'.format(final_acc))
        
        saved = validation(avg_epoch_loss)
        loss_train.append(avg_epoch_loss)
        print("\n")
        
def test():
    test_data, test_label = utils.load_corpus(config.test_path, max_sen_len=config.max_sen_len)
    batch_test = utils.batch_iter(test_data, test_label, batch_size=len(test_label))
    y_true = []
    y_pred = []
    
    try:
        model.load_state_dict(torch.load(config.model_save_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in batch_test:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(x_batch)
            
            # Ensure predictions and labels are on CPU for numpy conversion
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    
    # Add some basic checks
    print(f"Number of test samples: {len(y_true)}")
    print(f"Number of predictions: {len(y_pred)}")
    print(f"Unique values in predictions: {set(y_pred)}")
    print(f"Unique values in true labels: {set(y_true)}")
    
    evaluate(y_pred, y_true)

def validation(train_loss):
    model.eval()
    eval_data, eval_label = utils.load_corpus(config.test_path, max_sen_len=config.max_sen_len)
    batch_eval = utils.batch_iter(eval_data, eval_label, batch_size=config.batch_size)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    steps = 0
    
    with torch.no_grad():
        for x_batch, y_batch in batch_eval:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get loss
            loss = model(x_batch, y_batch)
            total_loss += loss.item()  # Now this should be a scalar
            
            # Get predictions for accuracy
            predictions = model(x_batch)  # This will return predictions
            total_correct += (predictions == y_batch).sum().item()
            total_samples += len(y_batch)
            steps += 1
    
    avg_loss = total_loss / steps
    accuracy = (total_correct / total_samples) * 100
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    if avg_loss < config.best_loss:
        config.best_loss = avg_loss
        print("better model---------------------")
        print("save model---------------------")
        torch.save(model.state_dict(), config.model_save_path)
        print("current best loss: {0:.6f}".format(avg_loss))
        config.hit_patient = 0
    else:
        config.hit_patient += 1
    
    return True

def recordLoss():
    f = open(config.train_loss_path, 'w')
    for i in loss_train:
        f.write(str(i) + ",\n")
    f.close()
    f = open(config.test_loss_path, 'w')
    for i in loss_test:
        f.write(str(i) + ",\n")
    f.close()


train()
recordLoss()
test()
