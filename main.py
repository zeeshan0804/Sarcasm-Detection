import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from config import Config
from model import Model
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import utils

if torch.cuda.is_available():
    print("Using: ", torch.cuda.get_device_name(0))
else:
    print("Using: CPU")

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the required arguments
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
num_capsules = 10  # Example value, adjust as needed
num_routes = 32    # Example value, adjust as needed
d_model = 768      # Typically the hidden size of BERT
num_classes = 2    # Number of output classes (e.g., sarcasm or not sarcasm)
ner_dim = 10       # Example value, adjust as needed

# Load the BERT model
bert_model = BertModel.from_pretrained(bert_model_name)

# Load GloVe embeddings
word2id = utils.load_word2id()  # Load word2id mapping
glove_embeddings = utils.load_glove(word2id)  # Load GloVe embeddings

# Create the model instance with the required arguments
model = Model(bert_model, glove_embeddings).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-3)

def train():
    # Load the training data
    train_contents, train_labels = utils.load_corpus(config.train_path, config.max_sen_len)

    model.train()
    loss_train = []
    for epoch in range(config.epoch):
        total_epoch_loss = 0
        total_correct = 0
        total_samples = 0
        steps = 0

        for batch_x, batch_y in utils.batch_iter(train_contents, train_labels, config.batch_size):
            # Map tokens to GloVe indices
            batch_x_glove = [utils.map_tokens_to_glove_indices(tokens, word2id) for tokens in batch_x]
            batch_x_glove = torch.tensor(batch_x_glove).to(device)
            
            print(f"GloVe embeddings size: {model.glove_embeddings.weight.size()}")
            print(f"Max index in batch_x_glove: {batch_x_glove.max().item()}")

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x_glove, batch_y)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
            steps += 1

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

def validation(avg_epoch_loss):
    # Load the validation data
    val_contents, val_labels = utils.load_corpus(config.val_file, config.max_sen_len)

    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    steps = 0

    with torch.no_grad():
        for batch_x, batch_y in utils.batch_iter(val_contents, val_labels, config.batch_size):
            # Map tokens to GloVe indices
            batch_x_glove = [utils.map_tokens_to_glove_indices(tokens, word2id) for tokens in batch_x]
            batch_x_glove = torch.tensor(batch_x_glove).to(device)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x, batch_y)
            loss = F.cross_entropy(logits, batch_y)

            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
            steps += 1

    avg_val_loss = total_val_loss / steps
    final_acc = (total_correct / total_samples) * 100
    print('Validation Loss: {0:.3f}'.format(avg_val_loss),
          'Validation Acc: {0:.3f}%'.format(final_acc))
    return avg_val_loss

def test():
    # Load the test data
    test_contents, test_labels = utils.load_corpus(config.test_file, config.max_sen_len)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in utils.batch_iter(test_contents, test_labels, config.batch_size):
            # Map tokens to GloVe indices
            batch_x_glove = [utils.map_tokens_to_glove_indices(tokens, word2id) for tokens in batch_x]
            batch_x_glove = torch.tensor(batch_x_glove).to(device)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x, batch_y)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds)}")
    print(f"F1 Score: {f1_score(all_labels, all_preds, average='weighted')}")

if __name__ == "__main__":
    train()
    test()