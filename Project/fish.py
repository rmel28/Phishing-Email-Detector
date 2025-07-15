import pandas as pd
import re
import torch

#libraies for exploring data
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

#libraries for Model training and evaluation
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, classification_report, confusion_matrix


#-----load the dataset using pandas-----
data_frame = pd.read_csv("phishing_email.csv.zip")
#print(data_frame.shape) -> (82797, 2)


#Randomly sample 15,000 rows from the full dataset
# Sample 15,000 rows first
data_frame = data_frame.sample(n=15000, random_state=42).reset_index(drop=True)

#Oversample phishing emails to balance the classes
phish_df = data_frame[data_frame['label'] == 1]
oversampled_data = pd.concat([data_frame, phish_df], ignore_index=True)
oversampled_data = oversampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Now use oversampled_data instead of data_frame
data_frame = oversampled_data

print(data_frame.head())


#-----Preprocess the Data-----
#remove null - lowercase
#why not doing all the preprocessing? -> in phishing attackers use all of the concpets fake. so if you remove everything the model will not learn the fake emails


#Remove nulls
data_frame.dropna(inplace= True) #removes any rows in the dataset that contain missing/null values. inplace=True -> changes made directly to data_fram without reassigning it
data_frame['label']= data_frame['label'].astype(int) #converts the label column to be an integer


#Lowercasing
def clean_text(text):
   return text.lower()


data_frame['text_combined']= data_frame['text_combined'].apply(clean_text)


#Exploring the data using graphs
plt.figure(figsize= (6,4))
sns.countplot(data= data_frame, x= 'label') # Creates a bar chart to show how many emails are labeled as legit (0) vs phishing (1), # It counts each unique value in the 'label' column and displays them as bars
plt.title('Phishing vs Legit Emails')
plt.xticks([0,1], ['Legit', 'Phishing'])
plt.ylabel('Count')
# plt.show()


#Special characters Distribuition
#!@#$%^&*...
#boxplot comparing the number of special characters used in legit vs phishing emails
data_frame['special_characters']= data_frame['text_combined'].apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x)) #counts how many sepcial charcters are in each email and puts them in a new collum


plt.figure(figsize= (8,5))
sns.boxplot(x= 'label', y='special_characters', data= data_frame, hue= 'label', palette= 'coolwarm', dodge= False)


plt.xticks([0,1], ['Legit', 'Phishing'])
plt.title('Special Characters Count by Class')
plt.xlabel('Label')
plt.ylabel('Special Characters Count')
plt.grid('Special Characters Count')
plt.grid(True)
# plt.show()


#Most common words
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


stop_words = set(stopwords.words('english')) #loads a set of common english words


#combine all phishing emails: Joins all texts into oe string -> converts to lowercase -> splits into words
phishing_words= ' '.join(data_frame[data_frame['label'] == 1]['text_combined']).lower().split()


filtered_words= [word for word in phishing_words if word.isalpha() and word not in stop_words]#keeps only alphabetic words ignores numbers and puncuation, removes the stop words like 'and', 'is', etc


word_freq = Counter(filtered_words).most_common(20)#counts how often each word appears and get the top 20 most common words in phishing emails


#Barplot of top words
words, counts= zip(*word_freq) #puts the 20 word-count pairs into two seperate lists: words and counts
plt.figure(figsize= (10,6))
sns.barplot(x = list(counts), y= list(words), hue= list(counts), palette= 'magma', legend= False)
plt.title('Top 20 words in Phishing Emails')
plt.xlabel('Frequency')
# plt.show()


from wordcloud import WordCloud
wordcloud_text = ' '.join(filtered_words)


wordcloud = WordCloud(width=800, height=400, background_color='white', colormap= 'cool').generate(wordcloud_text)


plt.figure(figsize= (12, 6))
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title('Word Cloud of Phishing Email Words')
# plt.show()


#-----Split the data-----
#20% test data - 70% train data - 10% validation


#test data
#from sklearn.model_selection...
train_val_texts, text_texts, train_val_labels, test_labels = train_test_split(
   data_frame['text_combined'].tolist(), data_frame['label'].tolist(), test_size= 0.2, random_state= 42, stratify= data_frame['label']) #splits up data 80-20 into train text and test text - same with labels. Keeps the ratio the same among all data sets


#reamining 80% into train and validation
train_val_texts, val_texts, train_labels, val_labels = train_test_split(
   train_val_texts, train_val_labels, test_size= 0.125, random_state= 42, stratify= train_val_labels #takes the 80% from ealier and splits it into training and validation
)#10% of original = 12.5% of the 80%


#-----Tokenization using BERT Tokenizer-----
#Encodings that are tokenized versions of text that BERT can understand
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


#tokenize the data set
train_encodings = tokenizer(train_val_texts, truncation= True, padding= True)
val_encodings = tokenizer(val_texts, truncation= True, padding= True)


#View IDs
# print(train_encodings['input_ids'][0])
# print(train_encodings['attention_mask'][0])


#formatting the data using PyTorch
class EmailDataset(torch.utils.data.Dataset):
   def __init__(self, encodings, labels):#encodings = toeknized dat - labels = 0 or 1(phishing or legit)
       self.encodings = encodings
       self.labels = labels


   def __getitem__(self, idx):##lets PyTorch retirve one item (a single email and its label from the data set using and index sinxe its a lsit)
       item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}#builds a dict - for each endosign it grabs thevalue at that index and turns it into a PyTorch tesnor
       #tensor - list of lists of list...
       item['labels'] = torch.tensor(self.labels[idx])
       return item


   def __len__(self):
       return len(self.labels)


train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)




#-----Train(finetune) BERT model-----
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels= 2)


#Define Evaluation Metrics
#evaluate how well the trainer is on the test data
def compute_metrics(pred):
   labels = pred.label_ids
   preds = pred.predictions.argmax(-1)
   precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average= 'binary')
   acc = accuracy_score(labels, preds)
   return{
       'accuracy': acc,
       'f1': f1,
       'precision': precision,
       'recall': recall
   }


training_args = TrainingArguments(
   output_dir='./results',            # Saves checkpoint after each epoch
   eval_strategy="epoch",             # Validates after each epoch
   save_strategy="epoch",             # Saves checkpoint after each epoch
   save_total_limit=2,                # Keep only last 2 checkpoints
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=5,
   weight_decay=0.01,
   optim="adamw_torch",
   logging_dir='./logs',              # Log once per epoch
   logging_strategy="epoch",
   load_best_model_at_end=True,       # Load best val score model
   metric_for_best_model="f1",        # Based on compute_metrics
   greater_is_better=True             # Because higher F1 is better
)


#sets up full training pipeline- model, confid, datasets, token, padding logic, eval metrics
trainer = Trainer(
   model = model,
   args = training_args,
   train_dataset = train_dataset,
   eval_dataset = val_dataset,
   tokenizer = tokenizer,
   data_collator = DataCollatorWithPadding(tokenizer= tokenizer),
   compute_metrics = compute_metrics
)


#Start training
trainer.train()

#save the model
model_dir = "C:/Users/bmons/Desktop/Model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained('C:/Users/bmons/Desktop/Model')
tokenizer = BertTokenizer.from_pretrained('C:/Users/bmons/Desktop/Model')

print("Log history:", trainer.state.log_history)
#Extract the logs
log_history = trainer.state.log_history

#Gather Data
train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
eval_acc = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]
epochs = list(range(1, len(eval_loss)+1))

#----Test the model on the 20% data-----
test_encodings = tokenizer(text_texts, truncation= True, padding= True)

#convert to Pytorch
test_dataset = EmailDataset(test_encodings, test_labels)

eval_args = TrainingArguments(
    output_dir='./eval_results',
    per_device_eval_batch_size=16,
)

eval_trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer
)

# Run evaluation
results = eval_trainer.evaluate(test_dataset)
print("Evaluation Metrics on Test Set:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

    predictions = eval_trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

print("\nClassification Report:")
print(classification_report(test_labels, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, preds))
