import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download

# Download necessary NLTK data
download('punkt')
download('stopwords')

# Load dataset
dataset = load_dataset('toughdata/quora-question-answer-dataset')

# Convert to DataFrame
df = pd.DataFrame(dataset['train'])

# Data exploration
print(df.info())
print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# Tokenization, Stop Word Removal, Stemming/Lemmatization
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [ps.stem(w) for w in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['question'], examples['answer'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset
train_dataset, eval_dataset = train_test_split(tokenized_datasets['train'], test_size=0.2)

# Define model
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train and evaluate
trainer.train()
trainer.evaluate()

# Data visualization
# Distribution of Question Lengths
df['question_length'] = df['question'].apply(lambda x: len(x.split()))
sns.histplot(df['question_length'])
plt.title('Distribution of Question Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Distribution of Answer Lengths
df['answer_length'] = df['answer'].apply(lambda x: len(x.split()))
sns.histplot(df['answer_length'])
plt.title('Distribution of Answer Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['question_length', 'answer_length']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap between Question Length and Answer Length')
plt.show()

# Plotting model performance
results = trainer.evaluate()
metrics = ['eval_loss', 'eval_accuracy']
values = [results['eval_loss'], results['eval_accuracy']]

plt.bar(metrics, values)
plt.title('Model Performance')
plt.ylabel('Score')
plt.show()

# Training loss over time
training_loss = trainer.state.log_history

# Extracting only the 'loss' and 'epoch' entries
loss_values = [entry['loss'] for entry in training_loss if 'loss' in entry]
epochs = [entry['epoch'] for entry in training_loss if 'epoch' in entry]

plt.plot(epochs, loss_values, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
