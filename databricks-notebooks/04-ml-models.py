from pyspark.sql.types import *
from pyspark.sql.functions import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch

#ADLS configuration 
spark.conf.set(
  "fs.azure.account.key.hospitalpatientstorage.dfs.core.windows.net",
 dbutils.secrets.get(scope="hospitalanalyticvaultscope", key="storage-connection")
)
le = LabelEncoder()
silver_path = "abfss://silver@hospitalpatientstorage.dfs.core.windows.net/patient_flow"

# Read silver data (assume append-only)
silver_df = spark.read.format("delta").load(silver_path)
# -------------------------
# TEXT PRE-PROCESSING STEP
# 

df = df.select("clinical_note", "intent_label").dropna()
df_clean = df.withColumn(
    "clean_text",
    trim(lower(regexp_replace("clinical_note", "[^a-zA-Z ]", "")))
)

pdf = df_clean.select("clean_text", "label").toPandas()
#Label encoding
pdf["label"] = le.fit_transform(pdf["intent_label"])


# Train/Test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    pdf["clinical_note"].tolist(),
    pdf["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenization
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings   = tokenizer(val_texts, truncation=True, padding=True, max_length=128)




class HealthcareDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = HealthcareDataset(train_encodings, train_labels)
val_dataset   = HealthcareDataset(val_encodings, val_labels)


# Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_)
)


# Training arguments
training_args = TrainingArguments(
    output_dir="/dbfs/mnt/models/distilbert-healthcare",
    evaluation_strategy="epoch", # Run validation/evaluation once after each epoch.
    learning_rate=2e-5, #how much the model weights change after each update.
    per_device_train_batch_size=8, #Use 8 training samples at a time for each training step.
    per_device_eval_batch_size=8, #Use 8 validation samples at a time during evaluation.
    num_train_epochs=3, #Train on the entire training dataset 3 times.
    weight_decay=0.01, #Applies regularization to reduce overfitting by discouraging very large weights.
    save_strategy="epoch" #Save a model checkpoint after each epoch.
)

# Trainer
trainer = Trainer(
    model=model,   # DistilBERT model to fine-tune
    args=training_args, # Training configuration (epochs, batch size, learning rate, etc.)
    train_dataset=train_dataset, # Dataset used for training the model
    eval_dataset=val_dataset # Dataset used for validation/evaluation
)

trainer.train()
# Save model
model.save_pretrained("/dbfs/mnt/models/distilbert-healthcare")
tokenizer.save_pretrained("/dbfs/mnt/models/distilbert-healthcare")