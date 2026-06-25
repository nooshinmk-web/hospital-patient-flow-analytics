from pyspark.sql.types import *
from pyspark.sql.functions import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

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
