# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

catalog = "users"
schema = "max_carduner"
base_data_path = f"/Volumes/{catalog}/{schema}"

# COMMAND ----------

from finreganalytics.utils import get_dbutils
get_dbutils().widgets.removeAll()

# COMMAND ----------

import os.path

from databricks.model_training import foundation_model as fm

from finreganalytics.utils import setup_logging, get_dbutils

setup_logging()

SUPPORTED_INPUT_MODELS = fm.get_models().to_pandas()["name"].to_list()
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Meta-Llama-3.1-8B", SUPPORTED_INPUT_MODELS, "base_model"
)
get_dbutils().widgets.text(
    "data_path", f"{base_data_path}/training/ift/jsonl", "data_path"
)

get_dbutils().widgets.text("training_duration", "10ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "dbfs:/databricks/mlflow-tracking/3333570012308106/10881276cc5e4007a5f01259c28077e4/artifacts/contd-pretrain-meta-llama-3-1-8b-tg1i9b/checkpoints/ep1-ba3", #replace with yours
    "custom_weights_path",
)

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
data_path = get_dbutils().widgets.get("data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None

# COMMAND ----------

run = fm.create(
  model=base_model,
  train_data_path=f"{data_path}/train.jsonl",
  eval_data_path=f"{data_path}/val.jsonl",
  register_to="main.finreg",
  training_duration=training_duration,
  learning_rate=learning_rate,
  custom_weights_path=custom_weights_path,
  task_type="INSTRUCTION_FINETUNE",
)

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

run.name

# COMMAND ----------

display(fm.list())
