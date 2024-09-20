# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

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
    "data_path", f"{base_data_path}/training/cpt/text/train/", "data_path"
)

get_dbutils().widgets.text("training_duration", "1ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "5e-7", "learning_rate")

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
data_path = get_dbutils().widgets.get("data_path")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")

# COMMAND ----------

run = fm.create(
    model=base_model,
    train_data_path=data_path,
    eval_data_path=data_path,
    register_to=f"{catalog}.{schema}",
    training_duration=training_duration,
    learning_rate=learning_rate,
    task_type="CONTINUED_PRETRAIN",
)

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

run.name

# COMMAND ----------

display(fm.list())
