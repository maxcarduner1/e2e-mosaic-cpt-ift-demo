# Databricks notebook source
# 14.3 MLR LTS
%pip install -r ../requirements.txt

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
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId") # change this if you don't want to use existing cluster for the data prep that converts the input delta table into .jsonl files for us

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

get_dbutils().widgets.text("training_duration", "10ep", "training_duration") # Check the mlflow experiment metrics to see if you should increase this number
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "dbfs:/databricks/mlflow-tracking/3333570012308106/10881276cc5e4007a5f01259c28077e4/artifacts/contd-pretrain-meta-llama-3-1-8b-tg1i9b/checkpoints/ep1-ba3", #replace with yours
    "custom_weights_path",
)

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None

# COMMAND ----------

run = fm.create(
  model=base_model,
  train_data_path=f"{catalog}.{schema}.chat_completion_training_dataset",
  register_to=f"{catalog}.{schema}",
  training_duration=training_duration,
  learning_rate=learning_rate,
  custom_weights_path=custom_weights_path,
  task_type="CHAT_COMPLETION",
  data_prep_cluster_id=cluster_id
)

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

run.name

# COMMAND ----------

display(fm.list())

# COMMAND ----------

# using the UI serve this model as endpoint and set it in the ift_endpoint_name in next notebook and use in a cloned review app notebook
