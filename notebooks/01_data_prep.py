# Databricks notebook source
# MAGIC %md
# MAGIC - change params to your catalog/schema
# MAGIC - add some pdfs to the volume raw_data
# MAGIC - DBR 14.3 LTS

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from finreganalytics.dataprep.dataloading import load_and_clean_data, split
from finreganalytics.utils import get_spark

catalog = "users"
schema = "max_carduner"
raw_data_path = f"/Volumes/{catalog}/{schema}"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_data")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.training")

# COMMAND ----------

dbutils.fs.mkdirs(f"{raw_data_path}/training/cpt/text/train")
dbutils.fs.mkdirs(f"{raw_data_path}/training/cpt/text/val")

# COMMAND ----------

# MAGIC %md
# MAGIC upload pdfs to raw_data for this to work

# COMMAND ----------

doc_df = load_and_clean_data(f"{raw_data_path}/raw_data")
display(doc_df)

# COMMAND ----------

splitted_df = split(
    doc_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df)

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.splitted_documents")

# COMMAND ----------

cpt_df = get_spark().read.table(f"{catalog}.{schema}.splitted_documents")
cpt_train_df, cpt_val_df = cpt_df.select("text").randomSplit([0.98, 0.02])
cpt_train_df.write.mode("overwrite").format("text").save(
    f"{raw_data_path}/training/cpt/text/train"
)
cpt_val_df.write.mode("overwrite").format("text").save(
    f"{raw_data_path}/training/cpt/text/val"
)

# COMMAND ----------

path = f"{raw_data_path}/training/cpt/text/train"
display(spark.sql(f"select count(1) from text.`{path}`"))

# COMMAND ----------

path = f"{raw_data_path}/training/cpt/text/val"
display(spark.sql(f"select count(1) from text.`{path}`"))

# COMMAND ----------

root_path = f"{raw_data_path}/training/cpt/text/val"
dbutils.fs.rm(f"{root_path}/_committed_*", recurse=True)
dbutils.fs.rm(f"{root_path}/_started_*", recurse=True)
dbutils.fs.rm(f"{root_path}/_SUCCESS", recurse=False)

# COMMAND ----------

root_path = f"{raw_data_path}/training/cpt/text/train"
dbutils.fs.rm(f"{root_path}/_committed_*", recurse=True)
dbutils.fs.rm(f"{root_path}/_started_*", recurse=True)
dbutils.fs.rm(f"{root_path}/_SUCCESS", recurse=False)
