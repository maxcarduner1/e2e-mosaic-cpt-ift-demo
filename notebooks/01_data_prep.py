# Databricks notebook source
# MAGIC %md
# MAGIC - change params to your catalog/schema
# MAGIC - add .txt files to the volume raw_data
# MAGIC - Compute:
# MAGIC   - Runtime: DBR 14.3 LTS 
# MAGIC   - For this first notebook will likely want to make it multinode as well, disable autoscaling and just set it to 4 or 6 nodes to process the data faster

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
# MAGIC upload .txt files to raw_data for this to work

# COMMAND ----------

# use below to read each file into a row of a spark dataframe

from pyspark.sql import functions as F
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from typing import Iterator
import re

# Assuming the volume with .txt files is mounted and its path is specified
txt_files_volume_path = f"{raw_data_path}/raw_data/"

# Read all .txt files from the volume into a dataframe
doc_df = spark.read.text(txt_files_volume_path + "*.txt") \
              .withColumn("path", F.input_file_name())

def clean(txt):
    txt = re.sub(r"\n", "", txt)
    txt = re.sub(r" ?\.", ".", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s\.,;:!?()\-]", "", txt)
    return txt

def ensure_utf8_encoding(txt):
    # Encode to UTF-8, ignoring errors, then decode back to a string
    txt = clean(txt)
    return txt.encode('utf-8', 'ignore').decode('utf-8')

@pandas_udf("string")
def ensure_utf8_encoding_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for series in batch_iter:
        yield series.apply(ensure_utf8_encoding)

# Assuming doc_df is already defined and has a column named "text"
doc_df = doc_df.withColumn("utf8_text", ensure_utf8_encoding_udf(col("value")))

# display(doc_df)

#Concatenate all lines in each file into a single string, and associate with file name
doc_df = doc_df.groupBy("path") \
    .agg(F.collect_list("utf8_text").alias("all_lines")) \
    .withColumn("text", F.concat_ws("\n", F.col("all_lines"))) \
    .drop("all_lines")

display(doc_df)

# COMMAND ----------

# uncomment below if you want to load in pdfs
# doc_df = load_and_clean_data(f"{raw_data_path}/raw_data")
# display(doc_df)

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
