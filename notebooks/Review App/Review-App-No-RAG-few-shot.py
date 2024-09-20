# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U databricks-agents databricks-sdk mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#Lovekush - update this
catalog = 'users'
db = 'max_carduner'

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
from databricks.sdk import WorkspaceClient

context = get_context()

secret_scope_name = "max_demo" #LK update this

databricks_host = context.browserHostName
databricks_api_token = context.apiToken

import os
# Add secrets to desired secret scope and key in Databricks Secret Store
w = WorkspaceClient(host=databricks_host, token=databricks_api_token)
try:
  w.secrets.create_scope(scope=secret_scope_name)
except:
  pass #already created

w.secrets.put_secret(scope=secret_scope_name, key='api_token', string_value=databricks_api_token)
w.secrets.put_secret(scope=secret_scope_name, key='host_url', string_value=databricks_host)

os.environ['DATABRICKS_HOST'] = dbutils.secrets.get(scope=secret_scope_name, key='host_url')
os.environ['DATABRICKS_API_TOKEN'] = dbutils.secrets.get(scope=secret_scope_name, key='api_token')



# COMMAND ----------

def wait_for_model_serving_endpoint_to_be_ready(ep_name):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # TODO make the endpoint name as a param
    # Wait for it to be ready
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(ep_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

# COMMAND ----------

import mlflow
import yaml 

rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-meta-llama-3-1-70b-instruct",
        "host_url": databricks_host
    },
    "input_example": {
        "messages": [
            {"role": "user", "content": "What is Apache Spark"},
            {"role": "assistant", "content": "Apache spark is a distributed, OSS in-memory computation engine."},
            {"role": "user", "content": "Does it support streaming?"}
        ]
    },
    #Lovekush - see below and add your examples here, feel free to tweak this prompt
    "llm_config": {
        "llm_parameters": {"max_tokens": 5000, "temperature": 0.01},
        "llm_prompt_template": '''You are a marketing professional trusted assistant at blackbaud that helps write rough draft copy into the approved tone and voice. Only focus on the content that is provided by the user, don't add any additional context, just focus on getting it into the approved tone. Do not repeat information, answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer. See examples below: 
        
        copy:
        final copy:

        copy:
        final copy:

        copy:
        final copy:

        copy:
        final copy:
        
        copy: {copy}''',
        "llm_prompt_template_variables": ["copy"], 
    },
}
try:
    with open('chain_config_no_rag.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')
model_config = mlflow.models.ModelConfig(development_config='chain_config_no_rag.yaml')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.

# COMMAND ----------

# MAGIC %md When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

# DBTITLE 1,Chat History Extractor Chain
# MAGIC %%writefile chain_no_rag.py
# MAGIC from operator import itemgetter
# MAGIC import mlflow
# MAGIC import os
# MAGIC
# MAGIC from langchain_openai import ChatOpenAI,OpenAI
# MAGIC
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import (
# MAGIC     PromptTemplate,
# MAGIC     ChatPromptTemplate,
# MAGIC     MessagesPlaceholder,
# MAGIC )
# MAGIC from langchain_core.messages import HumanMessage, AIMessage
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # Return the chat history, which is is everything before the last question
# MAGIC def extract_chat_history(chat_messages_array):
# MAGIC     return chat_messages_array[:-1]
# MAGIC
# MAGIC # Load the chain's configuration
# MAGIC model_config = mlflow.models.ModelConfig(development_config="chain_config_no_rag.yaml")
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC # retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", llm_config.get("llm_prompt_template")),
# MAGIC         # Note: This chain does not compress the history, so very long converastions can overflow the context window.
# MAGIC         MessagesPlaceholder(variable_name="formatted_chat_history"),
# MAGIC         # User's most current question
# MAGIC         ("user", "{copy}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # Format the converastion history to fit into the prompt template above.
# MAGIC def format_chat_history_for_prompt(chat_messages_array):
# MAGIC     history = extract_chat_history(chat_messages_array)
# MAGIC     formatted_chat_history = []
# MAGIC     if len(history) > 0:
# MAGIC         for chat_message in history:
# MAGIC             if chat_message["role"] == "user":
# MAGIC                 formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
# MAGIC             elif chat_message["role"] == "assistant":
# MAGIC                 formatted_chat_history.append(AIMessage(content=chat_message["content"]))
# MAGIC     return formatted_chat_history
# MAGIC
# MAGIC
# MAGIC # # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "copy": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
# MAGIC         "formatted_chat_history": itemgetter("messages")
# MAGIC         | RunnableLambda(format_chat_history_for_prompt),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC ## Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

import os
# Log the model to MLflow
with mlflow.start_run(run_name=f"dbdemos_review_no_rag"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain_no_rag.py'),  #Chain code file e.g., /path/to/the/chain.py 
        model_config='chain_config_no_rag.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        pip_requirements="requirements.txt",
        
    )
# 
# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

MODEL_NAME = "review_app_no_rag_1"
MODEL_NAME_FQN = f"{catalog}.{db}.{MODEL_NAME}"
# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)
uc_registered_model_info

# COMMAND ----------

from databricks import agents

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True, environment_vars= #update this LK below 
                                {
                                   "DATABRICKS_HOST": "{{secrets/max_demo/host_url}}",
                                   "DATABRICKS_TOKEN": "{{secrets/max_demo/api_token}}"
                                  })

instructions_to_reviewer = f"""### Instructions for Testing the our Databricks Documentation Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""


# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)
wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant stakeholders access to the Review App
# MAGIC
# MAGIC Now, grant your stakeholders permissions to use the Review App. To simplify access, stakeholders do not require to have Databricks accounts.

# COMMAND ----------

user_list = []
# Set the permissions.
agents.set_permissions(model_name=MODEL_NAME_FQN, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

deployment_info.review_app_url

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We've seen how we can improve our chatbot, adding more advanced capabilities to handle a chat history.
# MAGIC
# MAGIC As you add capabilities to your model and tune the prompt, it will get harder to evaluate your model performance in a repeatable way.
# MAGIC
# MAGIC Your new prompt might work well for what you tried to fixed, but could also have impact on other questions.
# MAGIC
# MAGIC ## Next: Introducing offline model evaluation with Mosaic AI Agent Evaluation
# MAGIC
# MAGIC To solve these issue, we need a repeatable way of testing our model answer as part of our LLMOps deployment!
# MAGIC
# MAGIC Open the next [03-Offline-Evaluation]($./03-Offline-Evaluation) notebook to discover how to evaluate your model.
