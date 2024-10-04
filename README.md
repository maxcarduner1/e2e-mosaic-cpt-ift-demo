# End-to-end Fine-Tuning of LLMs to create customer Tone/Voice chatbot

In this demo, we provide notebooks to aid in the process of building a tone/brand chatbot using the Mosaic AI Model Training stack on Databricks. High level steps are as follows:

1. Prepare data for continued pre-training (CPT)
2. Perform CPT (unsupervised)
3. Prepare data for Instruction Fine-Tuning (IFT) by creating synthetic outlines of given articles (you will need to provide these articles)
4. Run IFT (will need to spin up endpoint)
5. Evaluate
6. Plug into Review App notebooks to spin up a review app with your final model endpoint
