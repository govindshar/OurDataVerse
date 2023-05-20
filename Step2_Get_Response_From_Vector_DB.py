import os
import openai
from dotenv import load_dotenv
#from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from llama_index import  PromptHelper

from langchain.chat_models import AzureChatOpenAI

from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from llama_index import ServiceContext,StorageContext, load_index_from_storage
#python3.7 -m pip install -U sqlalchemyfrom flask_sqlalchemy import SQLAlchemy


# Load env variables (create .env with OPENAI_API_KEY and OPENAI_API_BASE)
load_dotenv()
openai_api_version = "2023-03-15-preview"
deployment_name = "gpt-35-turbo"

#deployment_name = "text-davinci-003"
# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = openai_api_version
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

#from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader


# Create LLM via Azure OpenAI Service
#llm = AzureOpenAI(temperature=0.1, model_name =deployment_name)
llm = AzureChatOpenAI(deployment_name=deployment_name,temperature=0.2,
                      openai_api_version=openai_api_version)

llm_predictor = ChatGPTLLMPredictor(llm=llm, retry_on_throttling=False)
embedding_llm = LangchainEmbedding(OpenAIEmbeddings)


# Define prompt helper
max_input_size = 3000
num_output = 256
chunk_size_limit = 1000
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size=max_input_size, 
                             num_output=num_output, 
                             max_chunk_overlap=max_chunk_overlap, 
                             chunk_size_limit=chunk_size_limit)

# Define service_context 
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embedding_llm, prompt_helper=prompt_helper)



# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context,service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context, verbose=False)


while (True==True):
    query=input("\n Please enter your keywords ")
    answer = query_engine.query(query)
    print(answer)
