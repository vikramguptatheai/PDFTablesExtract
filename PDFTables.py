import subprocess

import streamlit as st
import openai
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
openai.api_key = "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"

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
#sub_p_res = subprocess.run(['apt', 'install', 'ghostscript', 'python3-t'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>



sub_p_res = subprocess.run(['pip', 'install', 'camelot-py', 'pymupdf', 'frontend', 'ghostscrip'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>


import camelot
from llama_index import Document, SummaryIndex

# https://en.wikipedia.org/wiki/The_World%27s_Billionaires
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI

from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from pathlib import Path
from typing import List

llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=-1)
service_context = ServiceContext.from_defaults(llm=llm)
#from PDFComparisions import base_engine, s_engine
sub_p_res = subprocess.run(['wget', '"https://www.dropbox.com/scl/fi/waoz9bo9yiemnhnqvu0cc/billionaires_page.pdf?rlkey=4i08msa7zr1lpnuq2y1vs2xgw&dl=1"', '-O', 'billionaires_page.pd'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
# initialize PDF reader
reader = PyMuPDFReader()
file_path = "billionaires_page.pdf"
docs = reader.load(file_path=file_path)

# use camelot to parse tables
def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs


table_dfs = get_tables(file_path, pages=[3, 7])
# shows list of top billionaires in 2023
table_dfs[0]
# define query engines over these tables
df_query_engines = [PandasQueryEngine(table_df) for table_df in table_dfs]


response = df_query_engines[0].query(
    "What's the net worth of the second richest billionaire in 2023?"
)

response = df_query_engines[1].query("How many billionaires were there in 2009?")
print(str(response))




# setup baseline table Enginer]
def loadBaseEngine():
    vector_index0 = VectorStoreIndex(doc_nodes)
    vector_query_engine0 = vector_index0.as_query_engine()

    engine_to_use=vector_query_engine0
    return engine_to_use



"""
### Build Recursive Retriever

We define a top-level vector index that does top-k lookup over a set of Nodes. We define two special nodes (`IndexNode` objects) linking to each of these tables.

We define a `RecursiveRetriever` object to recursively retrieve/query nodes. We then put this in our `RetrieverQueryEngine` along with a `ResponseSynthesizer` to synthesize a response.

We pass in mappings from id to retriever and id to query engine. We then pass in a root id representing the retriever we query first.
"""


def loadRrecursiveTableEngine():
    llm = OpenAI(temperature=0, model="gpt-4")

    service_context = ServiceContext.from_defaults(
        llm=llm,
    )
    doc_nodes = service_context.node_parser.get_nodes_from_documents(docs)
    summaries = [
    "This node provides information about the world's richest billionaires in 2023",
    "This node provides information on the number of billionaires and their combined net worth from 2000 to 2023.",
    ]
    df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}")
    for idx, summary in enumerate(summaries)
    ]

    df_id_query_engine_mapping = {
        f"pandas{idx}": df_query_engine
        for idx, df_query_engine in enumerate(df_query_engines)
    }

    # construct top-level vector index + query engine
    vector_index = VectorStoreIndex(doc_nodes + df_nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=1)

    from llama_index.retrievers import RecursiveRetriever
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.response_synthesizers import get_response_synthesizer


    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        query_engine_dict=df_id_query_engine_mapping,
        verbose=True,
    )

    response_synthesizer = get_response_synthesizer(
        # service_context=service_context,
        response_mode="compact"
    )

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, response_synthesizer=response_synthesizer
    )



    engine_to_use=query_engine
    return engine_to_use



if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about Documents's research!"}]


def main():
    #setup engine for base or advance RAG

    st.header("CGI VishnuAI Documents Table Extraction Framework")
    st.sidebar.info("Select standard  or CGI Advance Recursive Engine")
    engine_to_use=st.sidebar.selectbox('steps',['Standard','CGIRecursive'])
    st.sidebar.info("Question to try to see the value of CGI Advance approach")
    st.sidebar.info("How many billionaires were there in 2009?")
    st.sidebar.info("What is the average age of top 5 billionaires in 2023? Make sure age is a float.")

    if prompt := st.chat_input("Please ask your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if (engine_to_use=='Standard'):
                    engine_to_use=loadBaseEngine()
                elif(engine_to_use=='CGIRecursive'):
                    engine_to_use=loadRrecursiveTableEngine()
                else:
                    engine_to_use=loadBaseEngine()
                response = engine_to_use.query(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history

if __name__ == "__main__":
    main()