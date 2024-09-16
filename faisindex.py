from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
import pandas as pd
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
import faiss
import numpy as np
import streamlit as st
from langchain.prompts.chat import MessagesPlaceholder
from langchain.chains import LLMChain

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)

if 'df' not in st.session_state:
    file_path = 'documents2.xlsx'
    xls = pd.ExcelFile(file_path)
    st.session_state['df'] = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

df = st.session_state['df']

def combine_product_info_all_columns(df):
    documents = []
    for idx, row in df.iterrows():
        product_description = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        metadata = {col: row[col] for col in df.columns}
        documents.append(Document(page_content=product_description, metadata=metadata))
    return documents

if 'faiss_index' not in st.session_state:
    complete_documents = combine_product_info_all_columns(df)

    def split_text(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=False
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    chunked_documents = split_text(complete_documents)

    def embed_product_text(chunked_documents):
        embeddings = []
        metadata = []
        for chunk in chunked_documents:
            product_chunk = chunk.page_content
            embedding = embedding_function.embed_query(product_chunk)
            embeddings.append(embedding)
            metadata.append(chunk.metadata)
        return np.array(embeddings), metadata

    embedded_documents, metadata = embed_product_text(chunked_documents)
    embedded_documents = np.array(embedded_documents, dtype=np.float32)

    embedding_dim = embedded_documents.shape[1]

    faiss_index = faiss.IndexFlatL2(embedding_dim)

    faiss.normalize_L2(embedded_documents)

    faiss_index.add(embedded_documents)

    st.session_state['faiss_index'] = faiss_index
    st.session_state['metadata'] = metadata
else:
    faiss_index = st.session_state['faiss_index']
    metadata = st.session_state['metadata']

def search_faiss(query, index, k=10):
    query_embedding = embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding]).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    results = [metadata[i] for i in indices[0]]
    return results

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(
        k=100,
        memory_key="history",
        input_key="input",
        return_messages=True
    )

memory = st.session_state['memory']

if 'recommended_products' not in st.session_state:
    st.session_state['recommended_products'] = None

PROMPT_TEMPLATE = """
Sen bir müşteri hizmetleri temsilcisi gibi davran ve aşağıdaki ürün bilgisini kullanarak soruları cevapla:

{context}

---

Sohbet geçmişi:

{history}

---

Müşteriye soruları yanıtlarken şu adımları izle:
1. Eğer bir ürün önerdiysen ve müşteri bu ürün hakkında soru soruyorsa, önceki önerdiğin ürüne göre cevap ver.
2. Eğer yeni bir ürün talebi varsa, FAISS indeksinden uygun ürünü bul ve öner.
3. Yanıtların samimi ve kullanıcı dostu olsun. Örneğin: "Önerdiğim buzdolabının fiyatı 5000 TL'dir."
4. Müşterinin sorularına en doğru ve ilgili cevabı vermeye çalış.

---

Müşteri sorusu: {input}
"""

def generate_response_with_gpt(context_text, query_text, openai_api_key):
    system_message_prompt = SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATE)

    chat_prompt = ChatPromptTemplate(
        input_variables=["context", "input", "history"],
        messages=[
            system_message_prompt,
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    chain = LLMChain(llm=model, prompt=chat_prompt, memory=memory)
    inputs = {'context': context_text, 'input': query_text}
    response_text = chain.run(inputs)
    return response_text

def search_and_generate_response(query, faiss_index, openai_api_key):
    results = search_faiss(query, faiss_index, k=3)
    st.session_state['recommended_products'] = results

    retrieved_context = "\n\n".join([
        "\n".join([f"{key}: {value}" for key, value in result.items()])
        for result in results
    ])

    response_text = generate_response_with_gpt(retrieved_context, query, openai_api_key)
    return response_text

st.title("Buzi - Buzdolabı Asistanı")

if st.button("ARAMA GEÇMİŞİNİ SIFIRLA"):
    st.session_state['recommended_products'] = None
    st.session_state['memory'].clear()
    st.session_state['messages'] = []
    st.success("Sohbet geçmişi başarıyla sıfırlandı.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if prompt := st.chat_input("Merhaba! Ben asistanınız Buzi. Buzdolapları hakkında size bilgi verebilirim."):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response_text = search_and_generate_response(prompt, faiss_index, openai_api_key)
    st.session_state['messages'].append({"role": "bot", "content": response_text})

for message in st.session_state['messages']:
    role = message['role']
    content = message['content']
    if role == 'user':
        with st.chat_message("user"):
            st.markdown(content)
    elif role == 'bot':
        with st.chat_message("assistant"):
            st.markdown(content)
