from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
import langchain_community
import pandas as pd
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
import faiss
import numpy as np
import streamlit as st
import tiktoken

# Load environment variables
load_dotenv()
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# Embedding işlemi için OpenAI Embedding fonksiyonunu tanımla
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Dosyayı sadece bir kez yükleyip, tekrar tekrar yüklenmesini önleyelim
if 'df' not in st.session_state:
    file_path = 'documents2.xlsx'
    xls = pd.ExcelFile(file_path)
    st.session_state['df'] = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

df = st.session_state['df']

def combine_product_info_all_columns(df):
    documents = []
    for idx, row in df.iterrows():
        # Tüm ürün bilgilerini metin olarak birleştiriyoruz
        product_description = "\n".join([f"{col}: {row[col]}" for col in df.columns])

        metadata = {col: row[col] for col in df.columns}

        # Document oluşturup listeye ekliyoruz
        documents.append(Document(page_content=product_description, metadata=metadata))

    return documents

# FAISS ve embedding'leri sadece bir kez oluşturalım
if 'faiss_index' not in st.session_state:
    # Apply the function to the dataframe with all columns
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

    embedded_documents = np.array(embedded_documents, dtype=np.float32)

    faiss.normalize_L2(embedded_documents)

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embedded_documents)

    st.session_state['faiss_index'] = faiss_index
    st.session_state['metadata'] = metadata
else:
    faiss_index = st.session_state['faiss_index']
    metadata = st.session_state['metadata']


# FAISS ile sorgu yapma
def search_faiss(query, index, k=5):
    query_embedding = embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, k)

    results = [metadata[i] for i in indices[0]]
    return results


# Bellek modülü ile önceki yanıtların kaydedilmesi
memory = ConversationBufferWindowMemory(k=5)

# Bellek güncelleme fonksiyonu
def update_memory_with_response(query, response, memory):
    memory.save_context({"input": query}, {"output": response})

# GPT-3.5 ile Soru Cevaplama Yapısı
PROMPT_TEMPLATE = """
Sen bir müşteri hizmetleri temsilcisi gibi davran ve aşağıdaki ürün bilgilerini kullanarak soruları cevapla:

{context}

---

Müşteriye soruları yanıtlarken şu adımları izle:
1. Eğer bir ürün hakkında bilgi veriyorsan, daha fazla ayrıntı isteyip istemediğini sor (örn: renk, marka, fiyat aralığı).
2. Eğer ürün seçenekleri genişse, müşteriye maksimum 2 ürün öner.
3. Yanıtların samimi ve kullanıcı dostu olsun. Örneğin: "Evet, elimizde beyaz buzdolabı var. Tercih ettiğiniz bir marka var mı?"
4. Eğer soru buzdolabı veya buzlukta saklanabilecek yiyeceklerle ilgiliyse, şu yanıtları kullan:
   - Buzdolabında genellikle süt ürünleri, sebzeler, meyveler, pişmiş yemekler ve içecekler saklanır. 
   - Et, balık ve tavuk gibi çiğ gıdalar genellikle buzdolabının alt rafında saklanmalıdır.
   - Buzlukta ise dondurulmuş sebzeler, dondurma, et, balık, tavuk ve buz saklanabilir. Dondurulmuş gıdalar uzun süre saklanabilir ve gerektiğinde çözdürülüp kullanılabilir.
5. Meta datalara iyi odaklan, kullanıcının sorusuna en doğru cevabı vermeye çalış ve dinamik yapıdan kopma. Ürünler hakkında detaylı açıklamalar ver.
6. Neden bu ürünü tercih etmeliyim? Neden Bu ürünü almalıyım gibi sorulara ikna edici cevaplar vermelisin

---

Soru: {question}
"""


def generate_response_with_gpt(context_text, query_text, openai_api_key):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    previous_context = memory.load_memory_variables({})["history"]
    prompt = prompt_template.format(context=previous_context + context_text, question=query_text)
    model = ChatOpenAI(openai_api_key=openai_api_key)
    response_text = model.predict(prompt)
    update_memory_with_response(query_text, response_text, memory)
    return response_text


# RAG işlemi: FAISS sonuçlarını alıp GPT'ye bağlam olarak veriyoruz
def search_and_generate_response(query, faiss_index, openai_api_key):
    results = search_faiss(query, faiss_index, k=5)

    retrieved_context = "\n\n".join([
        "\n".join([f"{key}: {value}" for key, value in result.items()])
        for result in results
    ])

    response_text = generate_response_with_gpt(retrieved_context, query, openai_api_key)

    update_memory_with_response(query, response_text, memory)

    return response_text


# Streamlit uygulaması
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='chat_form'):
    query_text = st.text_input("Merhaba! Ben asistanınız Buzi. Buzdolapları hakkında size bilgi verebilirim.:")
    submit_button = st.form_submit_button(label='Gönder')

if submit_button and query_text:
    st.session_state['messages'].append({"role": "user", "content": query_text})

    response_text = search_and_generate_response(query_text, faiss_index, openai_api_key)

    st.session_state['messages'].append({"role": "bot", "content": response_text})

# Mesajları göstermek
if st.session_state['messages']:
    for i in range(0, len(st.session_state['messages']), 2):
        with st.container():
            if i < len(st.session_state['messages']) and st.session_state['messages'][i]["role"] == "user":
                st.markdown(f"Kullanıcı: {st.session_state['messages'][i]['content']}")
            if i + 1 < len(st.session_state['messages']) and st.session_state['messages'][i + 1]["role"] == "bot":
                st.markdown(f"Buzi: {st.session_state['messages'][i + 1]['content']}")

        st.markdown("---")
