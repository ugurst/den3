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
        # Belgeyi anlamlı bir bütün olarak bölmek için daha büyük bir chunk_size belirleyin
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Her chunk'ın maksimum karakter boyutunu 1000 olarak belirledim
            chunk_overlap=100,  # Çakışmayı 100 karakter yaparak bağlam bütünlüğünü koruyabiliriz
            length_function=len,
            add_start_index=False  # Her satır tek başına olduğu için start index eklemeye gerek yok
        )

        chunks = text_splitter.split_documents(documents)
        print(f"{len(documents)} belge {len(chunks)} parçaya bölündü.")
        return chunks

    chunked_documents = split_text(complete_documents)

    def embed_product_text(chunked_documents):
        embeddings = []
        metadata = []
        for chunk in chunked_documents:
            product_chunk = chunk.page_content  # Metin parçası
            embedding = embedding_function.embed_query(product_chunk)  # Embedding işlemi
            embeddings.append(embedding)
            metadata.append(chunk.metadata)  # Metadata bilgilerini saklıyoruz
        return np.array(embeddings), metadata

    # Embedding işlemi ve metadata alma
    embedded_documents, metadata = embed_product_text(chunked_documents)
    embedded_documents = np.array(embedded_documents, dtype=np.float32)

    # Embedding boyutunu alın
    embedding_dim = embedded_documents.shape[1]

    faiss_index = faiss.IndexFlatL2(embedding_dim)

    embedded_documents = np.array(embedded_documents, dtype=np.float32)

    faiss.normalize_L2(embedded_documents)

    # FAISS indexi oluştur
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embedded_documents)  # Embedding'leri FAISS indexine ekliyoruz

    # FAISS indexini ve metadata'yı session_state'de sakla
    st.session_state['faiss_index'] = faiss_index
    st.session_state['metadata'] = metadata
else:
    faiss_index = st.session_state['faiss_index']
    metadata = st.session_state['metadata']


# FAISS ile sorgu yapma
def search_faiss(query, index, k=5):
    query_embedding = embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding])

    # FAISS ile en yakın k komşuyu bul
    distances, indices = index.search(query_embedding, k)

    # Bulunan sonuçları metadata ile eşleştirip geri döndür
    results = [metadata[i] for i in indices[0]]
    return results


# RAG işlemi: FAISS sonuçlarını alıp GPT'ye bağlam olarak veriyoruz
def search_and_generate_response(query, faiss_index, openai_api_key):
    # FAISS ile ürünü ara
    results = search_faiss(query, faiss_index, k=5)

    # FAISS sonuçlarını bağlama dönüştür (tüm metadata'yı al)
    retrieved_context = "\n\n".join([
        "\n".join([f"{key}: {value}" for key, value in result.items()])  # Her bir metadata anahtar-değer çiftini alıyoruz
        for result in results
    ])

    # GPT yanıtını al
    response_text = generate_response_with_gpt(retrieved_context, query, openai_api_key)

    return response_text

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
memory = ConversationBufferWindowMemory(k=10)

def generate_response_with_gpt(context_text, query_text, openai_api_key):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Kullanıcı ve bot mesajlarını bağlama dahil ediyoruz
    previous_context = "\n".join([f"Kullanıcı: {msg['content']}" if msg['role'] == "user" else f"Buzi: {msg['content']}" 
                                  for msg in st.session_state['messages']])
    
    # Sohbet geçmişini ve bağlamı kullanarak GPT modeline prompt oluşturuyoruz
    prompt = prompt_template.format(context=previous_context + context_text, question=query_text)
    
    model = ChatOpenAI(openai_api_key=openai_api_key)
    response_text = model.predict(prompt)
    
    memory.save_context({"input": query_text}, {"output": response_text})
    return response_text


# Streamlit uygulaması
# İlk olarak session_state'te 'messages' anahtarını başlatıyoruz
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='chat_form'):
    query_text = st.text_input("Merhaba! Ben asistanınız Buzi. Buzdolapları hakkında size bilgi verebilirim.:")
    submit_button = st.form_submit_button(label='Gönder')

# Eğer form gönderildiyse (Enter'a basıldığında)
if submit_button and query_text:
    # Kullanıcı mesajını session_state'e ekle
    st.session_state['messages'].append({"role": "user", "content": query_text})

    # FAISS ile sorgu yapıp GPT yanıtını alıyoruz
    response_text = search_and_generate_response(query_text, faiss_index, openai_api_key)

    # Bot yanıtını mesajlara ekle
    st.session_state['messages'].append({"role": "bot", "content": response_text})

# Mesajları aşağıdan yukarıya doğru sırayla göstermek için ters çevirme
# Mesajları aşağıdan yukarıya doğru sırayla göstermek için ters çevirme
if st.session_state['messages']:
    for i in range(0, len(st.session_state['messages']), 2):
        with st.container():  # Kullanıcı ve bot mesajlarını bir container içine alıyoruz
            # Kullanıcı mesajı varsa
            if i < len(st.session_state['messages']) and st.session_state['messages'][i]["role"] == "user":
                st.markdown(f"*Kullanıcı:* {st.session_state['messages'][i]['content']}")
            # Bot yanıtı varsa
            if i + 1 < len(st.session_state['messages']) and st.session_state['messages'][i + 1]["role"] == "bot":
                st.markdown(f"*Buzi:* {st.session_state['messages'][i + 1]['content']}")

        st.markdown("---")  # Mesajlar arasında ayırıcı çizgi (isteğe bağlı)

