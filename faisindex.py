import os
import re
import faiss
import openai
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
    MessagesPlaceholder
)

load_dotenv()
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)

if 'df' not in st.session_state:
    df = pd.read_excel('documents2.xlsx')
    st.session_state['df'] = df
else:
    df = st.session_state['df']

def prepare_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        metadata = row.to_dict()
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def create_faiss_index(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embeddings = [embedding_function.embed_query(chunk.page_content) for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata

# İndeks ve metadata dosyalarının yolları
INDEX_PATH = 'faiss_index.bin'
METADATA_PATH = 'metadata.pkl'

if 'faiss_index' not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        # İndeksi ve metadataları diskten yükleyin
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        st.session_state['faiss_index'] = faiss_index
        st.session_state['metadata'] = metadata
    else:
        # İndeksi ve metadataları oluşturun
        documents = prepare_documents(df)
        faiss_index, metadata = create_faiss_index(documents)
        # İndeksi ve metadataları disk üzerinde kaydedin
        faiss.write_index(faiss_index, INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        st.session_state['faiss_index'] = faiss_index
        st.session_state['metadata'] = metadata
else:
    faiss_index = st.session_state['faiss_index']
    metadata = st.session_state['metadata']

def search_faiss(query, k=10):
    embedding = embedding_function.embed_query(query)
    embedding = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(embedding)
    _, indices = faiss_index.search(embedding, k)
    return [metadata[i] for i in indices[0]]

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(
        k=100, memory_key="history", input_key="input", return_messages=True
    )
memory = st.session_state['memory']

PROMPT_TEMPLATE = """
Sen bir müşteri hizmetleri temsilcisi gibi davran ve aşağıdaki ürün bilgisini kullanarak soruları cevapla:

{context}

---

Sohbet geçmişi:

{history}

---

Müşteriye soruları yanıtlarken şu adımları izle:
1. Eğer bir ürün önerdiysen ve müşteri bu ürün hakkında soru soruyorsa, önceki önerdiğin ürüne göre cevap ver.
2. Eğer müşteri en ucuz ürünü istiyorsa, ürünleri fiyatına göre sıralayıp en ucuz ürünü öner.
3. Eğer yeni bir ürün talebi varsa, FAISS indeksinden uygun ürünü bul ve öner.
4. Eğer müşteri bir bütçe belirtmişse ve uygun ürün yoksa, mevcut ürünlerin en düşük ve en yüksek fiyatlarını bildir ve alternatifler sun.
5. Yanıtların samimi ve kullanıcı dostu olsun. Örneğin: "Önerdiğim buzdolabının fiyatı 7000 TL'dir."
6. Müşterinin sorularına en doğru ve ilgili cevabı vermeye çalış.

---

Müşteri sorusu: {input}
"""

def extract_budget_and_intent(query_text):
    model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    prompt = ChatPromptTemplate.from_template("""
Müşterinin aşağıdaki sorusundan bütçe bilgisini ve en ucuz ürünü isteyip istemediğini çıkart.
Eğer bütçe belirtilmemişse 'None' yaz.
Eğer müşteri en ucuz ürünü istiyorsa 'evet' yaz, istemiyorsa 'hayır' yaz.

Soru: {query_text}

Bütçe (sadece sayı olarak, TL cinsinden): [Bütçe]
En ucuz ürünü istiyor mu? [evet/hayır]
""")
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.run({'query_text': query_text}).strip().lower()
    budget_match = re.search(r"bütçe.*\[(.*?)\]", result)
    intent_match = re.search(r"en ucuz.*\[(.*?)\]", result)
    budget = re.sub(r'[^\d]', '', budget_match.group(1)) if budget_match else None
    budget = float(budget) if budget else None
    wants_cheapest = intent_match.group(1) == 'evet' if intent_match else False
    return budget, wants_cheapest

def generate_response(context_text, query_text):
    system_prompt = SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_prompt = ChatPromptTemplate(
        input_variables=["context", "input", "history"],
        messages=[
            system_prompt,
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    chain = LLMChain(llm=model, prompt=chat_prompt, memory=memory)
    return chain.run({'context': context_text, 'input': query_text})

def search_and_generate_response(query):
    budget, wants_cheapest = extract_budget_and_intent(query)
    results = search_faiss(query, k=10)
    all_prices = []
    for item in metadata:
        price = re.sub(r'[^\d]', '', str(item.get('Fiyat', '')))
        if price:
            all_prices.append((float(price), item))
    min_price_item = min(all_prices, key=lambda x: x[0])[1] if all_prices else None
    min_price = min(all_prices, key=lambda x: x[0])[0] if all_prices else None
    max_price = max(all_prices, key=lambda x: x[0])[0] if all_prices else None

    if wants_cheapest and min_price_item:
        st.session_state['recommended_products'] = [min_price_item]
        context = "\n".join([f"{k}: {v}" for k, v in min_price_item.items()])
        return generate_response(context, query)

    if budget is not None:
        filtered = []
        for item in results:
            price = re.sub(r'[^\d]', '', str(item.get('Fiyat', '')))
            if price and float(price) <= budget:
                filtered.append(item)
        if not filtered:
            if min_price and max_price:
                return f"Üzgünüm, {budget} TL bütçeyle uygun bir buzdolabı bulamadım. Mevcut buzdolaplarımızın fiyatları {min_price} TL ile {max_price} TL arasında değişmektedir."
            else:
                return "Üzgünüm, şu anda elimizde ürün bulunmamaktadır."
        st.session_state['recommended_products'] = filtered
    else:
        st.session_state['recommended_products'] = results

    context = "\n\n".join([
        "\n".join([f"{k}: {v}" for k, v in item.items()])
        for item in st.session_state['recommended_products'][:3]
    ])
    return generate_response(context, query)

st.title("Buzi - Buzdolabı Asistanı")

if st.button("ARAMA GEÇMİŞİNİ SIFIRLA"):
    st.session_state['recommended_products'] = None
    memory.clear()
    st.session_state['messages'] = []
    st.success("Sohbet geçmişi başarıyla sıfırlandı.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if prompt := st.chat_input("Merhaba! Ben asistanınız Buzi. Buzdolapları hakkında size bilgi verebilirim."):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = search_and_generate_response(prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
