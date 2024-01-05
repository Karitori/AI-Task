from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import openai
import os
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv('OPENAI_API_BASE')
OPENAI_DEPLOYMENT_NAME = os.getenv('OPENAI_DEPLOYMENT_NAME')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_DEPLOYMENT_VERSION = os.getenv('OPENAI_DEPLOYMENT_VERSION')

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv('OPENAI_EMBEDDING_DEPLOYMENT_NAME')
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    pdf_path = os.path.join("e:\\Coding\\Projects\\PythonProjects\\AI-Task\\uploaded_pdfs\\", request.json['fileName'])
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    embeddings=OpenAIEmbeddings(deployment= OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type='azure',
                                chunk_size=1)
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    db.save_local('.dbs/faiss_index')

    return jsonify({'message': 'Embeddings created and saved successfully'}), 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    if question:
        chat_history = request.json.get('chat_history', [])


        openai.api_type = 'azure'
        openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
        openai.api_key = OPENAI_API_KEY
        openai.api_version = OPENAI_DEPLOYMENT_VERSION

        llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                        model_name=OPENAI_MODEL_NAME,
                        openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                        openai_api_key=OPENAI_API_KEY,
                        openai_api_type='azure')

        embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                    model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                    openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                    openai_api_type='azure',
                                    chunk_size=1)

        vectorStore = FAISS.load_local('.dbs/faiss_index', embeddings)

        retriever = vectorStore.as_retriever(search_type='similarity', search_kwargs={'k': 2})

        QUESTION_PROMPT = PromptTemplate.from_template('Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.')

        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                retriever=retriever,
                                                condense_question_prompt=QUESTION_PROMPT,
                                                return_source_documents=True,
                                                verbose=False)

        result = qa({'question': question, 'chat_history': chat_history})
        return jsonify({'answer': result['answer']}), 200
    else:
        return jsonify({'error': 'No question provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)