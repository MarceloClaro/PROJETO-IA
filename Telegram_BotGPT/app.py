import telebot
import os
os.environ["OPENAI_API_KEY"] = "sua chave"
from llama_index import Prompt, VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

bot = telebot.TeleBot('sua chave telegram aqui')

documents = SimpleDirectoryReader('dados').load_data()

TEMPLATE_STR = (
    "Nós fornecemos informações de contexto abaixo.\n"
    "---------------------\n"    
    "Você deverá desempenhar o papel de um vendedor de veículos e atenderá os clientes interessados em obter informações sobre veículos como carro, motos, etc. O seu nome é CyberBotGPT, sou o seu Vendedor Digital. Você deverá somente responder as dúvidas dos clientes usando como base no contexto abaixo que contem informações sobre os veículos vendidos na loja.\n"
    "{context_str}"
    "\n---------------------\n"
    "Com base nessas informações, por favor responda à pergunta.: {query_str}\n"
)
QA_TEMPLATE = Prompt(TEMPLATE_STR)

index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist()
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Olá! Como posso ajudar você?")


@bot.message_handler(func=lambda message: message.text.lower() in ['adeus', 'tchau', 'até logo', 'bye', 'depois eu volto'])
def goodbye(message):
    bot.send_message(message.chat.id, "Até breve!")


@bot.message_handler(content_types=['text'])
def send_text(message):
    answer = query_engine.query(message.text)
    bot.send_message(message.chat.id, answer)

bot.polling()
