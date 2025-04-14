import os
import json
import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate

# Chargement des variables d’environnement depuis .env
load_dotenv()

# Chargement du prompt depuis le YAML
def load_yaml_prompt(path: str, key: str = "prompt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content[key]

# Charger le prompt
BENCH_PROMPT = load_yaml_prompt(
    "/Users/bilalramadan/Statap-panels/Statap-panels-1/llm-dev/bench_prompt.yaml"
)

# Création du prompt LangChain
prompt_template = ChatPromptTemplate.from_template(BENCH_PROMPT)
print("Variables attendues :", prompt_template.input_variables)

# Initialisation du modèle Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    temperature=0.0,
)

# Envoi d'un message comme dans la doc officielle
messages = [
    SystemMessage(content="Tu es un expert en Parfumerie de luxe, répond comme ci-tu travaillais pour la maison Louis Vuitton."),
    HumanMessage(content="Présentez moi les différents parfums que vous proposez?")
]


# Obtenir la réponse
response = chat.invoke(messages)
print(response.content)
