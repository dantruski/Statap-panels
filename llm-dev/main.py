import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Charge le .env à jour et écrase les éventuelles anciennes valeurs
load_dotenv(override=True)

# Vérification de la présence des variables d'environnement
required_vars = [
    "OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_DEPLOYMENT_NAME",
    "OPENAI_API_VERSION",
]

for var in required_vars:
    value = os.getenv(var)
    if not value:
        raise ValueError(f"❌ Variable {var} manquante dans le .env")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"❌ Variable {var} contient un caractère non-ASCII : {value}")

# Instanciation du modèle Azure via LangChain avec les bons paramètres
chat = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Envoi d'un message comme dans la doc officielle
messages = [
    SystemMessage(content="Tu es un expert en Parfumerie de luxe, répond comme ci-tu travaillais pour la maison Louis Vuitton."),
    HumanMessage(content="Présentez moi les différents parfums que vous proposez?")
]



# Obtenir la réponse
response = chat.invoke(messages)
print(response.content)
