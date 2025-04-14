import os
import json
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate

# Chargement des variables d’environnement depuis .env
load_dotenv()

# Définition du modèle de sortie attendu
class RagScore(BaseModel):
    score: float
    explanation: str



# Charger le contenu du fichier YAML (prompt template)
def load_yaml_prompt(path: str, key: str = "prompt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content[key]


BENCH_PROMPT = load_yaml_prompt(
    "/Users/bilalramadan/Statap-panels/Statap-panels-1/llm-dev/bench_prompt.yaml"
)

# Initialisation du modèle Azure
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)


# Création du parser JSON basé sur le modèle Pydantic
parser = JsonOutputParser(pydantic_object=RagScore)

# Pipeline complet
prompt_template = ChatPromptTemplate.from_template(BENCH_PROMPT)
chain = prompt_template | llm | parser
print("Variables attendues :", prompt_template.input_variables)


# Charger le dataset
with open("/Users/bilalramadan/Statap-panels/Statap-panels-1/llm-dev/temp_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Évaluation de chaque exemple
results = []
for entry in dataset:
    context = {
        "question": entry["query_modifier_question"],
        "generated_answer": entry["generated_answer"],
        "answer": entry["answer"]
    }
    try:
        evaluation = chain.invoke(context)
        print(type(evaluation))  # devrait afficher <class 'dict'> si c'est mal configuré
        entry["score"] = evaluation.get("score")
        entry["explanation"] = evaluation.get("explanation")
    except Exception as e:
        entry["score"] = None
        entry["explanation"] = f"Erreur: {str(e)}"
    results.append(entry)

# Sauvegarde du résultat
output_path = "/Users/bilalramadan/Statap-panels/Statap-panels-1/llm-dev/evaluation_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Résultats enregistrés dans {output_path}")



