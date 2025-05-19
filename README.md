# Projet de statistiques appliquées 2025 : Analyse de conversations pour établir des panels utilisateurs (Jumbo Mana)

![alt text](https://www.lejournaldesentreprises.com/sites/lejournaldesentreprises.com/files/styles/landscape_web_lg_2x/public/2025-04/Jumbo-Mana-vient-de-dployer-Lisa-son-nouvel-ava-374440.jpeg?h=74596f6a&itok=QwtmNfXl.png)

L’amélioration des services proposés par un organisme public ou privé nécessite l’identification des besoins des utilisateurs. Cerner les préférences des agents permet d’adapter les stratégies préexistantes, voire d’innover afin de favoriser la satisfaction des utilisateurs, contribuant à renforcer la valorisation. Cependant, répondre de façon pertinente aux requêtes des usagers nécessite une analyse précise de leurs motifs de frustration.
Pour résoudre cette difficulté, la start-up française Jumbo Mana, spécialisée dans l’IA générative et l’interface homme-machine, développe des avatars virtuels capables d’interagir de manière fluide et réaliste avec les utilisateurs, enrichissant ainsi leur expérience immersive. Ils sont adaptés à plusieurs secteurs tels que les musées et expositions, le tourisme, la santé, etc.
Ainsi, notre projet, encadré par cette start-up, consiste à établir des panels d’utilisateurs en nous appuyant sur des conversations pour identifier plus clairement leurs besoins, définir des profils types et ajuster ensuite les services et stratégies en conséquence.

## Structure du projet
### Phase 1 : Appropriation de la notion d’embedding
Nous nous concentrons dans un premier temps sur la notion d’embedding, en particulier dans le cadre de l’analyse textuelle de notre étude. Après avoir présenté ce concept, nous cherchons à implémenter un modèle d’embedding "maison" afin de mieux en comprendre le fonctionnement, d’examiner ses limites et surtout de répondre à la question suivante : comment est générée la matrice d’embedding ?
Comme nous le présenterons dans ce rapport, nous utilisons des méthodes issues de l’apprentissage supervisé (k-NN, réseaux de neurones) pour construire nos propres modèles d’embeddings, en précisant leurs limites. Nous les comparerons alors entre-eux et à des modèles plus sophistiqués (BERT) en nous basant sur des méthodes non-supervisées (t-SNE).


### Phase 2 : Classification
Nous optons dans cette partie pour une architecture de réseau de neurones plus adaptée à nos données séquentielles, les RNN. Nous les expliquons succintement avant de les appliquer ensuite à la classification des conversations. Cependant, les limites en terme de mémorisation nous conduirons à utiliser une architecture LSTM. Finalement, nous exploitons des Large Language Models (LLM), comme GPT, afin de compléter notre approche et d’évaluer leur performance sur nos données textuelles. Les LLM sont reconnus pour leur capacité à capter en profondeur le contexte et à gérer les relations complexes entre les documents d’un corpus donné. Dans l’analyse de nos conversations, ils offrent des résultats intéressants, notamment pour la génération des embeddings.

### Structure du repository
Chaque membre du groupe ayant travaillé en priorité sur une tâche spécifique, nous avons séparé ce dépôt en 3 différentes branches :
- La génération d'embeddings avec les méthodes k-NN et MLP est présentée dans la branche de `Titouan`.
- La comparaison d'embeddings avec t-SNE est située dans la branche de `Bilal`.
- La classification de sentiments par LSTM est située dans la branche de `Daniel`.
- Enfin, la génération de profils d'utilisateurs par LLM est proposée dans la branche `LLM-clean`.  



