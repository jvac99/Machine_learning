# O Cenário do Aprendizado de Máquina

## Mãos à obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow - Aurélien Géron

### O que é o Aprendizado de Máquina?

* Aprendizado de Máquina é a ciência (e a arte) da programação de computadores para que eles possam aprender com os dados.

* [Aprendizado de Máquina é o] campo de estudo que dá aos computadores a habilidade de aprender sem ser explicitamente programado. — Arthur Samuel, 1959.

* Os exemplos utilizados pelo sistema para o aprendizado são chamados de conjuntos de treinamentos. Cada exemplo de treinamento é chamado de instância de treinamento (ou amostra).

* Aplicar técnicas do AM para se aprofundar em grandes quantidades de dados pode ajudar na descoberta de padrões que não eram aparentes. Isto é chamado de mineração de dados.

### Tipos de Sistemas do Aprendizado de Máquina

* Serem ou não treinados com supervisão humana (supervisionado, não supervisionado, semissupervisionado e aprendizado por reforço);

* Se podem ou não aprender rapidamente, de forma incremental (aprendizado online versus aprendizado por lotes);

* Se funcionam simplesmente comparando novos pontos de dados com pontos de dados conhecidos, ou se detectam padrões em dados de treinamento e criam um modelo preditivo, como os cientistas (aprendizado baseado em instâncias versus
aprendizado baseado em modelo).

### Aprendizado Supervisionado/Não Supervisionado

    Os sistemas de Aprendizado de Máquina podem ser classificados de acordo com a quantidade e o tipo de supervisão que recebem durante o treinamento. Existem quatro categorias principais de aprendizado: supervisionado, não supervisionado, semissupervisionado e por reforço.

#### Aprendizado Supervisionado

    No aprendizado supervisionado, os dados de treinamento que você fornece ao algoritmo incluem as soluções desejadas, chamadas de rótulos

    A classificação é uma tarefa típica do aprendizado supervisionado. O filtro de spam é um bom exemplo disso: ele é treinado com muitos exemplos de e-mails junto às classes (spam ou não spam) e deve aprender a classificar novos e-mails.

    Prever um alvo de valor numérico é outra tarefa típica, como o preço de um carro a partir de um conjunto de características (quilometragem, idade, marca, etc.) denominadas previsores. Esse tipo de tarefa é chamada de regressão (Figura 1-6)1 . Para treinar o sistema, você precisa fornecer muitos exemplos de carros incluindo seus previsores e seus labels (ou seja, seus preços).

Algoritmos mais importantes do aprendizado supervisionado

* k-Nearest Neighbours
* Regressão Linear
* Regressão Logística
* Máquinas de Vetores de Suporte (SVM)
* Árvores de Decisão e Florestas Aleatórias
* Redes Neurais

#### Aprendizado Não Supervisionado

    No aprendizado não supervisionado, os dados de treinamento não são rotulados. O sistema tenta aprender sem um professor.

Algoritmos mais importantes do aprendizado não supervisionado

* Clustering
    * k-Means
    * Clustering Hierárquico [HCA, do inglês]
    * Maximização da Expectativa

* Visualização e redução da dimensionalidade
    * Análise de Componentes Principais [PCA, do inglês]
    * Kernel PCA
    * Locally-Linear Embedding (LLE)
    * t-distributed Stochastic Neighbor Embedding (t-SNE)

* Aprendizado da regra da associação
    * Apriori
    * Eclat

Extração de características

    A redução da dimensionalidade é uma tarefa relacionada na qual o objetivo é simplificar os dados sem perder muita informação. Uma maneira de fazer isso é mesclar várias características correlacionadas em uma. Por exemplo, a quilometragem de um carro pode estar muito correlacionada com seu tempo de uso, de modo que o algoritmo da redução de dimensionalidade irá mesclá-los em uma característica que representa o desgaste do carro. Isso é chamado de extração de características.

    A detecção de anomalias: o sistema é treinado com instâncias normais e, quando vê uma nova instância, pode dizer se ela parece normal ou se é uma provável anomalia. Exemplos: a detecção de transações incomuns em cartões de crédito para evitar fraudes, detectar defeitos de fabricação ou remover automaticamente outliers de um conjunto de dados antes de fornecê-lo a outro algoritmo de aprendizado.
    
    Aprendizado de regras de associação, cujo objetivo é se aprofundar em grandes quantidades de dados e descobrir relações interessantes entre atributos. Por exemplo, suponha que você possua um supermercado. Executar uma regra de associação em seus registros de vendas pode revelar que as pessoas que compram molho de churrasco e batatas fritas também tendem a comprar carnes. Desta forma, você vai querer colocar esses itens próximos uns dos outros. 

#### Aprendizado Semi-supervisionado

    Alguns algoritmos podem lidar com dados de treinamento parcialmente rotulados, uma grande quantidade de dados não rotulados e um pouco de dados rotulados.

    A maior parte dos algoritmos de aprendizado semissupervisionado é de combinações de algoritmos supervisionados e não supervisionados. Por exemplo, as redes neurais de crenças profundas [DBNs, do inglês] são baseadas em componentes não supervisionados, chamados máquinas restritas de Boltzmann [RBMs, do inglês], empilhados uns em cima dos outros. As RBMs são treinadas sequencialmente de forma não supervisionada, e então todo o sistema é ajustado utilizando-se técnicas de aprendizado supervisionado.

#### Aprendizado por Reforço

    O sistema de aprendizado, chamado de agente nesse contexto, pode observar o ambiente, selecionar e executar ações e obter recompensas em troca — ou penalidades na forma de recompensas negativas. Ele deve aprender por si só qual é a melhor estratégia, chamada de política, para obter o maior número de recompensas ao longo do tempo. Uma política define qual ação o agente deve escolher quando está em determinada situação.