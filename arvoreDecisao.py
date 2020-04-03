import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
#DADOS ATUAL DA ESTACAO SAO JOAQUIM ANO DE 2016

PATH = "~/Documentos/IA/exemplo/dtgeada/data/saojoaquim_sc/dadosdiarios2016.txt"
#dfDadosDiarios = df = pd.read_csv(PATH, delimiter =";")
tabelaBinaria = df = pd.read_csv(PATH, delimiter =";")

id = tabelaBinaria['Id'].unique()

resp1 = []
for dado in tabelaBinaria['Resp1'].dropna():
    resp1.append(dado)

resp2 = []
for dado in tabelaBinaria['Resp2'].dropna():
    resp2.append(dado)

resp3 = []
for dado in tabelaBinaria['Resp3'].dropna():
    resp3.append(dado)

resp4 = []
for dado in tabelaBinaria['Resp4'].dropna():
    resp4.append(dado)

resp5 = []
for dado in tabelaBinaria['Resp5'].dropna():
    resp5.append(dado)

resp6 = []
for dado in tabelaBinaria['Resp6'].dropna():
    resp6.append(dado)

resp7 = []
for dado in tabelaBinaria['Resp7'].dropna():
    resp7.append(dado)

resp8 = []
for dado in tabelaBinaria['Resp8'].dropna():
    resp8.append(dado)

resp9 = []
for dado in tabelaBinaria['Resp9'].dropna():
    resp9.append(dado)

resp10 = []
for dado in tabelaBinaria['Resp10'].dropna():
    resp10.append(dado)

#d = {'data':data,'tempmin': tempmin,'tempmax':tempmax,'tempmed':tempmed,'um':um, 'ven':ven}
d = {'Id':id,'Resp1': resp1,'Resp2': resp2,'Resp3': resp3,'Resp4': resp4,'Resp5': resp5,'Resp6': resp6,'Resp7': resp7,'Resp8': resp8,'Resp9': resp9,'Resp10': resp10}

dfDadosDiariosFinal = pd.DataFrame(data=d)
#dfDadosDiariosFinal
# DADOS DE GEADA DA ESTACAO SAO JOAQUIM ANO DE 2016

PATH = "~/Documentos/IA/exemplo/dtgeada/data/saojoaquim_sc/geadas/SAOJOAQUIM_SC_83920 _2016.csv"
dfDadosGeada = df = pd.read_csv(PATH, delimiter =";")
#antigo ------------
#dfDadosGeada.sort_values(by=['Data'])

dfDadosGeada

#fraca = []
#moderada = []
#forte= []
iniciante = []
intermediario = []
experiente = []

listageada = dfDadosGeada['Nivelresposta']

i = 0
for id_tabela in dfDadosDiariosFinal['Id']:
    naoachou = True
    for id_tabela2 in dfDadosGeada['ID']:
        if id_tabela == id_tabela2:
            #print listageada[i]
            if listageada[i] == 'INICIANTE':
                #print('FRACA')
                iniciante.append(1)
                intermediario.append(0)
                experiente.append(0)
            if listageada[i] == 'INTERMEDIARIO':
                #print('MODERADA')
                iniciante.append(0)
                intermediario.append(1)
                experiente.append(0)
            if listageada[i] == 'EXPERIENTE':
                #print('FORTE')
                iniciante.append(0)
                intermediario.append(0)
                experiente.append(1)
            naoachou = False
            i=i+1
    if(naoachou):
        iniciante.append(0)
        intermediario.append(0)
        experiente.append(0)
            
            

d = {'nivel_iniciante': iniciante,'nivel_intermediario':intermediario,'nivel_experiente':experiente}

dfgeadaFinal = pd.DataFrame(data=d)
#dfgeadaFinal.to_csv('testegeada')
#Unindo os dois Dataframe

dataframe = pd.concat([dfDadosDiariosFinal, dfgeadaFinal], axis=1)
dataframe.head()
print(dataframe)
features = list(dataframe.columns[1:11])
X = dataframe[features]

target = list(dataframe.columns[11:14])
Y = dataframe[target]

Y

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
#Dados de Irati-PR --  PARA TESTAR A ARVORE - - - -

PATH = "~/Documentos/IA/exemplo/dtgeada/data/irati_pr/dadosdiarios.txt"
dfDadosDiariosPredict = df = pd.read_csv(PATH, delimiter =";")

datapred = dfDadosDiariosPredict['Id'].unique()

resp1 = []
for dado in dfDadosDiariosPredict['Resp1'].dropna():
    resp1.append(dado)

resp2 = []
for dado in dfDadosDiariosPredict['Resp2'].dropna():
    resp2.append(dado)

resp3 = []
for dado in dfDadosDiariosPredict['Resp3'].dropna():
    resp3.append(dado)

resp4 = []
for dado in dfDadosDiariosPredict['Resp4'].dropna():
    resp4.append(dado)

resp5 = []
for dado in dfDadosDiariosPredict['Resp5'].dropna():
    resp5.append(dado)

resp6 = []
for dado in dfDadosDiariosPredict['Resp6'].dropna():
    resp6.append(dado)

resp7 = []
for dado in dfDadosDiariosPredict['Resp7'].dropna():
    resp7.append(dado)

resp8 = []
for dado in dfDadosDiariosPredict['Resp8'].dropna():
    resp8.append(dado)

resp9 = []
for dado in dfDadosDiariosPredict['Resp9'].dropna():
    resp9.append(dado)

resp10 = []
for dado in dfDadosDiariosPredict['Resp10'].dropna():
    resp10.append(dado)

#d = {'data':data,'tempmin': tempmin,'tempmax':tempmax,'tempmed':tempmed,'um':um, 'ven':ven}
d = {'datapred':datapred,'Resp1': resp1,'Resp2': resp2,'Resp3': resp3,'Resp4': resp4,'Resp5': resp5,'Resp6': resp6,'Resp7': resp7,'Resp8': resp8,'Resp9': resp9,'Resp10': resp10}

dfDadosDiariosPredictFinal = pd.DataFrame(data=d)

features = list(dfDadosDiariosPredictFinal.columns[1:11])
XP = dfDadosDiariosPredictFinal[features]


XP
# tempmax tempmed tempmin um ven

result = clf.predict(XP)

i = 0
def transforme(r):
 
    if r[1] == 1:
        return "Iniciante"
    elif r[2] == 1:
        return "Intermediario"
    elif r[0] == 1:
        return "Experiente"
    else:
        return "Sem classificação"
    
print(result)  
for resp in result:
    print(str(datapred[i])+' - '+transforme(resp))
    i=i+1
    
from sklearn.datasets import load_iris

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 