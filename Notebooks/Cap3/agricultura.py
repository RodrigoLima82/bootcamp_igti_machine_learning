#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:32:50 2019

Programa que contém os códigos para a aula de Aplicações de IoT para Agricultura

@author: tulio
"""

import numpy as np  #utilizada para tratar vetores e matrizes
import tensorflow as tf
import keras  #utilizada para criar os modelos de deep learning
#from keras import backend as k #utiliza
from tensorflow.keras.models import Sequential #utilizada para criar o modelo sequencial
from tensorflow.keras.layers import Activation #utilizada para obter a funcao de ativação
from tensorflow.keras.layers import Dense,Flatten,Conv2D #utilizada para importar as camadas Dense e Flatten
from tensorflow.keras.optimizers import Adam # utilizada para importar o otimizador do tipo Adam
from tensorflow.keras.metrics import categorical_crossentropy #utilizada para avaliar o modelo
from tensorflow.keras.preprocessing.image import ImageDataGenerator #utilizada no pre-processamento das imagens
from tensorflow.keras.layers import BatchNormalization #utilizada para normalizar as "bateladas" no processo de treinamento
from tensorflow.keras.layers import * #importa todos os tipos de camadas
from matplotlib import pyplot as plt #utilizada para realizar o plot dos gráficos
from sklearn.metrics import confusion_matrix #utilizada para criar a matriz de confusão
import itertools #utilizada para criar os iterators (para os loops)
import matplotlib.image as mpimg  #utilizada para realizar o plot das imagens
from mlxtend.plotting import plot_confusion_matrix  #utilizar para plotar a matriz de confusão 

#caminhos para cada um dos conjuntos de imagens a serem utilizadas
caminho_treinamento='BD_treinamento'  #dividido em 2 pastas (sadias e contaminadas) - 60/60
caminho_validacao='BD_validacao' #dividido em 2 pastas (sadias e contaminadas) - 20/20
caminho_teste='BD_teste' #dividido em 2 pastas (sadias e contaminadas) - 25/25

#---------------------------------------------------
# Conhecendo e Preparando o BD
#----------------------------------------------------

#cria a batelada utilizando dados que estão no disco 
# ImageDataGenerator - utilizada para adicionar as imagens e converter em um formato padrão (224x224)
batelada_treino=ImageDataGenerator().flow_from_directory(caminho_treinamento,target_size=(224,224),classes=['sadias','contaminadas'],batch_size=10)
batelada_validacao=ImageDataGenerator().flow_from_directory(caminho_validacao,target_size=(224,224),classes=['sadias','contaminadas'],batch_size=5)
batelada_teste=ImageDataGenerator().flow_from_directory(caminho_teste,target_size=(224,224),classes=['sadias','contaminadas'],batch_size=10)


#utilizado para interar sobre a batelada de dados 
img,labels = next(batelada_treino)

#utilizado para mostrar as imagens 
plt.figure()
plt.imshow(img[0].astype(np.uint8)) # seleciona a imagem da posição [0]
plt.title("{}".format(labels[0]))
plt.show()  # mostra a imagem

#---------------------------------------------------
# Criando o primeiro modelo de classificação
#----------------------------------------------------

#criando o modelo de rede convolucionária
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))# 32= número de neurônios na camada/ (3,3)= filtro utilizado para percorrer 
#a imagem / (224,224,3)= tamanho das imagens comprimento 224x largura 224 e RGB=3
model.add(Flatten()) #utilizada para criar um vetor para a entrada de dados na camada de saída
model.add(Dense(2,activation='softmax'))# camada de saída da rede 2 neurônios. 10= sadia /01= contaminada

#mostrando a configuração da rede CNN criada
model.summary()


#definindo o otimizador e a função perda
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])


#treinamento do modelo
history=model.fit_generator(batelada_treino,steps_per_epoch=12,validation_data=batelada_validacao,validation_steps=4,epochs=20,verbose=2)
#deve ser utilizada, pois estamos realizando o treinamento via batelada
#steps_per_epoch = define a quantidade de epocas utilizadas para treinamento, baseando-se no numero de dados utilizados
#vamos utilizar 120 imagens para treinamento (60 sadias e 60 contaminadas), como a batelada é de 10, temos 120/10 = 12 vezes
# validation_data = utilizado para gerar a validação (compara o desempenho do treinamento com o valor real): a cada epoca de treinamento,
#compara o resultado obtido com a previsão realizada nas
#imagens de validação
#verbose=2 - indica o que desejamos exibir na saída do treinamento


# Lista os dados históricos do treinamento
print(history.history.keys())
# summarize history para a accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()

#---------------------------------------------------
# Realiza a previsão do modelo
#----------------------------------------------------

#previsão sobre qual imagem corresponde a cada elemento
teste_img, teste_labels=next(batelada_teste)

#utilizado para mostrar as imagens 
plt.figure()
plt.imshow(teste_img[0].astype(np.uint8)) 
plt.title("{}".format(teste_labels[0]))
plt.show()  # mostra a imagem


#testar a classificação da imagens 
teste_labels=teste_labels[:,0] # transforma sadias (10) em 1 e contaminadas (01) em apenas 0

#realiza a previsão utilizando os dados de teste
previsao=model.predict_generator(batelada_teste,steps=1,verbose=0)
#como no fit, devemos utilizar o generator, pois estamos utilizando as bateladas de dados 
print(previsao)

#criando a matriz de confusão para comparar os resultados
matriz_confusao=confusion_matrix(teste_labels,previsao[:,0])
nomes_das_classes=['contaminadas','sadias']
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=nomes_das_classes)
plt.show()

#-----------------------------------------------------------------------------
#  Melhorando a prevesão do modelo - TRANSFER LEARNING
#----------------------------------------------------------------------------


vgg16_model=tf.keras.applications.vgg16.VGG16() # classe já pre-treinada para ser utilizada em nosso classificador

vgg16_model.summary()# vamos ver como o modelo do vgg16 foi construído

print(type(vgg16_model))


#transformando o tipo model do vgg16 em sequencial
model=Sequential()  #cria um modelo sequencial 
for layer in vgg16_model.layers[:-1]: #extrai cada uma das camadas do vgg16 
    model.add(layer)            #adiciona no modelo criado até a penultima camada
    
model.summary()
print(type(model))


# retirar a ultima camada do modelo, pois só desejamos classificar entre 2 grupos de imagens
#model.layers.pop()

#colocando as camadas intermediárias em modo de "hibernação"
for layer in model.layers:
    layer.trainable=False
#colocar em modo de hibernação, garante que, durante o treinamento, os pesos não serão atualizados
    
#adicionando a ultima camada para a classificação entre 2 grupos de imagens (cachorros ou gatos)    
model.add(Dense(2, activation='softmax'))

#mostra o novo modelo CNN (nosso+vgg16)
model.summary()


#----------------------------------------------------------------------------
#  Inicia o treinamento através dos novos pesos
#---------------------------------------------------------------------------

#definindo o otimizador e a função perda
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

#treinamento do modelo
history=model.fit_generator(batelada_treino,steps_per_epoch=12, validation_data=batelada_validacao, validation_steps=4, epochs=20,verbose=2)


#previsão utilizando o modelo+VGG16

#previsão sobre qual imagem corresponde a cada elemento
teste_img, teste_labels=next(batelada_teste)

#utilizado para mostrar as imagens 
plt.figure()
plt.imshow(teste_img[4].astype(np.uint8)) #seleciona a imagem da posição 4
plt.title("{}".format(teste_labels[4]))
plt.show()  # mostra a imagem


#testar a classificação da imagens 
teste_labels=teste_labels[:,0] # transforma sadias 10 em 1 e contaminadas 01 em apenas 0

#realiza a previsão utilizando os dados de teste
previsao=model.predict_generator(batelada_teste,steps=1,verbose=0)
#como no fit, devemos utilizar o generator, pois estamos utilizando as bateladas de dados 
print(previsao)


#criando a matriz de confusão para comparar os resultados
matriz_confusao=confusion_matrix(teste_labels,np.round(previsao[:,0])) # a diferença é que a rede gera valores float, então devemos converter
#em valores inteiros (0,1)
nomes_das_classes=['contaminadas','sadias']
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=nomes_das_classes)
plt.show()



# Lista os dados históricos do treinamento
print(history.history.keys())
# summarize history para a accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Treinamento', 'Teste'], loc='upper left')
plt.show()



















