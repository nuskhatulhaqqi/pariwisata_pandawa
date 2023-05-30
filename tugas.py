import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import svm
from sklearn.svm import SVC
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv


Home, Learn, Proses, Model, Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing dan TF-IDF', 'Model', 'Implementasi'])

with Home:
   st.title("""Klasifikasi Sentimen Wisata Menggunakan Data Google Maps dengan Algoritma SVM (Support Vector Machine)
   (Studi Kasus: Pantai Pandawa Bali )
   """)
   st.write('Informatika Pariwisata B')
   st.text("""
            1. Theresia Nazela 200411100028
            2. Nuskhatul Haqqi 200411100034   
            """)


with Learn:
   st.title("""Klasifikasi Sentimen Wisata Menggunakan Data Google Maps dengan Algoritma SVM (Support Vector Machine)
            (Studi Kasus: Pantai Pandawa Bali )
   """)
   st.write('Pada Penelitian ini akan dilakukan pengklasifikasian ulasan/komentar menggunakan metode SVM. Secara Garis besar SVM merupakan metode klasifikasi linier, sehingga menggunakan kernel untuk mengatasi data yang bersifat nonlinier. Dapat digunakan untuk klasifikasi dan regresi.')
   st.write('Dalam Klasifikasi ini data yang digunakan adalah ulasan atau komentar dari aplikasi Google Maps dengan studi kasus di Pantai Pandawa Bali.')
   st.title('Klasifikasi data inputan berupa : ')
   st.write('1. Text : data komentar atau ulasan yang diambil dari Goggle Maps')
   st.write('2. Label: kelas keluaran [positif, negatif]')

   st.title("""Asal Data""")
   st.write("Dataset yang digunakan adalah data hasil crowling dari Goggle Maps dataset dapat diambil di https://github.com/nuskhatulhaqqi/data_mining/blob/main/pariwisata_pandawa.csv")
   st.write("Total datanya yang digunakan ada 1000 data dengan  228 data negatif dan 772 data positif ")
   # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
   # if uploaded_files is not None :
   data = pd.read_csv('pariwisata_pandawa.csv',encoding= 'unicode_escape')
   # else:
      # for uploaded_file in uploaded_files:
         # data = pd.read_csv(uploaded_file)
         # st.write("Nama File Anda = ", uploaded_file.name)
         # st.dataframe(data)
      


with Proses:
   st.title("""Preprosessing""")
   "### Dataset"
   data
   "### Melakukan Cleansing "
   clean_tag = re.compile('@\S+')
   clean_url = re.compile('https?:\/\/.*[\r\n]*')
   clean_hastag = re.compile('#\S+')
   clean_symbol = re.compile('[^a-zA-Z]')
   def clean_punct(text):
      text = clean_tag.sub('', text)
      text = clean_url.sub('', text)
      text = clean_hastag.sub(' ', text)
      text = clean_symbol.sub(' ', text)
      return text
# Buat kolom tambahan untuk data description yang telah diremovepunctuation   
   preprocessing = data['Text'].apply(clean_punct)
   clean=pd.DataFrame(preprocessing)
   clean
   "### Melakukan Casefolding "

   def clean_lower(lwr):
      lwr = lwr.lower() # lowercase text
      return lwr
   # Buat kolom tambahan untuk data description yang telah dicasefolding  
   clean = clean['Text'].apply(clean_lower)
   casefolding=pd.DataFrame(clean)
   casefolding

   def casefolding(text):
      casefolding=[]
      for i in range(len(text)):
         casefolding.append(text[i])
      return casefolding
   casefolding = casefolding(clean)

   "### Melakukan Tokenisasi "
   def tokenisasi (text):
      tokenize=[]
      for i in range(len(text)):
         token=word_tokenize(text[i])
         tokendata = []
         for x in token :
            tokendata.append(x)
         tokenize.append(tokendata)
      return tokenize
   token= tokenisasi(casefolding)
   token
   "### Melakukan Stopword Removal "
   def stopword(text):  
      stopword=[]
      for i in range(len(text)):
         listStopword =  set(stopwords.words('indonesian')+stopwords.words('english'))
         removed=[]
         for x in (text[i]):
            if x not in listStopword:
               removed.append(x)
         stopword.append(removed)
         # print(removed)
      return stopword
   sw = stopword(token)
   sw

   "### Melakukan Stemming "
   def stemming(text):
      stemming=[]
      for i in range(len(text)):
         factory = StemmerFactory()
         stemmer = factory.create_stemmer()
         katastem=[]
         for x in (text[i]):
            katastem.append(stemmer.stem(x))
         stemming.append(katastem)
      return stemming
   # sm = stemming(sw)
   # stem = pd.DataFrame(sm)
   # stem.to_csv('hasil_stemming.csv',index=None)
   stm=[]
   with open('hasil_stemming.csv','r') as file:
      reader = csv.reader(file)
      for row in reader:
         if (row is not None ):
            stm.append(row)
   del stm[0]
   stm
   

   
   "### Hasil Proses Pre-Prosessing "
   def gabung(text):
      join=[]
      for i in range(len(text)):
         joinkata = ' '.join(text[i])
         join.append(joinkata)
      return join
   # join = gabung(sm)
   # hasilpreproses = pd.DataFrame(join, columns=['Text'])
   # hasilpreproses.to_csv('hasilpreproses.csv', index=None)
   hasilpreproses = pd.read_csv('hasilpreproses.csv')
   hasilpreproses
   hasilpreproses = hasilpreproses['Text'].values.astype('U')

   st.title("""TF-IDF""")
   tr_idf_model  = TfidfVectorizer()
   tf_idf_vector = tr_idf_model.fit_transform(hasilpreproses)
   tf_idf_array = tf_idf_vector.toarray()
   words_set = tr_idf_model.get_feature_names_out()
   # df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
   # df_tf_idf.to_csv('df_tf_idf.csv' , index=None)
   df_tf_idf = pd.read_csv('df_tf_idf.csv')
   df_tf_idf
      



with Model:
   st.title("""Modelling""")
   y = data.Label
   # split data
   X_train,X_test,y_train,y_test = train_test_split(df_tf_idf,y,test_size=0.2,random_state=4)
   clf = svm.SVC(kernel='linear')
   clf.fit(X_train, y_train)
   X_pred = clf.predict(X_test)
   akurasi = round(100 * accuracy_score(y_test,X_pred))
   st.subheader("Metode Yang Digunakan Adalah Support Vector Machine")
   st.write("Akurasi Terbaik Dari Skenario Uji Coba Diperoleh Sebesar : {0:0.2f} %" . format(akurasi))

   with open('svm_pickle','wb') as r:
      pickle.dump(clf,r)
   with open('vec_pickle','wb') as r:
      pickle.dump(tr_idf_model,r)


with Implementasi:
   st.title("""Implementasi Data""")

   inputan = st.text_input('Masukkan Ulasan')


   def submit():
      clean_symbol,casefolding,token,stopword,katastem,joinkata = preproses(inputan)
      # input
      inputs = np.array([inputan])

      with open('svm_pickle', 'rb') as r:
         d = pickle.load(r)
      with open('vec_pickle', 'rb') as r:
         data = pickle.load(r)

      X_pred = d.predict((data.transform(inputs)).toarray())
      hasil =f"Berdasarkan data yang Anda masukkan, maka ulasan masuk dalam kategori  : {X_pred[0]}"
      if (X_pred[0] == 'positif'):
         st.success(hasil)
      else:
         st.warning(hasil)
      st.subheader('Preprocessing')
      st.write('Cleansing :', clean_symbol)
      st.write("Case Folding :",casefolding)
      st.write("Tokenisasi :",token)
      st.write("Stopword :",stopword)
      st.write("Steeming :",katastem)
      st.write("Siap Proses :",joinkata)

   all = st.button("Submit")
   if all :
      def preproses(inputan):
         clean_tag = re.sub('@\S+','', inputan)
         clean_url = re.sub('https?:\/\/.*[\r\n]*','', clean_tag)
         clean_hastag = re.sub('#\S+',' ', clean_url)
         clean_symbol = re.sub('[^a-zA-Z]',' ', clean_hastag)
         casefolding = clean_symbol.lower()
         token=word_tokenize(casefolding)
         listStopword = set(stopwords.words('indonesian')+stopwords.words('english'))
         stopword=[]
         for x in (token):
            if x not in listStopword:
               stopword.append(x)
         factory = StemmerFactory()
         stemmer = factory.create_stemmer()
         katastem=[]
         for x in (stopword):
           katastem.append(stemmer.stem(x))
         joinkata = ' '.join(katastem)
         return clean_symbol,casefolding,token,stopword,katastem,joinkata
      st.balloons()
      submit()







