import numpy as np
import Sourcefile_extract_vecto as ex_vec
from gensim.models import Word2Vec,KeyedVectors
import pickle
# vector of line
def CombineVector(line_sent,df):
    model=KeyedVectors.load_word2vec_format(r"C:\Users\hust_pc\Desktop\Improve_Bug\GoogleNews-vectors-negative300.bin",binary=True,limit =700000)
    vector_line = np.zeros(shape=(1,300))
    count=0
    for word_sent in  line_sent:
        vector_line += model.wv[word_sent]
        count+=1
    vector_line=vector_line/count
    return vector_line
def Extract_vecto(corpus):
    #load  Tf_Idf
    pickle_in=open(r"C:\Users\hust_pc\Desktop\Improve_Bug\tf_idf.pickle","rb")
    df=pickle.load(pickle_in)
    pickle_in.close()

    number_line=[]
    for sourcefile in corpus:
        number_line.append(len(sourcefile))
    line_max= max(number_line)
    matrix=[]
    for sourcefile in corpus:
        vecto=np.zeros(shape=(line_max,300))
        for i in range(len(sourcefile)):
            vecto[i]= ex_vec.CombineVector(sourcefile[i],df)
        matrix.append(vecto)
    pickle_out=open("vector_source.pickle","wb")
    pickle.dump(matrix,pickle_out)
    pickle_out.close()

    


