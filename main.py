import Sourcefile_preprocessing as pre_source
import Sourcefile_extract_vecto as ex_source
import Sourcefile_processfile as open_file
import pickle
from gensim.models import KeyedVectors,Word2Vec
path=r"C:\Users\hust_pc\Desktop\Improve_Bug\Training_Test\sourceFile_aspectj"

if __name__ == "__main__":
    pickle_input= open("sourfile_preprocessed.pickle","rb")
    corpus=pickle.load(pickle_input)
    pickle_input.close()
    ex_source.Extract_vecto(corpus)

    

     
    
    


    

    