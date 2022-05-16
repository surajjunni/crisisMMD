from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
import tensorflow
import warnings
import datetime
import optparse
import os, errno
import performance as performance
import keras.callbacks as callbacks
import data_process_multimodal_pair as data_process
import cnn_filter as cnn_filter
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard
from gensim.models import KeyedVectors
from keras.layers import Input, Activation, Add, Concatenate, Dropout
from keras.models import load_model
from keras.layers import concatenate
from time import time
import pickle
from tensorflow.keras.layers import BatchNormalization
from crisis_data_generator_image_optimized import DataGenerator
import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import load_model

class ImgInstance(object):
    def __init__(self, id=1, imgpath="", label=""):
        self.id = id
        self.imgpath = imgpath
        self.label = label
"""
def resnet_model():
    IMG_HEIGHT=224
    IMG_WIDTH=224
    restnet = ResNet50(include_top=False, weights='imagenet', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    # for layer in restnet.layers: # in case if we want to freeze the
    #     layer.trainable = False
    last_layer_output = restnet.layers[-1].output
    last_layer_output = keras.layers.Flatten()(last_layer_output)
    return last_layer_output,restnet;
"""
def vgg_model():
    vgg16 = VGG16(weights='imagenet')
    # Freeze All Layers Except Bottleneck Layers for Fine-Tuning
    # for layer in vgg16.layers:
    #     if layer.name in ['fc1', 'fc2', 'logit']:
    #         continue
    #     layer.trainable = False
    last_layer_output = vgg16.get_layer('fc2').output
    # vgg16.summary()
    return last_layer_output, vgg16


def check_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def file_exist(w2v_checkpoint):
    if os.path.exists(w2v_checkpoint):
        return True
    else:
        return False


def save_model(model, model_dir, model_file_name, tokenizer, label_encoder):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name + ".hdf5"
    tokenizer_file = model_dir + "/" + base_name + ".tokenizer"
    label_encoder_file = model_dir + "/" + base_name + ".label_encoder"

    configfile = model_dir + "/" + base_name + "_v2.config"
    configFile = open(configfile, "w")
    configFile.write("model_file=" + model_file + "\n")
    configFile.write("tokenizer_file=" + tokenizer_file + "\n")
    configFile.write("label_encoder_file=" + label_encoder_file + "\n")
    configFile.close()

    files = []
    files.append(configfile)

    # serialize weights to HDF5
    model.save(model_file)
    files.append(model_file)

    # saving tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(tokenizer_file)

    # saving label_encoder
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(label_encoder_file)


def write_results(out_file, file_name, accu, P, R, F1, wAUC, AUC, report, conf_mat):
    accu = accu * 100
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(accu)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n"
    print(result)
    print (report)
    out_file.write(file_name + "\n")
    out_file.write(result)
    out_file.write(report)
    out_file.write(conf_mat)

def dir_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

"""
It assumes the inputs are text files, train, development and test. 
"""
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data", default=None, type="string")
    parser.add_option('-v', action="store", dest="val_data", default=None, type="string")
    parser.add_option('-t', action="store", dest="test_data", default=None, type="string")
    parser.add_option('-m', action="store", dest="model_file", default="best_model.hdf5", type="string")
    parser.add_option('-o', action="store", dest="outputfile", default="results.tsv", type="string")
    parser.add_option("-w", "--w2v_checkpoint", action="store", dest="w2v_checkpoint",
                      default="data_w2v_info.model", type="string")
    parser.add_option("-d", "--log_dir", action="store", dest="log_dir", default="model_log/", type="string")
    # parser.add_option("-l","--log_file", action="store", dest="log_file", default="./log", type="string")
    parser.add_option("-c", "--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log/",
                      type="string")
    parser.add_option("-x", "--vocab_size", action="store", dest="vocab_size", default=20000, type="int")
    parser.add_option("--embedding_dim", action="store", dest="embedding_dim", default=300, type="int")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=32, type="int")
    parser.add_option("--nb_epoch", action="store", dest="nb_epoch", default=20, type="int")
    parser.add_option("--max_seq_length", action="store", dest="max_seq_length", default=25, type="int")
    parser.add_option("--patience", action="store", dest="patience", default=100, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=10, type="int")
    #parser.add_option("-n", "--num_of_inst", action="store", dest="num_of_inst", default=10, type="int")
    parser.add_option("--text_sim_score", action="store", dest="text_sim_score", default=0.6, type="float")
    parser.add_option("--image_sim_score", action="store", dest="image_sim_score", default=0.6, type="float")
    parser.add_option("--total_sim_score", action="store", dest="total_sim_score", default=0.6, type="float")
    parser.add_option("--label_index", action="store", dest="label_index", default=6, type="int")
    parser.add_option("--image_dump", action="store", dest="image_dump", default="all_images_data_dump.npy", type="string")
    parser.add_option("-m1",action="store", dest="model_path1",type="string")
    parser.add_option("-m2",action="store", dest="model_path2",type="string")
    parser.add_option("-m3", action="store", dest="model_path3", type="string")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)
    test_file = options.test_data
    out_file = options.outputfile
    train_file = options.train_data
    model1_path=options.model_path1
    model2_path=options.model_path2
    model3_path=options.model_path3
    best_model_path = options.model_file
    log_path = options.checkpoint_log
    log_dir = os.path.abspath(os.path.dirname(log_path))
    dir_exist(log_dir)
    
    ######## Parameters ########                                                
    MAX_SEQUENCE_LENGTH = options.max_seq_length                                
    MAX_NB_WORDS = options.vocab_size                                           
    EMBEDDING_DIM = options.embedding_dim                                       
    batch_size = options.batch_size                                             
    nb_epoch = options.nb_epoch                                                 
    patience_early_stop = options.patience                                      
    patience_learning_rate = options.patience                                   
    dir_exist(options.checkpoint_log)                                           
    delim = "\t"                                                                
                                                                                
    #### training dataset                                                       
    dir_name = os.path.dirname(train_file)                                      
    base_name = os.path.basename(train_file)                                    
    #base_name = os.path.splitext(base_name)[0]                                 
                                                                                
    train_x, train_image_list, train_y, train_le, train_labels, word_index, tokenizer = data_process.read_train_data_multimodal(
        train_file,                                                             
        MAX_NB_WORDS,                                                           
        MAX_SEQUENCE_LENGTH,int(options.label_index),                           
        delim)                                                                  
    #print(train_image_list)                                                    
    print(train_x.shape,train_y.shape) 
    nb_classes = len(set(train_labels))
    with open(options.image_dump, 'rb') as handle:                              
        images_npy_data = pickle.load(handle)
 
    base_name = os.path.basename(train_file)
    #base_name = os.path.splitext(base_name)[0]
    log_file = "image_log_v2.txt"
    model1=load_model(model1_path)
    model2=load_model(model2_path)
    model3=load_model(model3_path)
    models=[model1,model2,model3] 

    ############ Test data  ########
    dir_name = os.path.dirname(out_file)
    base_name = os.path.basename(out_file)
    base_name = os.path.splitext(base_name)[0]
    out_file_name = dir_name + "/" + base_name + ".txt"
    out_file = open(out_file_name, "w")

    test_x, test_image_list, test_y, test_le, test_labels, ids = data_process.read_dev_data_multimodal(test_file,
                                                                                                       tokenizer,
                                                                                                       MAX_SEQUENCE_LENGTH,int(options.label_index),
                                                                                                       delim)
                                                                                
    print ("Number of classes: " + str(nb_classes))                             
    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,  
              "n_classes": nb_classes, "shuffle": False}                        
    print("image size: "+str(len(test_image_list)))                             
    print("test x: "+str(len(test_x)))                                          
    print("test y: "+str(len(test_y)))                                          
    test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)

    preds = [model.predict_generator(test_data_generator, verbose=1) for model in models]
    import numpy as np
    print(preds[0])
    preds = np.array(preds)
    summed = np.sum(preds,axis=0)
    a=np.argmax(summed,axis=1)
    test_prob = np.zeros((a.size, a.max()+1))
    test_prob[np.arange(a.size),a] = 1
    test_prob = test_prob.astype(int)
    print(test_prob)
    #print("dev true len: "+str(len(dev_y)))
    #print("dev pred len: " + str(len(dev_prob)))
    #AUC, accu, P, R, F1, report = performance.performance_measure_cnn(ensemble_pred[:len(dev_prob)], dev_prob, train_le)
    print(test_y)
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(test_y[:len(test_prob)], test_prob, train_le)
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.2f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC))+ "\n"
    print("results-cnn:\t"+base_name+"\t"+result)                               
    print (report)                                                              
    out_file.write( test_file+ "\n")                                            
    out_file.write(result)                                                      
    out_file.write(report)                                                      
    conf_mat_str = performance.format_conf_mat(test_y[:len(test_prob)], test_prob, train_le)
    out_file.write(conf_mat_str+"\n")                                           
    out_file.close()
