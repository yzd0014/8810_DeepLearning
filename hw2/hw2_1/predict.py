import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import json
import time
import sys
tf.logging.set_verbosity(tf.logging.ERROR)
#data processing
              
vocab_2_int, int_2_vocab, max_sentence_length = pickle.load(open('data_mapping.p', 'rb'))

testing_data_path = sys.argv[1]
test_output_path = sys.argv[2]

testing_data = open(testing_data_path + '/id.txt')
testing_data = testing_data.read()
testing_data = testing_data.split()

outf = open(test_output_path, "w")
batch_size = 1
    
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph('V2CV2.meta')
    loader.restore(sess,tf.train.latest_checkpoint('./'))

    logits = loaded_graph.get_tensor_by_name('logits:0')
    frame_inputs = loaded_graph.get_tensor_by_name('frame_inputs:0')
    caption_inputs = loaded_graph.get_tensor_by_name('caption_inputs:0')
    targets = loaded_graph.get_tensor_by_name('targets:0')
    
    for i_video in range(len(testing_data)):
        video_name = testing_data[i_video]
        features = np.load(testing_data_path + '/feat/'+ video_name + '.npy')
        video_feat = [features]
        video_feat.reverse()

        predict_caption = np.zeros((batch_size, max_sentence_length)) + vocab_2_int['<PAD>']
        predict_caption[:,0] = vocab_2_int['<GO>']

        for i in range(max_sentence_length):
            test_logits = sess.run(logits, feed_dict={frame_inputs: video_feat, caption_inputs: predict_caption})
            predict_word = test_logits[:,i].argmax(axis=-1)

            if predict_word[0] == vocab_2_int['<PAD>']:
                break
            if i+1 < max_sentence_length:
                predict_caption[:,i+1] = predict_word
            if predict_word[0] == vocab_2_int['<EOS>']: 
                break

        sentence = ''
        for word in predict_caption[0]:
            if word == vocab_2_int['<EOS>'] or word == vocab_2_int['<PAD>']:
                break
            if word != vocab_2_int['<GO>']:
                sentence = sentence + int_2_vocab[word] + ' '

        sentence = sentence.capitalize()    
        sentence = list(sentence)
        sentence[-1] = '.'
        sentence = ''.join(sentence)
        sentence = video_name + ',' + sentence
        outf.write(sentence)
        outf.write('\n')
    
outf.close()
print('Testing finished.')