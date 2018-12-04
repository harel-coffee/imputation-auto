    class alm_tf:
    def __init__(self,loss_name = 'cross_entropy',hidden_units = [],activation_fn = tf.nn.relu,n_classes = 9,batch_gd = 1,batch_size = 10,num_epochs=10,learning_rate = 0.1,use_saved_weight = 0):
        self.loss_name = loss_name
        self.hidden_units = hidden_units
        self.n_classes = n_classes
        self.batch_gd = batch_gd
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation_fn = activation_fn
        self.leraning_rate = learning_rate
        self.use_saved_weight = use_saved_weight
        self.sess = tf.Session()
           
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
   
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)        
     
    def fit(self,x,y):        
        #take 20% as validation set
        # x is input , y_ is output 
        self.x_input = np.array(x).astype('float32')
        self.y_input = np.array(y).astype('float32')
        self.y_input = label_vector(self.y_input)
           
        validation_pivot = int(self.x_input.shape[0]*0.2)
        rnd_idx = np.random.permutation(self.x_input.shape[0])
        validation_idx = rnd_idx[range(validation_pivot)]
        train_idx = rnd_idx[range(validation_pivot,self.x_input.shape[0])]
           
        self.x_train = self.x_input[train_idx,:]
        self.x_validation = self.x_input[validation_idx,:]
        self.y_train = self.y_input[train_idx,:]
        self.y_validation = self.y_input[validation_idx,:]
   
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes])
   
        #add hidden layers
        self.n_hlayer = len(self.hidden_units)
        self.W = list(np.zeros(self.n_hlayer))
        self.b = list(np.zeros(self.n_hlayer))
        self.h = list(np.zeros(self.n_hlayer))
        self.h_active =list(np.zeros(self.n_hlayer))
        self.W_s = list(np.zeros(self.n_hlayer))
        self.b_s = list(np.zeros(self.n_hlayer))
        self.h_s = list(np.zeros(self.n_hlayer))
        self.h_active_s = list(np.zeros(self.n_hlayer))
           
        for i in range(self.n_hlayer):
            if i == 0:
                self.W[i] =  tf.Variable(tf.zeros([self.x_train.shape[1],self.hidden_units[i]]))
                self.b[i] =  tf.Variable(tf.zeros([self.hidden_units[i]]))
                #self.W[i] =  self.weight_variable([self.x_train.shape[1],self.hidden_units[i]])
                #self.b[i] =  self.bias_variable([self.hidden_units[i]])
                self.h[i] =  tf.matmul(self.x,self.W[i]) + self.b[i]
                self.h_active[i] = self.activation_fn(self.h[i])
                self.W_s[i] = np.zeros([self.num_epochs,self.x_train.shape[1],self.hidden_units[i]])
            else:
                self.W[i] =  tf.Variable(tf.zeros([self.hidden_units[i-1],self.hidden_units[i]]))
                self.b[i] =  tf.Variable(tf.zeros([self.hidden_units[i]]))
#                 self.W[i] =  self.weight_variable([self.hidden_units[i-1],self.hidden_units[i]])
#                 self.b[i] =  self.bias_variable([self.hidden_units[i]])
                self.h[i] =  tf.matmul(self.h_acitve[i-1],self.W[i]) + self.b[i]
                #self.h[i] =  tf.matmul(self.h_acitve[i-1],self.W[i])
                self.h_active[i] = self.activation_fn(self.h[i])   
                self.W_s[i] = np.zeros([self.num_epochs,self.hidden_units[i],self.hidden_units[i]])
                   
            self.b_s[i] = np.zeros([self.num_epochs,self.hidden_units[i]])
            self.h_s[i] = np.zeros([self.num_epochs,self.x_train.shape[0],self.hidden_units[i]])
            self.h_active_s[i] = np.zeros([self.num_epochs,self.x_train.shape[0],self.hidden_units[i]])
      
        if self.n_hlayer == 0: # no hidden units
            #self.W_output = tf.Variable(tf.zeros([self.x_train.shape[1],self.n_classes]))
            #self.b_output = tf.Variable(tf.zeros([self.n_classes]))
            if self.use_saved_weight == 0:
                self.W_output =  self.weight_variable([self.x_train.shape[1],self.n_classes])
                self.b_output =  self.bias_variable([self.n_classes])
            else:
                self.W_output = tf.Variable(np.array(pd.read_csv('W_output.csv')).astype('float32'))   
                self.b_output = tf.Variable(np.squeeze(np.array(pd.read_csv('b_output.csv'))).astype('float32'))   
            #matrix multiplication
            self.y = tf.matmul(self.x,self.W_output) + self.b_output
            self.y_prob = tf.nn.softmax(self.y)
               
            self.weight_output = np.zeros([self.num_epochs,self.x_train.shape[1],self.n_classes])
            self.bias_output = np.zeros([self.num_epochs,self.n_classes])
        else:
            #self.W_output = tf.Variable(tf.zeros([self.hidden_units[self.n_hlayer-1],self.n_classes]))
            #self.b_output = tf.Variable(tf.zeros([self.n_classes]))  
            if self.use_saved_weight == 0: 
                self.W_output =  self.weight_variable([self.hidden_units[self.n_hlayer-1],self.n_classes])
                self.b_output =  self.bias_variable([self.n_classes])
            else:
                self.W_output = tf.Variable(np.array(pd.read_csv('W_output.csv')).astype('float32'))     
                self.b_output = tf.Variable(np.squeeze(np.array(pd.read_csv('b_output.csv'))).astype('float32'))   
   
            #matrix multiplication
            self.y = tf.matmul(self.h_active[self.n_hlayer-1],self.W_output) + self.b_output
            self.y_prob = tf.nn.softmax(self.y)
               
            self.weight_output = np.zeros([self.num_epochs,self.hidden_units[self.n_hlayer-1],self.n_classes])
            self.bias_output = np.zeros([self.num_epochs,self.n_classes])     
   
           
        #loss function
        if self.loss_name == 'cross_entropy':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
   
        #loss function optimization
        self.train_step = tf.train.GradientDescentOptimizer(self.leraning_rate).minimize(self.loss)
           
        #evaluation metrics    
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
           
        #run training
        self.sess.run(tf.global_variables_initializer())
           
        #save initiated parameters
        pd.DataFrame(self.W_output.eval(session = self.sess)).to_csv('W_output.csv',index = False)
        pd.DataFrame(self.b_output.eval(session = self.sess)).to_csv('b_output.csv',index = False)
           
           
        if self.batch_gd == 1 :
            self.batch_size = self.x_train.shape[0]
           
        self.batch_idx = np.arange(0,self.x_train.shape[0],self.batch_size)
        if self.batch_idx[-1] < self.x_train.shape[0]:
            self.batch_idx = np.append(self.batch_idx,self.x_train.shape[0])
                           
        self.loss_value = np.zeros(self.num_epochs)    
        self.train_accuracy = np.zeros(self.num_epochs)    
        self.validation_accuracy = np.zeros(self.num_epochs)   
   
        self.train_y_predicted_prob = np.zeros([self.num_epochs,self.x_train.shape[0],self.n_classes])
        self.train_label_predicted = np.zeros([self.num_epochs,self.x_train.shape[0]])
           
        self.validation_y_predicted_prob = np.zeros([self.num_epochs,self.x_validation.shape[0],self.n_classes])
        self.validation_label_predicted = np.zeros([self.num_epochs,self.x_validation.shape[0]])
           
        self.train_auroc = np.zeros(self.num_epochs)
        self.train_auprc = np.zeros(self.num_epochs)
        self.validation_auroc = np.zeros(self.num_epochs)
        self.validation_auprc = np.zeros(self.num_epochs)
           
        self.train_lable_truth = self.y_train[:,1]
        self.validation_lable_truth = self.y_validation[:,1]
           
        self.train_prior = (self.train_lable_truth == 1).sum()/self.train_lable_truth.shape[0]
        self.validation_prior = (self.validation_lable_truth == 1).sum()/self.validation_lable_truth.shape[0]
            
           
        for i in range(self.num_epochs):
            for j in range(len(self.batch_idx)-1):
                self.x_train_batch = self.x_train[range(self.batch_idx[j],self.batch_idx[j+1]),:]
                self.y_train_batch = self.y_train[range(self.batch_idx[j],self.batch_idx[j+1]),:]
                self.loss_value[i] = self.loss.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train}) 
#                 self.y.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})
   
#                 self.weight_output[i,:,:] =  self.W_output.eval(session = self.sess)
#                 self.bias_output[i,:] =  self.b_output.eval(session = self.sess)
#                 
#                 for k in range(self.n_hlayer):
#                     self.W_s[k][i,:,:] = self.W[k].eval(session = self.sess)
#                     self.b_s[k][i,:] = self.b[k].eval(session = self.sess)
#                     self.h_s[k][i,:,:] = self.h[k].eval(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
#                     self.h_active_s[k][i,:,:] = self.h_active[k].eval(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
# #                 
                self.train_accuracy[i] = self.accuracy.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})
                #self.train_y_predicted_prob[i,:,:] = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})  
                #self.train_label_predicted[i,:] = self.train_y_predicted_prob[i,:,1]  
                #self.train_auroc[i] = auroc_cal(self.train_lable_truth,self.train_y_predicted_prob[i,:,1])
                #self.train_auprc[i] = auprc_cal(self.train_lable_truth,self.train_y_predicted_prob[i,:,1])
                train_label_predicted = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})[:,1]
                   
                metric = roc_prc_cal(self.train_lable_truth,train_label_predicted)
                self.train_auroc[i] = metric['roc']
                self.train_auprc[i] = metric['prc']
                   
                self.validation_accuracy[i] = self.accuracy.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})
                #self.validation_y_predicted_prob[i,:,:] = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})  
                #self.validation_label_predicted[i,:] = self.validation_y_predicted_prob[i,:,1]                  
                #self.validation_auroc[i] = auroc_cal(self.validation_lable_truth,self.validation_y_predicted_prob[i,:,1])
                #self.validation_auprc[i] = auprc_cal(self.validation_lable_truth,self.validation_y_predicted_prob[i,:,1])
                   
                validation_label_predicted = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})[:,1]
                   
                metric = roc_prc_cal(self.validation_lable_truth,validation_label_predicted)
                self.validation_auroc[i] = metric['roc']
                self.validation_auprc[i] = metric['prc']
                   
                # spot check the weight update for first feature and first hidden units  
#                 g00 = 0              
#                 for n in range(self.x_train_batch.shape[0]):
#                     x0 = self.x_train_batch[n,0]
#                     a0 = self.h_active_s[0][i,n,0]
#                     w00 = self.weight_output[i,0,0]
#                     w01 = self.weight_output[i,0,1]
#                     p0 = self.train_y_predicted_prob[i,n,0]
#                     p1 = self.train_y_predicted_prob[i,n,1]
#                     y0 = self.y_train[n,0]
#                     y1 = self.y_train[n,1]
#                     g00 = g00 + (p0-y0)*w00*a0*(1-a0)*x0 + (p1-y1)*w01*a0*(1-a0)*x0
#                 g00 = g00*self.leraning_rate/self.x_train_batch.shape[0]
   
                print ('[epochs:' + str(i)  + '] loss:' + str(self.loss_value[i]) + ', train_auroc:' + str(self.train_auroc[i]) + ', validation_auroc:' + str(self.validation_auroc[i]))
                self.train_step.run(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
           
        fit_result = pd.DataFrame(np.transpose(np.vstack([self.loss_value,self.train_auroc,self.validation_auroc,self.train_auprc,self.validation_auprc,self.train_accuracy,self.validation_accuracy])))
        fit_result.columns = ['loss','train_auroc','validation_auroc','train_auprc','validation_auprc','train_accuracy','validation_accuracy']
        fit_result.to_csv('fit_result_'+ str(datetime.now()) + '.csv',index = False)
        plt.figure()
        p1 = plt.subplot2grid((2,2),(0,0))
        p1.plot(range(self.num_epochs),self.loss_value)
        p1.set_title('Loss function VS epochs')
        p1.set_xlabel('epochs')
        p1.set_ylabel('loss function')
        p2 = plt.subplot2grid((2,2),(0,1))
        p2.plot(range(self.num_epochs),self.train_accuracy,label = 'train accuracy')
        p2.plot(range(self.num_epochs),self.validation_accuracy, label = 'validation_accuracy')
        p2.set_title('training and validation performance VS epochs')
        p2.set_xlabel('epochs')
        p2.set_ylabel('performance')
        p2.legend(loc = "upper right")
        p3 = plt.subplot2grid((2,2),(1,0))
        p3.plot(range(self.num_epochs),self.train_auroc,label = 'train AUROC')
        p3.plot(range(self.num_epochs),self.validation_auroc, label = 'validation AUROC')
        p3.set_title('AUROC VS epochs')
        p3.set_xlabel('epochs')
        p3.set_ylabel('AUROC')
        p3.legend(loc = "upper right")
        p4 = plt.subplot2grid((2,2),(1,1))
        p4.plot(range(self.num_epochs),self.train_auprc,label = 'train AUPRC')
        p4.plot(range(self.num_epochs),self.validation_auprc, label = 'validation AUPRC')
        p4.set_title('AUPRC VS epochs' + '-Train prior:' + str(self.train_prior) + ' -Validation prior:' + str(self.validation_prior))
        p4.set_xlabel('epochs')
        p4.set_ylabel('AUPRC')
        p4.legend(loc = "upper right")
   
    def predict_proba(self,x):           
        self.x_validation = np.array(x).astype('float32')        
        self.proba = tf.nn.softmax(self.y)
        self.prediction_proba = self.proba.eval(session = self.sess, feed_dict={self.x:self.x_validation})
        return self.prediction_proba