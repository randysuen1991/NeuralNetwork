import Classifier as C
import tensorflow as tf
import numpy as np


class NeuralNetworkClassifier(C.Classifier):
    
        
    def __init__(self,input_shape,num_classes,keep_prob=0.8,activation = tf.sigmoid,**kwargs):
        super().__init__()
        self.x = tf.placeholder(dtype=tf.float32,shape=input_shape)
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,num_classes])
        self.transfer_function = activation
        self.sess = tf.Session()
        self.counts = self._Initialize_counts()
        self.weights = dict()
        self.out = self.x
        self.keep_prob = keep_prob
        self.image_type = kwargs.get('image_type','gray')
    def _Initialize_counts(self):
        counts = dict()
        counts['bias'] = 0
        counts['conv2d'] = 0
        counts['fc'] = 0
        counts['max_pooling'] = 0
        counts['avg_pooling'] = 0
        counts['dropout'] = 0
        return counts
        
        
        
    def Compile(self):
        init = tf.global_variables_initializer()
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.out * self.y,reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
        self.sess.run(init)
        
        
    def Fit(self,x,y):
        self.Compile()
        cost = self.sess.run([self.loss,self.optimizer],feed_dict={self.x:x,self.y:y})            
        return cost
    def Run(self,accuracy,x,y):
        return self.sess.run([accuracy],feed_dict={self.x:x,self.y:y})
    
    @classmethod
    def Construct(cls,instance,layer_type,keep_prob=None,**kwargs):
                
        instance.counts[layer_type] += 1
        
        
        if layer_type == 'conv2d':
            instance.counts['bias'] += 1
            shape = kwargs.get('shape')
            num_channels = int(instance.out.__dict__['_shape'][3])
            shape.insert(2,num_channels)
            c_init = tf.truncated_normal(shape)
            c = tf.Variable(initial_value=c_init)
            b_init = tf.truncated_normal([shape[-1]])
            b = tf.Variable(initial_value=b_init)
            
            instance.weights['conv2d_'+str(instance.counts['conv2d'])] = c
            instance.weights['bias_'+str(instance.counts['bias'])] = b
            
            conv2d = tf.nn.conv2d(instance.out,c,strides=kwargs.get('strides',[1,1,1,1]),padding=kwargs.get('padding','SAME'))
            instance.out = instance.transfer_function(tf.add(conv2d,b))

        elif layer_type == 'pooling':
                
            pooling_type = kwargs.get('pooling_type',None)
            instance.counts[pooling_type+'_pooling'] += 1
                
                
            ksize = [1]
            ksize = ksize + kwargs.get('ksize',[1,1])
            if instance.image_type == 'gray':
                ksize += [1]
            elif instance.image_type =='rgb':
                ksize += [3]
            if pooling_type == 'avg' :
                instance.out = tf.nn.avg_pool(value=instance.out,ksize=ksize,strides=kwargs.get('strides',[1,1,1,1]),padding=kwargs.get('padding','SAME'))
            elif pooling_type == 'max' :
                instance.out = tf.nn.max_pool(value=instance.out,ksize=ksize,strides=kwargs.get('strides',[1,1,1,1]),padding=kwargs.get('padding','SAME'))
            elif layer_type =='dropout':
                instance.counts['dropout'] += 1
                instance.out = tf.nn.dropout(x = instance.out,keep_prob=kwargs.get('keep_prob',0.8))
            
            elif layer_type == 'flatten':
                instance.out = tf.reshape(instance.out,shape=[-1,int(np.prod(instance.out.__dict__['_shape'][1:]))])
                
            elif layer_type == 'fc':
            
                instance.counts['fc'] += 1
                instance.counts['bias'] += 1
                f_init = tf.truncated_normal([int(instance.out.__dict__['_shape'][1]),kwargs.get('shape')])
                f = tf.Variable(initial_value=f_init)
                b_init = tf.truncated_normal([kwargs.get('shape')])
                b = tf.Variable(initial_value=b_init)
            
                instance.weights['fc_'+str(instance.counts['fc'])] = f
                instance.weights['bias_'+str(instance.counts['bias'])] = b
                
                if kwargs.get('last',False) :
                    instance.out = tf.nn.softmax(tf.add(tf.matmul(instance.out,f),b))
                else :
                    instance.out = instance.transfer_function(tf.add(tf.matmul(instance.out,f),b))