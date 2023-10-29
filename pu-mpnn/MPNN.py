import numpy as np
import tensorflow as tf
import sys, time, warnings
from util import _permutation
from rdkit import Chem, rdBase
from sklearn.metrics import mean_absolute_error

class Model(object):

    def __init__(self, n_node, dim_node, dim_edge, dim_atom, dim_y, dim_h=50, n_mpnn_step=5, dr=0.1, batch_size=16, useGPU=True):

        warnings.filterwarnings('ignore')
        tf.logging.set_verbosity(tf.logging.ERROR)
        rdBase.DisableLog('rdApp.error') 
        rdBase.DisableLog('rdApp.warning')

        self.n_node=n_node
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.dim_atom=dim_atom
        self.dim_y=dim_y

        self.dim_h=dim_h
        self.n_mpnn_step=n_mpnn_step
        self.dr=dr
        self.batch_size=batch_size

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.trn_flag = tf.placeholder(tf.bool)
        
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.dim_node])
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, self.dim_edge])      
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, 1])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])
        
        self.mask = tf.clip_by_value(tf.reduce_max(self.node, 2, keepdims=True), 0, 1)
        
        self.hidden_0, self.hidden_n = self._MP(self.batch_size, self.node, tf.concat([self.edge, self.proximity], 3), self.mask, self.n_mpnn_step, self.dim_h)
        self.Y_pred = self._Readout(self.batch_size, self.mask, self.hidden_0, self.hidden_n, self.dim_h, self.dim_y, self.dr)
                 
        # session
        self.saver = tf.train.Saver()
        if useGPU:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0} )
            self.sess = tf.Session(config=config)


    def train(self, DV_trn, DE_trn, DP_trn, DY_trn, DV_val, DE_val, DP_val, DY_val, load_path=None, save_path=None):

        ## objective function
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        ## objective function
        for var in tf.trainable_variables():
            shape = var.shape
            array = np.asarray([dim.value for dim in shape])
            mulValue = np.prod(array)
            Total_params += mulValue
            
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            shape = var.shape
            array = np.asarray([dim.value for dim in shape])
            mulValue = np.prod(array)
            Trainable_params += mulValue
            
        NonTrainable_params = Total_params - Trainable_params    
        print(f'Total params:{Total_params}')
        print(f'Trainable params:{Trainable_params}')
        print(f'Non-trainable params:{NonTrainable_params}')
        reg = tf.square(tf.concat([tf.reshape(v, [-1]) for v in tf.trainable_variables()], 0))
        l2_loss = 1e-10 * tf.reduce_mean(reg)

        cost_Y_indiv = [tf.reduce_mean(tf.square(self.Y[:,yid:yid+1] - self.Y_pred[:,yid:yid+1])) for yid in range(self.dim_y)]
        cost_Y_total = tf.reduce_sum(cost_Y_indiv)
        
        vars_MP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MP')
        vars_Y = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y/'+str(yid)+'/readout') for yid in range(self.dim_y)]

        lr_list = np.array([1e-3, 1e-4, 1e-5])
        train_op_total = [tf.train.AdamOptimizer(learning_rate = lr).minimize(cost_Y_total + l2_loss) for lr in lr_list]
        train_op_indiv = [[tf.train.AdamOptimizer(learning_rate = lr).minimize(cost_Y_indiv[yid] + l2_loss, var_list=vars_Y[yid]) for lr in lr_list * 0.1] for yid in range(self.dim_y)] 
                
        self.sess.run(tf.initializers.global_variables())            
        np.set_printoptions(precision=5, suppress=True)

        n_batch = int(len(DV_trn)/self.batch_size)

        if load_path is not None: self.saver.restore(self.sess, load_path)
            
        ## tranining
        print('::: training')
        max_epoch=500
        for yid in range(-1, self.dim_y):

            lr_id = 0
            lr_epoch = 0
        
            trn_log = np.zeros(max_epoch)
            val_t = np.zeros(max_epoch)
            
            for epoch in range(max_epoch):

                #[DV_trn, DE_trn, DP_trn, DY_trn] = _permutation([DV_trn, DE_trn, DP_trn, DY_trn])
                
                trnscores = np.zeros(n_batch) 
                if epoch > 0:
                    for i in range(n_batch):
        
                        start_=i*self.batch_size
                        end_=start_+self.batch_size
                        
                        assert self.batch_size == end_ - start_
                        
                        if yid==-1:
                            trnresult = self.sess.run([train_op_total[lr_id], cost_Y_total],
                                                      feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], 
                                                                   self.proximity: DP_trn[start_:end_], self.Y: DY_trn[start_:end_], self.trn_flag: True}) 
                        else:
                            trnresult = self.sess.run([train_op_indiv[yid][lr_id], cost_Y_indiv[yid]],
                                                      feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], 
                                                                   self.proximity: DP_trn[start_:end_], self.Y: DY_trn[start_:end_], self.trn_flag: True}) 
                            
                        trnscores[i] = trnresult[1]
                    
                    trn_log[epoch] = np.mean(trnscores)        
                    print('--training with lr_id:', lr_id, 'yid: ', yid, ' epoch id: ', epoch, ' trn log: ', trn_log[epoch])
                
                # validation
                val_mae = self.test_mae(DV_val, DE_val, DP_val, DY_val,self.batch_size)
                val_t[epoch] = np.sum(val_mae)
                print('------val MAE sum: ', val_t[epoch], 'BEST: ', np.min(val_t[0:epoch+1]))
                print('------val MAE list: ', val_mae)         
              
                if np.min(val_t[0:epoch+1]) == val_t[epoch]:
                    self.saver.save(self.sess, save_path)
    
                if epoch - lr_epoch > 10 and np.min(val_t[0:epoch-10]) < np.min(val_t[epoch-10:epoch+1]):
                    self.saver.restore(self.sess, save_path)
                    lr_epoch = epoch - 0
                    lr_id = lr_id + 1
                    print('----decrease the learning rate, current BEST: ', self.test_mae(DV_val, DE_val, DP_val, DY_val, 5))
                    if lr_id == len(lr_list): break
                    
            print('----termination condition is met')
            self.saver.restore(self.sess, save_path)


    def test_mae(self, DV, DE, DP, DY, m):
    
        DY_hat = np.mean([self.test(DV, DE, DP) for _ in range(m)], 0)
        mae = np.array([mean_absolute_error(DY[:,yid:yid+1], DY_hat[:,yid:yid+1]) for yid in range(self.dim_y)])
        
        return mae
        
        
    def test(self, DV_tst, DE_tst, DP_tst, trn_flag = True):
    
        n_batch_tst = int(len(DV_tst)/self.batch_size)
        DY_tst_hat=[]
        for i in range(n_batch_tst):
        
            start_=i*self.batch_size
            end_=start_+self.batch_size
            assert self.batch_size == end_ - start_
            
            DY_tst_batch = self.sess.run(self.Y_pred,
                                         feed_dict = {self.node: DV_tst[start_:end_], self.edge: DE_tst[start_:end_],
                                                      self.proximity: DP_tst[start_:end_], self.trn_flag: trn_flag})
            
            DY_tst_hat.append(DY_tst_batch) 
        DY_tst_hat = np.concatenate(DY_tst_hat, 0)
        return DY_tst_hat 
    
    def predict(self, DV_tst, DE_tst, DP_tst, trn_flag = True):
        n_batch_tst = int(len(DV_tst)/self.batch_size)
        DY_tst_hat=[]
        for i in range(n_batch_tst):
        
            start_=i*self.batch_size
            end_=start_+self.batch_size
            
            assert self.batch_size == end_ - start_
            
            DY_tst_batch = self.sess.run(self.Y_pred,
                                         feed_dict = {self.node: DV_tst[start_:end_], self.edge: DE_tst[start_:end_],
                                                      self.proximity: DP_tst[start_:end_], self.trn_flag: trn_flag})
            
            DY_tst_hat.append(DY_tst_batch)
        
        DY_tst_hat = np.concatenate(DY_tst_hat, 0)
  
   
    def _MP(self, batch_size, node, edge, mask, n_step, hiddendim):

        def _embed_node(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim, activation = tf.nn.tanh)
        
            inp = inp * mask
        
            return inp

        def _edge_nn(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * hiddendim)
        
            inp = tf.reshape(inp, [batch_size, self.n_node, self.n_node, hiddendim, hiddendim])
            inp = inp * tf.reshape(1-tf.eye(self.n_node), [1, self.n_node, self.n_node, 1, 1])
            inp = inp * tf.reshape(mask, [batch_size, self.n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, self.n_node, 1, 1])

            return inp

        def _MPNN(edge_wgt, node_hidden, n_step):
        
            def _msg_nn(wgt, node):
            
                wgt = tf.reshape(wgt, [batch_size * self.n_node, self.n_node * hiddendim, hiddendim])
                node = tf.reshape(node, [batch_size * self.n_node, hiddendim, 1])
            
                msg = tf.matmul(wgt, node)
                msg = tf.reshape(msg, [batch_size, self.n_node, self.n_node, hiddendim])
                msg = tf.transpose(msg, perm = [0, 2, 3, 1])
                msg = tf.reduce_mean(msg, 3)
            
                return msg

            def _update_GRU(msg, node, reuse_GRU):
            
                with tf.variable_scope('mpnn_gru', reuse=reuse_GRU):
            
                    msg = tf.reshape(msg, [batch_size * self.n_node, 1, hiddendim])
                    node = tf.reshape(node, [batch_size * self.n_node, hiddendim])
            
                    cell = tf.nn.rnn_cell.GRUCell(hiddendim)
                    _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)
            
                    node_next = tf.reshape(node_next, [batch_size, self.n_node, hiddendim]) * mask
            
                return node_next

            nhs=[]
            for i in range(n_step):
                message_vec = _msg_nn(edge_wgt, node_hidden)
                node_hidden = _update_GRU(message_vec, node_hidden, reuse_GRU=(i!=0))
                nhs.append(node_hidden)
        
            out = tf.concat(nhs, axis=2)
            
            return out

        with tf.variable_scope('MP', reuse=False):
            
            edge_wgt = _edge_nn(edge)
            hidden_0 = _embed_node(node)
            hidden_n = _MPNN(edge_wgt, hidden_0, n_step)
            
        return hidden_0, hidden_n


    def _Readout(self, batch_size, mask, hidden_0, hidden_n, hiddendim, ydim, drate):
      
        def _readout(hidden_0, hidden_n, outdim):    
            
            def _attn_nn(inp, hdim):
            
                inp = tf.layers.dense(inp, hdim, activation = tf.nn.sigmoid)
                
                return inp
        
            def _tanh_nn(inp, hdim):
            
                inp = tf.layers.dense(inp, hdim)
            
                return inp

            attn_wgt = _attn_nn(tf.concat([hidden_0, hidden_n], 2), hiddendim * 10) 
            tanh_wgt = _tanh_nn(hidden_n, hiddendim * 10)
            readout = tf.reduce_mean(tf.multiply(tanh_wgt, attn_wgt) * mask, 1)
            
            readout = tf.layers.dense(readout, hiddendim * 10, activation = tf.nn.relu)
            readout = tf.layers.dropout(readout, drate, training = self.trn_flag)
            readout = tf.layers.dense(readout, hiddendim * 10, activation = tf.nn.relu)
            readout = tf.layers.dropout(readout, drate, training = self.trn_flag)
            readout = tf.layers.dense(readout, hiddendim * 10, activation = tf.nn.relu)
            
            pred = tf.layers.dense(readout, outdim) 
    
            return pred

        rout = []
        for j in range(ydim):
        
            with tf.variable_scope('Y/'+str(j)+'/readout', reuse=False):
                readout = _readout(hidden_0, hidden_n, 1)	
                
            rout.append(readout)
        
        rout = tf.concat(rout, axis=1)

        return rout
