import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from gym.spaces import MultiDiscrete, Box, MultiBinary, Dict
from BTFrame import BTFrame,position
np.set_printoptions(suppress=True,precision=3)


"""
The deep Q-network class
The implementation of DQN credit to chapter 5 in 'Reinforcement Learning with TensorFlow' written by Sayon Dutta.
The modification here is adding one more hidden layer, using ADAM optimizer, and add L2 regulization
"""

def leaky_relu(x, alpha=0.05):# If you're using newer version of tensorflow, you can use tf.nn.leaky_relu()
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
class DQN:
    def __init__(self,learning_rate,gamma,n_features,n_actions,epsilon,parameter_changing_pointer,memory_size):
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.batch_size = 100
        self.experience_counter = 0
        self.experience_limit = memory_size
        self.replace_target_pointer = parameter_changing_pointer
        self.learning_counter = 0
        self.memory = np.zeros([self.experience_limit,self.n_features*2 + 1 + 1])  #for experience replay

        self.build_networks()
        p_params = tf.get_collection('primary_network_parameters')
        t_params = tf.get_collection('target_network_parameters')
        self.replacing_target_parameters = [tf.assign(t,p) for t,p in zip(t_params,p_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.deadL1, self.deadL2=0,0
        
    def build_networks(self):
        #primary network
        hidden_units = [60,20]
        self.s = tf.placeholder(tf.float32,[None,self.n_features])
        self.qtarget = tf.placeholder(tf.float32,[None,self.n_actions])
        parameters={}
        with tf.variable_scope('primary_network'):
            c = ['primary_network_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, hidden_units[0]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b1 = tf.get_variable('b1', [1, hidden_units[0]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                l1 = leaky_relu(tf.matmul(self.s, w1) + b1)
                L2_reg=tf.nn.l2_loss(w1)
            # second layer
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [hidden_units[0], hidden_units[1]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b2 = tf.get_variable('b2', [1, hidden_units[1]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                l2 = leaky_relu(tf.matmul(l1, w2) + b2)
                L2_reg=L2_reg+tf.nn.l2_loss(w2)

            # third layer
            with tf.variable_scope('layer3'):
                w3 = tf.get_variable('w3', [hidden_units[1], self.n_actions],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b3 = tf.get_variable('b3', [1, self.n_actions],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                self.qeval = tf.matmul(l2, w3) + b3
                L2_reg=L2_reg+tf.nn.l2_loss(w3)
                self.L2_reg=L2_reg


        with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.qtarget,self.qeval))+self.L2_reg

        with tf.variable_scope('optimiser'):
                self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        #target network
        self.st = tf.placeholder(tf.float32,[None,self.n_features])

        with tf.variable_scope('target_network'):
            c = ['target_network_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, hidden_units[0]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b1 = tf.get_variable('b1', [1, hidden_units[0]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                l1 = leaky_relu(tf.matmul(self.st, w1) + b1)

            # second layer
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [hidden_units[0], hidden_units[1]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b2 = tf.get_variable('b2', [1, hidden_units[1]],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                l2 = leaky_relu(tf.matmul(l1, w2) + b2)
                
            # third layer
            with tf.variable_scope('layer3'):
                w3 = tf.get_variable('w3', [hidden_units[1], self.n_actions],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                b3 = tf.get_variable('b3', [1, self.n_actions],initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)
                self.qt = tf.matmul(l2, w3) + b3
    
    def target_params_replaced(self):
        self.sess.run(self.replacing_target_parameters)
        
    def store_experience(self,obs,a,r,obs_):
        index = self.experience_counter % self.experience_limit
        self.memory[index,:] = np.hstack((obs,[a,r],obs_))
        self.experience_counter+=1
        
    def fit(self):
        # sample batch memory from all memory
        if self.experience_counter < self.experience_limit:
            indices = np.random.choice(self.experience_counter, size=self.batch_size)
        else:
            indices = np.random.choice(self.experience_limit, size=self.batch_size)

        batch = self.memory[indices,:]
        qt,qeval = self.sess.run([self.qt,self.qeval],feed_dict={self.st:batch[:,-self.n_features:],self.s:batch[:,:self.n_features]})
        qtarget = qeval.copy()    
        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = self.memory[indices,self.n_features].astype(int)
        rewards = self.memory[indices,self.n_features+1]
        qtarget[batch_indices,actions] = rewards + self.gamma * np.max(qt,axis=1)

        _ = self.sess.run(self.train,feed_dict = {self.s:batch[:,:self.n_features],self.qtarget:qtarget})

        #increasing epsilon        
        if self.epsilon < 0.9:
            self.epsilon += 0.0002

        #replacing target network parameters with primary network parameters    
        if self.learning_counter % self.replace_target_pointer == 0:
            self.target_params_replaced()
            #print("target parameters changed")
        self.learning_counter += 1
     
    def epsilon_greedy(self,obs):
        #epsilon greedy implementation to choose action
        if np.random.uniform(low=0,high=1) < self.epsilon:
            return np.argmax(self.sess.run(self.qeval,feed_dict={self.s:obs[np.newaxis,:]}))
        else:
            return np.random.choice(self.n_actions)

"""Load market data with pandas datareader"""
import datetime as dt
import pandas_datareader as reader
start_date = dt.datetime(2013, 1, 1)
end_date = dt.datetime(2017, 12, 31)
ETF=reader.DataReader("0050.TW",start=start_date,end=end_date,data_source="yahoo")
TWD=reader.DataReader("TWD=X",start=start_date,end=end_date,data_source="yahoo")
SP500=reader.DataReader('^GSPC',start=start_date,end=end_date,data_source="yahoo")
UST10Y=reader.DataReader('^TNX',start=start_date,end=end_date,data_source="yahoo")

"""Organize market data into 2-d numpy ndarray, including n-day return, max and min price"""
TWD_=TWD.reindex(ETF.index)
SP500_=SP500.reindex(ETF.index)
UST10Y_=UST10Y.reindex(ETF.index)
ETF.fillna(method="ffill",inplace=True)
TWD_.fillna(method="ffill",inplace=True)
SP500_.fillna(method="ffill",inplace=True)
UST10Y_.fillna(method="ffill",inplace=True)
names=["ETF","TWD","SP500","UST10Y"]
df=pd.concat([ETF,TWD_,SP500_,UST10Y_],axis=1,keys=names)
print(len(ETF),len(TWD),len(df))
for name in names:
    columns=[]
    for n in [5,10,15,20,25,30,35,40,45,50,55,60]:
        df[name,"r"+str(n)]=df[name,"Close"]/df[name,"Close"].shift(n)-1
        columns.append("r"+str(n))
    for n in [5,10,20,30,60]:
        df[name,"max"+str(n)]=df[name,"Close"]/df[name,"High"].rolling(n).max()-1
        columns.append("max"+str(n))
        df[name,"min"+str(n)]=df[name,"High"]/df[name,"Low"].rolling(n).min()-1
        columns.append("min"+str(n))
        df[name,"mean"+str(n)]=df[name,"High"]/df[name,"Close"].rolling(n).mean()-1
        columns.append("mean"+str(n))
    columns.append("Close")
df.dropna(axis=0,inplace=True)
ndarrays=[]
for name in ["ETF"]:
    ndarrays.append(df[name,"Close"].values.reshape(-1,1))
prices=np.concatenate(ndarrays,axis=1)
ndarrays=[]
for name in names:
    ndarrays.append(df[name][columns].values)
states=np.concatenate(ndarrays,axis=1)
data=np.concatenate([prices,states],axis=1)


#functions of stop conditions
def stop_loss(state):
    MTM=state["MTM"]
    done = MTM<-50000
    if done:
        print('trigger stop loss when MTM reach ',MTM)
    return done,-0.3
def profit_take(state):
    MTM=state["MTM"]
    done = MTM>50000
    if done:
        print('trigger profit take when MTM reach ',MTM)
    return done, 0.3
stop_funcs=[stop_loss, profit_take]

#This function flattten the 2d array of original observation data from env into 1d array and concatenate MTM, PL, position in order to feed into tensorflow.
def convert_observe(obs_dict):
    obs=np.zeros(2+obs_dict["position"].shape[0]+obs_dict["data"].shape[1])
    obs[0],obs[1]=obs_dict["MTM"], obs_dict["PL"]
    obs[2:2+env.n_ticker], obs[2+env.n_ticker:]=obs_dict["position"],obs_dict["data"]
    return obs


#from BTFrame import BTFrame,position
tf.reset_default_graph()
env=BTFrame(data,n_ticker=1, lot_size=1000,obs_length=1,max_trade=2,max_position=np.array([5]),min_position=np.array([-5]), stop_funcs=stop_funcs)
n_features=data.shape[1]+2
dqn = DQN(learning_rate=0.01,gamma=0.9,n_features=n_features,
          n_actions=env.action_space.nvec[0],epsilon=0.2,parameter_changing_pointer=200,memory_size=500)

episodes, total_steps, record_freq = 400, 0, 30
gloable_log, episode_brief=[],[]
for episode in range(episodes):
    steps = 0		
    obs = env.reset()
    obs = convert_observe(obs)
    episode_reward = 0
    local_log=[]
    initial_price=env.current_price()
    while True:
        action = dqn.epsilon_greedy(obs)
        obs_,reward,terminate,_ = env.step(action)
        obs_ = convert_observe(obs_)
        dqn.store_experience(obs,action,reward,obs_)
        if total_steps > 500:
            dqn.fit()
        episode_reward+=reward
        if episode%record_freq==0: #record detail in every 'record_freq' episode
            local_log.append(np.concatenate([[env.date_index], env.current_price(),env.centralize_action(action),env.state["position"],[env.state["MTM"],env.state["PL"]]]))
        if terminate:
            if episode%record_freq==0: #record detail in every 'record_freq' episode
                gloable_log.append(np.stack(local_log,axis=0))
            final_price=env.current_price()
            stock_return=final_price/initial_price-1
            # record (equity return, MTM, PL) of each episode
            episode_brief.append(np.concatenate([stock_return,[env.state["MTM"],env.state["PL"]]]))
            break
        obs = obs_
        total_steps+=1
        steps+=1
episode_brief = np.stack(episode_brief)
episode_brief = np.concatenate([episode_brief,(episode_brief[:,1]+episode_brief[:,2]).reshape((-1,1))],axis=1)


df_result=pd.DataFrame(episode_brief,columns=["udn","MTM","PL","Total"])
MA=df_result.rolling(50).mean()[50:]
MA[["udn","Total"]].plot()
