import numpy as np
import pandas as pd
import gym
from gym.spaces import MultiDiscrete, Box, MultiBinary, Dict
"""----numpy version----"""
class BTFrame(gym.Env):#numpy version
    action_space=None
    observation_space=None
    intMax, intMin, floatMax, floatMin =np.iinfo(np.int).max, np.iinfo(np.int).min, np.finfo(np.float).max, np.finfo(np.float).min
    """
    initializer的參數
    data：ndarray類別，測試資料，須包含標的物價格與其他希望觀察的指標 (n_day, n_column) n_column=n_ticker + n_observe_columns
    n_ticker：int,data欄中包含價格資料的欄位數(data的前n_ticker欄為價格資料，以後的為state觀察資料)
    lot_size: int, 一次交易最小單位, 例如台股為1000 (註：若是期權的話輸入每一點變動價值)
    obs_length: int, state 觀察的時間序列長度，預設為1，若要觀察時間序列，設定大於1的整數
    max_position,min_position: 最大最小部位數 
    episode_length: 預設為data之一半長度
    stop_funcs: list, python function物件list, 額外之停止條件，每個function之輸入須為含有data, MTM, PL,position值之dict, 輸出須為(布林, 額外reward)
    
    arguments of initializer:
    data：numpy ndarray，the training data, including stock prices and any observation data like techtical indicators, of shape (n_day, n_column), where n_column = n_ticker + n_observe_columns
    n_ticker：int,indicating first n_ticker columns of data is prices data, the rest columns are observation datas(data的前n_ticker欄為價格資料，以後的為state觀察資料)
    lot_size: int, minimum trade size of the market, ex, 1000 in Taiwan stock market.
    obs_length: int, observation length, i.e. if you want the agent to observe time series, use a number greater than 1, defaut is 1.
    episode_length: Int, length for each episode range, Env will trigger stop condition once step to the end of range, which is the only stop condition by default. Add more stop condition in the stop_funcs input argument if necessory. Default value is half of data length
    max_position,min_position: the max and min of position in porfolio.
    stop_funcs: list of python function objects for additional stop condition. Each function has one dict input: state output (bool, extra_reward) (see step function for detail of state dict)

    """
    def __init__(self, data, n_ticker, lot_size, 
                 obs_length =1, max_trade=1,
                 max_position=None, min_position=None,
                 episode_length=None, # 
                 stop_funcs=None):
        assert data.shape[1]>n_ticker
        self.data = data
        self.obs_length = obs_length
        self.episode_length = episode_length if episode_length else int(data.shape[0]/2)
        assert self.episode_length>self.obs_length
        self.lot_size=lot_size
        self.n_ticker=n_ticker
        self.stop_funcs=stop_funcs
        #self.episode_brief=[]
        self.action_space = MultiDiscrete(np.repeat(max_trade*2+1,n_ticker))
        self.max_position, self.min_position = max_position, min_position
        self.observation_space = Dict({"data":Box(BTFrame.floatMin, BTFrame.floatMax, shape=[data.shape[1]-n_ticker, obs_length], dtype=np.float32),  
                                       "MTM":Box(BTFrame.floatMin, BTFrame.floatMax, shape=[1],dtype=np.float32),
                                       "PL":Box(BTFrame.floatMin, BTFrame.floatMax, shape=[1],dtype=np.float32),
                                       "position":Box(0,BTFrame.intMax, shape=[n_ticker],dtype=np.int32)
                                      })
        self.reset()
    #重設部位、歷史資料區間、日期,回傳初始state
    #Reset blotter, episode range, date_index, and returns initial states.
    def reset(self,reset_position=True, reset_range=True):
        if reset_range:
            self.episode_range = self.get_range()
        self.date_index = self.episode_range[0] + self.obs_length
        if reset_position:
            
            self.position=position(self.n_ticker, 
                                   self.max_position*self.lot_size if self.max_position else None, 
                                   self.min_position*self.lot_size if self.min_position else None)
        obs_data = self.data[self.episode_range[0]:self.date_index,self.n_ticker:]
        self.start_price=self.current_price() #紀錄各標的起始價 Record initial price of the episode.
        self.state= {"data":obs_data,"MTM":0,"PL":0,"position":np.zeros(self.n_ticker)}
        return self.state
        #reset range
# 隨機產生一個episode的episode區間之touple(stat, end)
# Randomly generize touple representing a range of for each episode
    def get_range(self): 
        data_end = self.data.shape[0]
        start = np.random.randint(0,data_end-self.episode_length)
        end = start + self.episode_length
        return (start, end)
    """
    Implementation of step for gym.Env class
    The reward is calculated by the net change of market value of position
    Arguments:
    action: 1d ndarray of int, size is according to action_space, each value represents trade action, postive for buy, negative for sell, zero for do nothing.
    output:
    state, dictionary with key: 
    "data": 2d ndarray, observation data
    "MTM" int,mart to market of portfolio
    "PL": int, realized profit/loss
    "position" : int 1d ndarray, current position size for each stock.
    """
    def step(self,action): 
        #convert action to (-n/2,0, n/2)
        action = self.centralize_action(action)        
        current_price = self.current_price()
        for i in range(len(action)):
            if action[i]!=0:
                self.position.append(self.date_index-1, i, action[i]*self.lot_size, current_price[i])
        current_marketvalue=self.position.total_MarketValue(current_price)
        PL_sum = self.position.total_PL()
        position = self.position.position()
        self.date_index += 1
        current_price = self.current_price()
        obs_data = self.data[self.date_index-self.obs_length:self.date_index, self.n_ticker:]
        new_marketvalue=self.position.total_MarketValue(current_price)
        new_MTM_sum = self.position.total_MTM(current_price)
        if current_marketvalue==0:
            reward=0
        else:
            reward = (new_marketvalue - current_marketvalue)/ current_marketvalue
        state = {"data":obs_data,"MTM":new_MTM_sum,"PL":PL_sum,"position":position}
        self.state=state
        #self.local_log.append(np.concatenate([[self.date_index,new_MTM_sum,PL_sum],
                                              #current_price,action,position]))
        #停止條件 
        #Stop condition
        done=False
        done = done or self.date_index >= self.episode_range[1] #已達測試資料結尾
        if self.stop_funcs:
            for func in self.stop_funcs:
                done0,extra_reward=func(self.state)
                done = done or done0
                reward+=extra_reward
        #if done:
            #underlying_perf=current_price/self.start_price-1
            #self.episode_brief.append(np.concatenate([underlying_perf,[new_MTM_sum,PL_sum,new_MTM_sum+PL_sum]]))#紀錄標的物表現，最終ＭＴＭ，累計ＰＬ，ＭＴＭ＋ＰＬ
        return state, reward, done, {}
    def render(self):
        print(self.date_index, "MTM: ", self.state["MTM"], ", PL: ", self.state["PL"], ", position: ", self.state["position"],"price: ", self.data[self.date_index,0])
    
    def current_price(self):
        return self.data[self.date_index-1,0:self.n_ticker]
    def centralize_action(self,action):
        shift=(self.action_space.nvec-1)/2
        return action-shift


"""----numpy version----
用來管理部位的輔助class，主要有log(list)跟blotter(ndarray)兩者，其中log以時間序列紀錄交易
log的每一筆用(date_index,ticker,trade,price)的tuple組成 blotter是以shape為(tickers,3)的ndarray 
每一個ticker的value為(total_cost, position, realized_PL)
A class to help the Env to mange the portolio.
Instant variables:
log: list of touples, recording trade activities chronically. Each touple represents (date_index,ticker,trade,price)
blotter:  numpy ndarary with shape (tickers, 3), each row is a ticker of columns (total_cost, position, realized_PL), total_cost and position can be negative by default which means can do short selling.
Input argument:
n_tickers: int, number of tickers to create blotter. Note that the size of blotter is decided by n_tickers and cannot add tickers once initialized.
max_position: maximum allowed position
min_position: maximum allowed position
"""
class position():# numpy version
    def __init__(self, n_tickers,max_pos=None,min_pos=None):
        self.log=[]
        self.blotter=np.zeros((n_tickers,3))#[cost,positioin,PL]
        self.max_position = max_pos if max_pos else np.repeat(np.iinfo(np.int).max,n_tickers)
        self.min_position = min_pos if min_pos else np.repeat(np.iinfo(np.int).min,n_tickers)

    """
    This method do two things, First, just append trade activity to log. Second, adjust blotter according to the trading
    If the trading direction is different from the position, i.e the diffent sign, then use the average unit cost for to calculate realize profit/loss.
    date_index: int, the index referring trade date
    ticker: int, refferring which ticker to update in blotter
    trade: int, positive for buy, negative for sell
    price: float, price to buy/sell
    """
    def append(self,date_index,ticker,trade,price):
        assert ticker<=self.blotter.shape[0]
        self.log.append((date_index,ticker,trade,price))
        total_cost, position, realized_PL = self.blotter[ticker]
        if trade>0:
            trade=min(self.max_position[ticker]-position, trade)
        else:
            trade=max(self.min_position[ticker]-position, trade)
        new_total_cost, new_position, PL = self.update_cost(total_cost, position, price, trade)
        realized_PL += PL
        self.blotter[ticker] = new_total_cost, new_position, realized_PL
        #return realized_PL
    """
    回傳更新後的total cost, position與實現損益如交易方向與淨部位相反，則以平均成本法減去部位成本
    若部位為空，則成本為負值
    Return updated (total_cost, position, PL) according to inputs
    
    """
    def update_cost(self,total_cost,position,price,trade):
        average_cost=total_cost/position if position!=0 else 0
        new_position=position+trade
        if position*trade<0:# 交易方向與現有部位不同向 trade and position are different sign 
            if abs(trade)>abs(position):#反向交易超過原有部位，會產生另一方向部位。 Size of trade is over position, thus create opposite position               
                new_total_cost = new_position *price
                PL = (price - abs(average_cost)) * position
            else:
                new_total_cost = total_cost + trade*average_cost
                PL = (price - abs(average_cost)) * -trade
            
        else: #交易方向與現有部位同向 Position and trade are the same sign
            new_total_cost = total_cost + price*trade
            PL=0
        return new_total_cost, new_position, PL
    """
    以下四個method為回傳1d ndarray以表示各個ticker
    The following four method returns 1d ndarray of each ticker.
    """
    def MTM(self,prices):
        assert self.blotter.shape[0]==len(prices)      
        return self.marketValue(prices) - self.cost()         
    def cost(self):
        return self.blotter[:,0]
    def position(self,tickers=None):
        return self.blotter[:,1]
    def PL(self,tickers=None):
        return self.blotter[:,2]
    def marketValue(self,prices):
        assert self.blotter.shape[0]==len(prices)
        return self.position() * prices
    """
    以下四個method為回傳總和(scalar)
    The following four method returns the total of each tickers.
    """
    def total_MTM(self,prices):
        return sum(self.MTM(prices))
    def total_cost(self):
        return sum(self.cost())
    def total_PL(self):
        return sum(self.PL())
    def total_MarketValue(self,prices):
        return sum(self.marketValue(prices))




