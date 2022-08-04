""""
    DQN with Tensorflow
    DQN的改进：
        1、Experience Replay
        2、固定Q_target，多次迭代后将Q估计(q_eval)的参数赋给Q现实(q_target)。
        3、用神经网络进行Q值函数拟合
        4、用经验回放缓存进行离线更新、
        5、目标网络和延迟更新
        6、用均方误差最小化TD_error。
"""
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepQNetwork:


    def __init__(
            self,
            n_actions,  #输出多少action
            n_features,   #接受多少个observation
            learning_rate=0.01, #学习率
            reward_decay=0.9, #奖励折扣因子
            e_greedy=0.9,
            replace_target_iter=300, #每300步更新目标网络
            memory_size=500, #经验池容量
            batch_size=32,  #样本数量
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter   #隔多少步将target_net的参数变成最新参数
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]  初始化记忆库
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')  #获取Q现实网络参数
        e_params = tf.get_collection('eval_net_params')   #获取Q估计网络参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  #网络参数更新

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)    #tensorboard

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    #构建网络
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        #神经网络的2个输入：
        #1、输入state，通过神经网络分析得到预测值，如state对应有2个action，则输出对应的q值。
        #2、输入q现实，通过q现实和q估计之间的误差计算，然后反向传递回去 ，提升参数。
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input接受observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss用来接受Q_target的值
        #两层网络L1,L2，神经元10个，第二层有多少动作输出多少
        #variable_scope（）用于定义创建变量的操作的上下文管理器
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables，第一层有10个神经元
            #\表示没有[],()的换行
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # q_eval第一层. collections 在更新target_net时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            #q_eval第二层，有多少个行为就有多少个值
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        #求误差--均方差
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        #梯度下降
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        #由于q现实值的输入是下一个状态，因此该网络的输入是s_
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
    #存储记忆
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):#hasattr函数用来判断对象是否包含对应的属性，如果对象有该属性则返回true，否则false。
            self.memory_counter = 0
        #记录一条[s,a,r,s_]记录
        transition = np.hstack((s, [a, r], s_))

        # 旧memory会被新memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
       # self.memory = deque(maxlen=self.memory_size)

        self.memory_counter += 1
    #选择动作
    def choose_action(self, observation):
        # 如果随机生成的数小于epsilon则按照Q现实(q_eval)最大值对应的索引作为action，否则在动作空间中随机产生动作
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    #agent学习
    def learn(self):
        # 检查是否替换target_net的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)  #判断要不要换参数
            print('\ntarget_params_replaced\n')

        # 随机抽取多少个记忆变成batch_memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        #获取q_next(target_net产生的q)和q_eval(eval_net产生的q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        #先让target=eval
        q_target = q_eval.copy()
        #返回一个长度为self.batch_size的索引值列表[0,1,2,....31]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #返回一个长度为32的动作列表，从抽取的记忆库batch_memory中的标记的第2列，self.n_features=2
        #即RL.store_transition(observation,action,reward,observation_)中的action
        #从0开始记，所以eval_act_index得到的是action的那一列。
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =  
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        最后将(q_target-q_eval)看作误差，反向传递给神经网络
        其中：所有为0的action是当时没有选择的action,只有之前有选择的action才有不为0的值；因此只反向传递不为0的action
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost) #记录cost误差

        # increasing epsilon 增加epsilon，降低行为的随机性。
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



