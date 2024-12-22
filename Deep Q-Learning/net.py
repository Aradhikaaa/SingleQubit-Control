import numpy as np
import tensorflow as tf
import os
os.system("clear")

class DeepQNetwork(tf.keras.Model):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None):
        """
        Parameters:
        n_actions (int): number of possible actions
        n_features (int): number of features in the state
        learning_rate (float): learning rate for the neural network
        reward_decay (float): discount factor for future rewards
        e_greedy (float): probability of choosing the greedy action
        replace_target_iter (int): interval for replacing the target network
        memory_size (int): size of the experience replay buffer
        batch_size (int): size of the batches for training
        e_greedy_increment (float or None): increment in the epsilon value for each step
        """
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features*2 + 2), dtype=np.float32)


        self._build_net()
        self.target_model.set_weights(self.eval_model.get_weights())
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.cost_his = []

    def _build_net(self):
        """
        Builds the evaluation network and the target network, which are used
        for selecting actions and computing the target Q-values respectively.
        The evaluation network is a 3-layer fully connected neural network with
        ReLU activations, and the target network is a copy of the evaluation network.
        """
        self.eval_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_features,)),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
                                  bias_initializer=tf.keras.initializers.Constant(0.1)),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
                                  bias_initializer=tf.keras.initializers.Constant(0.1)),
            tf.keras.layers.Dense(self.n_actions,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
                                  bias_initializer=tf.keras.initializers.Constant(0.1))
        ])

        # Target Network
        self.target_model = tf.keras.models.clone_model(self.eval_model)
        self.target_model.set_weights(self.eval_model.get_weights())

    def store_transition(self, s, a, r, s_):
        """
        Stores a transition in the experience replay buffer.

        Parameters:
        s (numpy.array): Current state
        a (int): Action taken
        r (float): Reward received
        s_ (numpy.array): Next state

        Returns:
        None
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        #print("Shape of action a :", a.shape)
        #print("Shape of reward r:", r.shape)
        transition = np.hstack((s, [a, r], s_))
        #print("Shape of state:", s.shape)
        #print("Shape of action:", transition[1].shape)
        #print("Shape of reward:", transition[2].shape)
        #print("Shape of next state:", s_.shape)
        
        #print("Shape of transition:", transition.shape)
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        """
        Chooses an action based on the observation.

        Parameters:
        observation (numpy.array): Observation from the environment

        Returns:
        int: Action chosen
        """
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_values = self.eval_model(observation)
            action = np.argmax(q_values[0])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        """
        Learns from the experience replay buffer.

        Samples a batch of experiences from the buffer, computes the target
        Q-values using the target network, and computes the loss between the
        target Q-values and the predicted Q-values from the evaluation network.
        The loss is then used to update the evaluation network.

        Parameters:
        None

        Returns:
        None
        """
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model(batch_memory[:, -self.n_features:], training=False)
        q_eval = self.eval_model(batch_memory[:, :self.n_features], training=False)

        q_target = q_eval.numpy().copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next.numpy(), axis=1)

        with tf.GradientTape() as tape:
            q_eval = self.eval_model(batch_memory[:, :self.n_features], training=True)
            loss = self.loss_fn(q_target, q_eval)

        grads = tape.gradient(loss, self.eval_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.eval_model.trainable_variables))

        self.cost_his.append(loss.numpy())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
