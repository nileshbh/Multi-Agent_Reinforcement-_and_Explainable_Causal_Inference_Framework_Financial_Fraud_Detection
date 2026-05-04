# step4_marl_agents.py
import numpy as np, pandas as pd, random, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

# ---------- 1. LOAD ENVIRONMENT DATA ----------
df = pd.read_csv("engineered_data.csv")
features = ['Amount_log','AvgAmount_t','Freq_t','Lag_Time']
states = df[features].values
labels = df['Class'].values  # reward proxy

print("\n✅ Environment ready with", states.shape[0], "transactions and", states.shape[1], "features.")

# ---------- 2. DEFINE SIMPLE ENVIRONMENT ----------
class FraudEnv:
    def __init__(self, states, labels):
        self.states, self.labels = states, labels
        self.n = len(states); self.ptr = 0
    def reset(self):
        self.ptr = 0
        return self.states[self.ptr]
    def step(self, action):
        reward = 1 if (action == self.labels[self.ptr]) else -1
        self.ptr += 1
        done = self.ptr >= self.n
        next_state = self.states[self.ptr-1] if not done else None
        return next_state, reward, done

env = FraudEnv(states, labels)

# ---------- 3. AGENT A: Q-Learning ----------
class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=0.9, eps=0.1):
        self.Q = {}; self.alpha, self.gamma, self.eps, self.n_actions = alpha, gamma, eps, n_actions
    def get_Q(self, s): return self.Q.get(tuple(np.round(s,2)), np.zeros(self.n_actions))
    def act(self, s):
        if random.random() < self.eps: return random.randint(0, self.n_actions-1)
        return np.argmax(self.get_Q(s))
    def learn(self, s, a, r, s_next):
        q = self.get_Q(s)
        q_next = self.get_Q(s_next) if s_next is not None else np.zeros(self.n_actions)
        q[a] += self.alpha*(r + self.gamma*np.max(q_next) - q[a])
        self.Q[tuple(np.round(s,2))] = q

# ---------- 4. AGENT B: DQN ----------
class DQNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs,64), nn.ReLU(),
                                 nn.Linear(64,32), nn.ReLU(),
                                 nn.Linear(32,n_actions))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self,n_inputs,n_actions=2):
        self.model = DQNetwork(n_inputs,n_actions)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
    def act(self,s): 
        with torch.no_grad(): return int(torch.argmax(self.model(torch.tensor(s,dtype=torch.float32))))
    def learn(self,s,a,r,s_next,done):
        s = torch.tensor(s,dtype=torch.float32)
        target = r if done else r + 0.9*torch.max(self.model(torch.tensor(s_next,dtype=torch.float32)))
        pred = self.model(s)[a]
        loss = self.loss_fn(pred, torch.tensor(target))
        self.optim.zero_grad(); loss.backward(); self.optim.step()

# ---------- 5. AGENT C: Actor-Critic ----------
class ActorCritic(nn.Module):
    def __init__(self,n_states,n_actions):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(n_states,64),nn.ReLU(),
                                   nn.Linear(64,n_actions),nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(n_states,64),nn.ReLU(),
                                    nn.Linear(64,1))
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    def act(self,s):
        probs = self.actor(torch.tensor(s,dtype=torch.float32))
        return int(torch.multinomial(probs,1).item()), probs
    def learn(self,s,r,done):
        s = torch.tensor(s,dtype=torch.float32)
        value = self.critic(s)
        target = torch.tensor(r) + (0.9 * value * (1-int(done)))
        loss = (target - value)**2
        self.opt.zero_grad(); loss.mean().backward(); self.opt.step()

# ---------- 6. TRAIN LOOP ----------
def train_agents(episodes=1):
    q_agent = QLearningAgent(); dqn = DQNAgent(states.shape[1]); ac = ActorCritic(states.shape[1],2)
    rewards_A, rewards_B, rewards_C = [],[],[]

    for ep in range(episodes):
        s = env.reset(); done=False
        totalA=totalB=totalC=0
        while not done:
            # Agent A
            aA = q_agent.act(s)
            s_next, r, done = env.step(aA)
            q_agent.learn(s,aA,r,s_next); totalA += r

            # Agent B
            aB = dqn.act(s)
            if s_next is not None: dqn.learn(s,aB,r,s_next,done)
            totalB += r

            # Agent C
            aC, probs = ac.act(s)
            ac.learn(s,r,done); totalC += r

            s = s_next if s_next is not None else env.reset()
        rewards_A.append(totalA); rewards_B.append(totalB); rewards_C.append(totalC)
        print(f"Episode {ep+1}: A={totalA}, B={totalB}, C={totalC}")

    return rewards_A,rewards_B,rewards_C

rewardsA,rewardsB,rewardsC = train_agents(episodes=5)

# ---------- 7. PLOT LEARNING CURVES ----------
plt.figure(figsize=(7,4))
plt.plot(rewardsA,label='Agent A (Q-Learning)')
plt.plot(rewardsB,label='Agent B (DQN)')
plt.plot(rewardsC,label='Agent C (Actor-Critic)')
plt.title('Learning Curves of MARL Agents')
plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.tight_layout(); plt.show()

print("\n✅ Step 4 MARL training simulation completed.")
