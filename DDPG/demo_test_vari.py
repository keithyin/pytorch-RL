from utils.pyprocess import OrnsteinUhlenbeckProcess

ou_process = OrnsteinUhlenbeckProcess(theta=0.15, mu=0.0, sigma=0.2, size=1)

for i in range(1000):
    print(ou_process.sample())