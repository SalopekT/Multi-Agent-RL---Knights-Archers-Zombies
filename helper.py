import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]

y1 = [4, 4, 4, 4, 4, 4]
y2 = [109, 56, 48, 35, 51,8]
y3 = [138, 69, 57, 32, 32, 9]
y4 = [76, 76, 76, 76, 76, 76]

plt.plot(x, y1, label="random agent")
plt.plot(x, y2, label="agent trained with prosocial rewards")
plt.plot(x, y3, label="agent trained without prosocial rewards")
plt.plot(x, y4, label="diagonal agent")

plt.xlabel("distortion level")
plt.ylabel("total reward")
plt.title("various agent results")
plt.legend()

plt.show()