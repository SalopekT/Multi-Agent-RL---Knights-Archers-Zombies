import matplotlib.pyplot as plt

# X values
x = [0, 1, 2, 3, 4, 5]

# Four example lines
y1 = [4, 4, 4, 4, 4, 4]
y2 = [109, 56, 48, , ,]
y3 = [, , , , , ]
y4 = [76, 76, 76, 76, 76, 76]

# Plot lines
plt.plot(x, y1, label="random agent")
plt.plot(x, y2, label="agent trained with prosocial rewards")
plt.plot(x, y3, label="agent trained without prosocial rewards")
plt.plot(x, y4, label="diagonal agent")

# Labels and legend
plt.xlabel("distortion level")
plt.ylabel("total reward")
plt.title("various agent results")
plt.legend()

# Show plot
plt.show()