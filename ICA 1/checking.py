a = []
b = []
c = []
for i in range(10240):
    a.append(2*i)
    b.append(2*i + 1)
    c.append((2*i) * (2*i + 1))

print(f"{c[0]} and {c[-1]}")
