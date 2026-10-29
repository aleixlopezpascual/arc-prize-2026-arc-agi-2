import math
# (version, local eval % [from docs/experiments.md], public LB % [from auto_submit.log])
rows=[("v5",2.33,30.69),("v6",2.33,28.47),("v7",2.33,30.97),("v8",2.33,30.97),
("v12",1.16,29.31),("v13",3.49,30.14),("v15",6.40,28.06),("v16",6.98,29.31),
("v17",6.98,26.94),("v18",7.56,25.00),("v19",8.14,26.39),("v20",8.14,25.97),
("v21",8.14,24.44),("v22",8.14,22.78),("v23",8.72,23.33),("v24",9.30,22.78),
("v25",None,23.47)]
rows=[r for r in rows if r[1] is not None]
x=[r[1] for r in rows]; y=[r[2] for r in rows]
def pear(a,b):
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    c=sum((i-ma)*(j-mb) for i,j in zip(a,b))
    return c/math.sqrt(sum((i-ma)**2 for i in a)*sum((j-mb)**2 for j in b))
def rank(v):
    s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
    i=0
    while i<len(s):
        j=i
        while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
        for k in range(i,j+1): r[s[k]]=(i+j)/2+1
        i=j+1
    return r
n=len(rows)
print("n =",n)
print("Pearson  r =",round(pear(x,y),3))
print("Spearman r =",round(pear(rank(x),rank(y)),3))
# slope
mx,my=sum(x)/n,sum(y)/n
slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x)
print("LB change per +1pt local eval: %.2f"%slope)
print("Local eval range %.2f -> %.2f ; LB first %.2f last %.2f"%(min(x),max(x),y[0],y[-1]))
