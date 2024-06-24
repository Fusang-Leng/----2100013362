## 排序算法

冒泡排序

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # 经过i轮冒泡后，后i个元素已经有序，不需要再比较
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                # 如果前一个元素比后一个元素大，则交换它们的位置
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

选择排序的基本思路是每次从未排序的数列中选择最小（或最大）的元素（最小即升序，最大即降序），将其放到已排序数列的末尾。重复这个过程，直到所有元素都被排序为止。选择排序是一种简单但效率较低的排序算法，其时间复杂度为O(n^2)，不适用于大规模数据的排序。

```python
#选择排序
arr = [8,3,2,6,1,4,9,7]
for i in range(0,len(arr)):
	for j in range(i+1,len(arr)):
		if arr[i] >= arr[j]:
			arr[i],arr[j] = arr[j],arr[i]
print(arr)
```

插入排序（Insertion Sort）的基本思路是将一个未排序的数列，逐个插入到已排序的数列中，使得插入后的数列仍然有序。插入排序的优点是实现简单，适用于小规模的数据排序，时间复杂度为O(n^2)，空间复杂度为O(1)。但是，对于大规模数据的排序，插入排序的效率较低。

2、实现步骤
（1）设数组长度为n。

（2）将第1个元素看成已排序部分，第2个元素到第n个元素看成未排序部分。

（3）从未排序部分取出第1个元素,将其插入已排序部分的合适位置。此时已排序部分的长度为1。

（4）从未排序部分取出第2个元素将其插入已排序部分的合适位置。此时已排序部分的长度为2。

（5）重复步骤4，直到所有元素都被插入到已排序部分。、

```
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        # 获取当前位置的值
        key = arr[i]
        # 将当前位置的值插入已排序部分的合适位置
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr
```

归并排序（可用于求逆序对数）

归并排序是一种基于分治策略的排序算法。它的基本思想是将待排序的序列不断地分成两个子序列，直到每个子序列只有一个元素，然后将这些子序列进行合并，直到最终得到一个有序序列。归并排序的时间复杂度为O(nlogn)，其中n为待排序序列的长度。归并排序是稳定的排序算法，即相等元素的相对位置不会改变。

2、实现步骤
（1）将序列中待排序数字分为若干组，每个数字分为一组。

（2）将若干组两两合并，保证合并的组都是有序的。

（3）重复第二步的操作，直到剩下最后一组即为有序数列

```python
# start--mid 和 mid+1--end 都是sorted list
def Merge(a,start,mid,end):
	tmp=[]
	l=start
	r=mid+1
	while l<=mid and r<=end:
		if a[l]<=a[r]:
			tmp.append(a[l])
			l+=1
		else:
			tmp.append(a[r])
			r+=1
	# 以下至少有一个extend了空列表
	tmp.extend(a[l:mid+1])
	tmp.extend(a[r:end+1])
	for i in range(start,end+1):
		a[i]= tmp[i-start]

# 二分
def MergeSort(a,start,end):
	if start==end:
		return

	mid=(start+end)//2
	MergeSort(a,start,mid)
	MergeSort(a,mid+1,end)
	Merge(a,start,mid,end)

a=[8,5,6,4,3,7,10,2]
MergeSort(a,0,7)
print(a)
```

快速排序

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left, left_swaps = merge_sort(arr[:mid])
    right, right_swaps = merge_sort(arr[mid:])
    
    merged = []
    i = j = swaps = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            swaps += len(left) - i
    
    merged += left[i:]
    merged += right[j:]
    
    total_swaps = left_swaps + right_swaps + swaps
    return merged, total_swaps

while True:
    n = int(input())
    if n == 0:
        break
    arr = [int(input()) for _ in range(n)]
    _, swaps = merge_sort(arr)
    print(swaps)
```



### 排序、栈、队列

#### 逆波兰表达式求值

```python
stack=[]
for t in s:
    if t in '+-*/':
        b,a=stack.pop(),stack.pop()
        stack.append(str(eval(a+t+b)))
    else:
        stack.append(t)
print(f'{float(stack[0]):.6f}')
```

#### 中序表达式转后序表达式

```python
pre={'+':1,'-':1,'*':2,'/':2}
for _ in range(int(input())):
    expr=input()
    ans=[]; ops=[]
    for char in expr:
        if char.isdigit() or char=='.':
            ans.append(char)
        elif char=='(':
            ops.append(char)
        elif char==')':
            while ops and ops[-1]!='(':
                ans.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1]!='(' and pre[ops[-1]]>=pre[char]:
                ans.append(ops.pop())
            ops.append(char)
    while ops:
        ans.append(ops.pop())
    print(''.join(ans))
```

#### 最大全0子矩阵

```python
for row in ma:
    stack=[]
    for i in range(n):
        h[i]=h[i]+1 if row[i]==0 else 0
        while stack and h[stack[-1]]>h[i]:
            y=h[stack.pop()]
            w=i if not stack else i-stack[-1]-1
            ans=max(ans,y*w)
        stack.append(i)
    while stack:
        y=h[stack.pop()]
        w=n if not stack else n-stack[-1]-1
        ans=max(ans,y*w)
print(ans)
```

#### 求逆序对数

```python
from bisect import *
a=[]
rev=0
for _ in range(n):
    num=int(input())
    rev+=bisect_left(a,num)
    insort_left(a,num)
ans=n*(n-1)//2-rev
```

```python
def merge_sort(a):
    if len(a)<=1:
        return a,0
    mid=len(a)//2
    l,l_cnt=merge_sort(a[:mid])
    r,r_cnt=merge_sort(a[mid:])
    merged,merge_cnt=merge(l,r)
    return merged,l_cnt+r_cnt+merge_cnt
def merge(l,r):
    merged=[]
    l_idx,r_idx=0,0
    inverse_cnt=0
    while l_idx<len(l) and r_idx<len(r):
        if l[l_idx]<=r[r_idx]:
            merged.append(l[l_idx])
            l_idx+=1
        else:
            merged.append(r[r_idx])
            r_idx+=1
            inverse_cnt+=len(l)-l_idx
    merged.extend(l[l_idx:])
    merged.extend(r[r_idx:])
    return merged,inverse_cnt
```

#### 单调栈py

```py
n = int(input())
a = list(map(int,input().split()))
stack = []
for i in range(n):
    while stack and a[stack[-1]]<a[i]: # 注意pop前要检查栈是否非空
        a[stack.pop()] = i+1 # 原地修改，较为简洁
    stack.append(i) # stack存元素下标而非元素本身
for x in stack:
    a[x] = 0
print(*a)
```



```python
#给定一个列表，输出每个元素之前小于它的最后一个元素的下标
def solve(lis):
    n=len(lis)
    stack=[]
    ans=[]
    for i in range(n):
        x=lis[i]
        while stack:
            (y,j)=stack[-1]
            if y>=x:
                stack.pop()
                continue
            break
        if not stack:
            stack.append((x,i))
            ans.append(-1)
        else:
            ans.append(stack[-1][1])
            stack.append((x,i))
    return ans
```

#### 奶牛排队

```python
N,res=int(input()),0
hi=[int(input()) for _ in range(N)]
# left[i]是i左边第一个不小于他的元素的索引，right[i]是i右边第一个不大于他的元素的索引。
# 容易知道，对于指定的i，如果i作为右端点，left[i]是左端点的一个上界，反之同理。
left,right=[-1 for _ in range(N)],[N for _ in range(N)]
stack1,stack2=[],[]

for i in range(N-1,-1,-1):
    while stack1 and hi[stack1[-1]]>hi[i]:
        stack1.pop()
    if stack1:right[i]=stack1[-1]
    stack1.append(i)

for i in range(N):
    while stack2 and hi[stack2[-1]]<hi[i]:
        stack2.pop()
    if stack2:left[i]=stack2[-1]
    stack2.append(i)

for i in range(N):
    for j in range(right[i]-1,i,-1):
        if left[j]<i:
            res=max(j-i+1,res)
            break

print(res)
```

#### 大小堆

```python
import heapq
def find_median(num):
    bigger_heap = []
    smaller_heap = []

    median = []
    for i,num in enumerate(num):
        if not bigger_heap or num < -bigger_heap[0]:
            heapq.heappush(bigger_heap, -num)
        else:
            heapq.heappush(smaller_heap, num)
        
        if len(smaller_heap)>len(bigger_heap):
            heapq.heappush(bigger_heap, -heapq.heappop(smaller_heap))
        if len(bigger_heap)-len(smaller_heap)>1:
            heapq.heappush(smaller_heap,-heapq.heappop(bigger_heap))
    
        if i%2 == 0:
            median.append(-bigger_heap[0])
    return median

T = int(input())
for _ in range(T):
    num = list(map(int,input().split()))
    print(len(find_median(num)))
    print(*find_median(num))
```

#### 各种表达式

###### 后缀表达

```py
def cal(a,b,operate):
    if operate=="+":return a+b
    if operate=="-":return a-b
    if operate=="*":return a*b
    if operate=="/":return a/b

from collections import deque
n,operators=int(input()),("+",'-','*','/')
raw=[deque(map(str,input().split())) for _ in range(n)]

for deq in raw:
    tmp_deq=deque()
    while len(deq)>=1:
        if deq[0] not in operators:
            tmp_deq.append(float(deq.popleft()))
        else:
            b=tmp_deq.pop()
            a=tmp_deq.pop()
            operate=deq.popleft()
            deq.appendleft(cal(a,b,operate))
    print('{:.2f}'.format(tmp_deq[0]))
```

##### 中缀转后缀不带括号

```py
def infix_to_postfix(expression):
    def get_precedence(op):
        precedences = {'+': 1, '-': 1, '*': 2, '/': 2}
        return precedences[op] if op in precedences else 0

    def is_operator(c):
        return c in "+-*/"

    def is_number(c):
        return c.isdigit() or c == '.'

    output = []
    stack = []
    number_buffer = []
    
    def flush_number_buffer():
        if number_buffer:
            output.append(''.join(number_buffer))
            number_buffer.clear()
	
    # 主体部分
    for c in expression:
        if is_number(c):
            number_buffer.append(c)
        elif c == '(':
            flush_number_buffer()
            stack.append(c)
        elif c == ')':
            flush_number_buffer()
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # popping '('
        elif is_operator(c):
            flush_number_buffer()
            while stack and get_precedence(c) <= get_precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(c)

    flush_number_buffer()
    while stack:
        output.append(stack.pop())

    return ' '.join(output)

# Read number of expressions
n = int(input())
# Read each expression and convert it
for _ in range(n):
    infix_expr = input()
    postfix_expr = infix_to_postfix(infix_expr)
    print(postfix_expr)
```

##### 带括号

```py
operators=['+','-','*','/']
cals=['(',')']
# 预处理数据的部分已省略。
def pre_to_post(lst):
    s_op,s_out=[],[]
    while lst:
        tmp=lst.pop(0)
        if tmp not in operators and tmp not in cals:
            s_out.append(tmp)
            continue

        if tmp=="(":
            s_op.append(tmp)
            continue

        if tmp==")":
            while (a:=s_op.pop())!="(":
                s_out.append(a)

        if tmp in operators:
            if not s_op:
                s_op.append(tmp)
                continue
            if is_prior(tmp,s_op[-1]) or s_op[-1]=="(":
                s_op.append(tmp)
                continue
            while (not (is_prior(tmp,s_op[-1]) or s_op[-1]=="(")
                or not s_op):
                s_out.append(s_op.pop())
            s_op.append(tmp)
            continue

    while len(s_op)!=0:
        tmp=s_op.pop()
        if tmp in operators:
            s_out.append(tmp)

    return " ".join(s_out)

def is_prior(A,B):
    if (A=="*" or A=="/") and (B=="+" or B=="-"):
        return True
    return False

def input_to_lst(x):
    tmp=list(x)

for i in range(int(input())):
    print(pre_to_post(expProcessor(input())))
```

### 树

#### 根据前中序得后序、根据中后序得前序

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
```

```python
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

#### 层次遍历

```python
from collections import deque
def levelorder(root):
    if not root:
        return ""
    q=deque([root])  
    res=""
    while q:
        node=q.popleft()  
        res+=node.val  
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res
```

#### 解析括号嵌套表达式

```python
def parse(s):
    node=Node(s[0])
    if len(s)==1:
        return node
    s=s[2:-1]; t=0; last=-1
    for i in range(len(s)):
        if s[i]=='(': t+=1
        elif s[i]==')': t-=1
        elif s[i]==',' and t==0:
            node.children.append(parse(s[last+1:i]))
            last=i
    node.children.append(parse(s[last+1:]))
    return node
```

#### 二叉搜索树的构建

```python
def insert(root,num):
    if not root:
        return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
```

#### 并查集

```python
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n
    def find(self,x):
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.h[rootx]<self.h[rooty]:
                self.p[rootx]=rooty
            elif self.h[rootx]>self.h[rooty]:
                self.p[rooty]=rootx
            else:
                self.p[rooty]=rootx
                self.h[rootx]+=1
```

#### 字典树的构建

```python
def insert(root,num):
    node=root
    for digit in num:
        if digit not in node.children:
            node.children[digit]=TrieNode()
        node=node.children[digit]
        node.cnt+=1
```

#### huffuman树的构建

```python
import heapq
class HuffmanTreeNode:
    def __init__(self,weight,char=None):
        self.weight=weight
        self.char=char
        self.left=None
        self.right=None

    def __lt__(self,other):
        return self.weight<other.weight

def BuildHuffmanTree(characters):
    heap=[HuffmanTreeNode(weight,char) for char,weight in characters.items()]
    heapq.heapify(heap)
    while len(heap)>1:
        left=heapq.heappop(heap)
        right=heapq.heappop(heap)
        merged=HuffmanTreeNode(left.weight+right.weight,None)
        merged.left=left
        merged.right=right
        heapq.heappush(heap,merged)
    root=heapq.heappop(heap)
    return root

def enpaths_huffman_tree(root):
    # 字典形如(idx,weight):path
    paths={}
    def traverse(node,path):
        if node.char:
            paths[(node.char,node.weight)]=path
        else:
            traverse(node.left,path+1)
            traverse(node.right,path+1)
    traverse(root,0)
    return paths

def min_weighted_path(paths):
    return sum(tup[1]*path for tup,path in paths.items())

n,characters=int(input()),{}
raw=list(map(int,input().split()))
for char,weight in enumerate(raw):
    characters[str(char)]=weight
root=BuildHuffmanTree(characters)
paths=enpaths_huffman_tree(root)
print(min_weighted_path(paths))
```

#### trie结构

```python
# 
for ____ in range(int(input())):
    trie={}
    n=int(input())
    lis=[]
    for _ in range(n):
        lis.append(input())
    lis.sort(key=len,reverse=True)
    flag=1
    for w in lis:
        temp=trie;l=len(w);f=1
        for x in w:
            if x in temp:
                temp=temp[x]
            else:
                f=0
                temp[x]={}
                temp=temp[x]
        if f:
            flag=0
    if flag:
        print("YES")
    else:
        print("NO")
```



### 图

#### bfs

```python
from collections import deque
def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    visited.add(start_node)
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

#### 棋盘问题（回溯法）

```python
def dfs(row, k):
    if k == 0:
        return 1
    if row == n:
        return 0
    count = 0
    for col in range(n):
        if board[row][col] == '#' and not col_occupied[col]:
            col_occupied[col] = True
            count += dfs(row + 1, k - 1)
            col_occupied[col] = False
    count += dfs(row + 1, k)
    return count
col_occupied = [False] * n
print(dfs(0, k))
```

#### dijkstra

```python
# 1.使用vis集合
def dijkstra(start,end):
    heap=[(0,start,[start])]
    vis=set()
    while heap:
        (cost,u,path)=heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==end: return (cost,path)
        for v in graph[u]:
            if v not in vis:
                heappush(heap,(cost+graph[u][v],v,path+[v]))
# 2.使用dist数组
import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

#### kruskal

```python
class DisjointSetUnion:
    def __init__(self, n):
        # 初始化每个节点的父节点为其自身，并初始化每个节点的秩(rank)为0
        self.parent = list(range(n))
        self.rank = [0] * n

    # 寻找节点x的根节点，使用路径压缩优化
    def find(self, x):
        if self.parent[x] != x:
            # 递归地将节点x的父节点设为其根节点
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # 合并两个节点x和y所属的集合
    def union(self, x, y):
        # 找到x和y的根节点
        xp = self.find(x)
        yp = self.find(y)
        if xp == yp:
            return False
        # 根据秩(rank)决定如何合并
        elif self.rank[xp] < self.rank[yp]:
            self.parent[xp] = yp
        elif self.rank[xp] > self.rank[yp]:
            self.parent[yp] = xp
        else:
            self.parent[yp] = xp
            self.rank[xp] += 1
        return True

# Kruskal算法，用于找到最小生成树（MST）
def kruskal(n, edges):
    dsu = DisjointSetUnion(n)
    mst_weight = 0
    # 按照边的权重从小到大排序
    for weight, u, v in sorted(edges):
        # 如果边(u, v)连接的两个节点不在同一个集合中，则合并它们并更新MST权重
        if dsu.union(u, v):
            mst_weight += weight
    return mst_weight


while True:
    try:
        n = int(input().strip())
        edges = []
        for i in range(n):
            row = list(map(int, input().split()))
            for j in range(i + 1, n):
                if row[j] != 0:
                    # 只添加非零权重的边
                    edges.append((row[j], i, j))
        # 计算并输出最小生成树的总权重
        print(kruskal(n, edges))
    except EOFError:
        break
```

#### prim

```python
vis=[0]*n
q=[(0,0)]
ans=0
while q:
    w,u=heappop(q)
    if vis[u]:
        continue
    ans+=w
    vis[u]=1
    for v in range(n):
        if not vis[v] and graph[u][v]!=-1:
            heappush(q,(graph[u][v],v))
print(ans)
```

#### 拓扑排序

```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return []  
    return topo_order
```

#### 判断无向图是否连通、有无回路

```py
def connect_graph(n,graph):
    visited = [False]*n
    visited[0] = True
    stack = [0]

    while stack:
        node = stack.pop()
        for nbr in graph[node]:
            if not visited[nbr]:
                stack.append(nbr)
                visited[nbr] = True

    return all(visited)

def loop(n,graph):
    visited = [False]*n
    def dfs(node,visited,parent):
        visited[node] = True
        for nbr in graph[node]:
            if not visited[nbr]:
                if dfs(nbr,visited,node):
                    return True
            elif parent != nbr:
                return True
        return False
    
    for node in range(n):
        if not visited[node]:
            if dfs(node,visited,-1):
                return True
        
    return False

n, m = map(int,input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u,v = map(int,input().split())
    graph[u].append(v)
    graph[v].append(u)

print("connected:yes" if connect_graph(n,graph) else "connected:no")
print("loop:yes" if loop(n,graph) else "loop:no")
```

### 题目

#### 拦截导弹

```python
k=int(input())
l=list(map(int,input().split()))
dp=[0]*k
for i in range(k-1,-1,-1):
    maxn=1
    for j in range(k-1,i,-1):
        if l[i]>=l[j] and dp[j]+1>maxn:
            maxn=dp[j]+1
    dp[i]=maxn
print(max(dp))
```

#### 欧拉筛

```py
def euler(r):
    prime = [0 for i in range(r+1)]
    common = []
    for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
            if i % j == 0:
                break
    return prime 
```

#### 合法出栈序列

```python
def legit(a,x):
    if len(x)!=len(a) or sorted(a)!=sorted(x):
        return "NO"
    x1=[]
    a1=[]
    for i in x:
        x1.append(i)
    num=0
    for i in a:
        a1.append(i)
        while a1 and a1[-1] == x1[num]:
            num+=1
            a1.pop()
    if num==len(x):
        return "YES"
    else:
        return "NO" 



a=input()
while True:
    try:
        x=input()
        print(legit(a,x))
    except BaseException:
        break
```

#### 小组队列

```python
from collections import deque					# 时间: 105ms

# Initialize groups and mapping of members to their groups
t = int(input())
groups = {}
member_to_group = {}



for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]  # Assuming the first member's ID represents the group ID
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id

# Initialize the main queue to keep track of the group order
queue = deque()
# A set to quickly check if a group is already in the queue
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        # Create a new group if it's a new member not in the initial list
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]:  # If the group's queue is empty, remove it from the main queue
                queue.popleft()
                queue_set.remove(group)
```

#### 遍历树

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def traverse_print(root, nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value: root}
    for child in root.children:
        pac[child] = nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traverse_print(pac[value], nodes)
        else:
            print(root.value)

n = int(input())
nodes = {}
children_list = []
for i in range(n):
    info = list(map(int, input().split()))
    nodes[info[0]] = TreeNode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
traverse_print(root, nodes)
```

#### 图的

```python
class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=1):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices += 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, key):
        if key in self.vertList:
            return self.vertList[key]
        else:
            return None

    def __contains__(self, key):
        return key in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        self.vertList[t].addNeighbor(self.vertList[f], cost)

    def getVertices(self):
        return self.vertList.keys()


def laplacian_matrix(graph,n):
    laplacian = [[0] * n for _ in range(n)]
    
    for vertex_id in graph.getVertices():
        vertex = graph.getVertex(vertex_id)
        degree = len(vertex.getConnections())
        laplacian[vertex_id][vertex_id] = degree
        
        for neighbor_id in vertex.getConnections():
            laplacian[vertex_id][neighbor_id.getId()] = -1
    
    return laplacian


def main():
    n, m = map(int, input().split())
    graph = Graph()
    
    for _ in range(m):
        a, b = map(int, input().split())
        graph.addEdge(a, b)
    
    laplacian = laplacian_matrix(graph,n)
    
    for row in laplacian:
        print(*row)


if __name__ == "__main__":
    main()
```

#### 鸣人和佐助

```python
 from collections import deque
def bfs():
    q = deque([start + (T, 0)])##用t代表剩下的查克拉
    
    visited = [[-1]*N for i in range(M)]
    visited[start[0]][start[1]] = T
    while q:
        x, y, t, time = q.popleft()
        time += 1
        for dx, dy in direc:
            if 0<=x+dx<M and 0<=y+dy<N:
                if (elem := graph[x+dx][y+dy]) == '*' and t > visited[x+dx][y+dy]:
                    visited[x+dx][y+dy] = t
                    q.append((x+dx, y+dy, t, time))
                elif elem == '#' and t > 0 and t-1 > visited[x+dx][y+dy]:
                    visited[x+dx][y+dy] = t-1
                    q.append((x+dx, y+dy, t-1, time))
                elif elem == '+':
                    return time
    return -1


M, N, T = map(int, input().split())
graph = [list(input()) for i in range(M)]
direc = [(0,1), (1,0), (-1,0), (0,-1)]
start, end = None, None
for i in range(M):
    for j in range(N):
        if graph[i][j] == '@':
            start = (i, j)
print(bfs())
```

#### 八皇后

```
def solve_n_queens():
    solutions = []  #所有解
    queens = [-1] * 8 
    
    def backtrack(row):
        if row == 8:  # +1等于8时说明已经全部完成 存入
            solutions.append(queens.copy())
        else:
            for col in range(8):
                if is_valid(row, col):  # 检查当前位置是否合法
                    queens[row] = col  # 在当前行放置皇后
                    backtrack(row + 1)  # 递归处理下一行
                    queens[row] = -1  # 回溯，撤销当前行的选择
    
    def is_valid(row, col):
        for r in range(row):
            if queens[r] == col or abs(row - r) == abs(col - queens[r]):# valid的条件是竖着不能有相同的且行列相差不能相等
                return False
        return True
    
    backtrack(0)
    
    return solutions
```



### 工具

语法篇

数据结构：

字符串：

​    处理：分割：str.split(“（分隔符）”)

​       大写：str.upper()

​       小写：str.lower()

​       首字母：str.title()

​       合并：+

列表：

​      添加多个元素：list.extend([a,b,c,...])

​      插入元素：list.insert(index,元素)；bisect库

​      删除已知元素：list.remove(元素)

​      删除已知索引的元素：del list[index]

​      弹出元素：list.pop(index)

​      顺序排序：list.sort()（ASCII码顺序）

​      倒序排序：list.sort(reverse=True)

​      指定顺序排序：list.sort(key= lambda s:排序指标（与s相关）)

​      拼接：list1+list2

​      正数第n个元素：list[n-1]

​      倒数第n个元素：list[-n]

​      元素个数：list.count(元素)



​      itertools 库

​      判断共有特征：all(特征 for 元素 in 列表)

​      索引，元素元组：enumerate()函数（遍历方法：for index，元素代称 in enumerate(列表)）

字典：

   建立：{}

​      dict(元组)

​      半有序：Ordereddict()

   添加/修改键值对：dict[key]=value

   遍历字典的键：for 元素 in dict() ； for 元素 in dict.keys()（注：一定要加s!）

   遍历字典的值：for 元素 in dict.values()（一定要加s!）

   删除键值对：del dict[键]

遍历键值对：for key,value in dict.items():

按顺序遍历：for key in sorted(dict.keys()):



   元组：建立：直接定义：(...,...,...,...)

​       含元组的列表：zip(a,b,c,...)

​      访问：元组[index]



 集合：建立：set()

​    向集合中添加元素：set.add()

​    添加多个元素：set.update()

​    删除元素：set.remove() 或set.discard()（前者有KeyError风险，后者没有）

​    随机删除：set.pop()

​    并集：set1 | set2（竖杠“|”在回车键上方）

​    交集：set1 & set2

​    差集（补集）：set1 - set2

​    对称差集（补集之交）：set1^set2

​    元素个数：len()

​    不可变集合：frozenset()



math库：向上取整：math.ceil()

​     向下取整：math.floor()

​     阶乘：math.factoria()

​     数学常数：math.pi（圆周率），math.e（自然对数的底）

​     开平方：math.sqrt(x)

​     x的y次幂：math.pow(x,y)

​     e的x次幂：math.exp(x)

​     对数函数：math.log(真数，底数)（不填底数默认为自然对数）

​     三角：math.sin(),math.cos(),math.tan()

​     反三角：math.asin(),math.acos(),math.atan()



heapq库：列表转堆：最小值在上层：heapq.heapify(list)；最大值在上层：heapq._heapify_max(list)

​     插入元素：heapq.heappush(堆名，被插元素)

​     弹出元素：heapq.heappop(堆名)（可被命名为其他变量临时调用）

​          （应用：堆排序：a=[heapq.heappop(b) for _ in range(len(b))],返回排序后的b）

​     插入元素的同时弹出顶部元素：heapq.heappushpop(堆名，被插元素)

​     （或heapq.heapreplace(堆名，被插元素)）

​     以上操作在最大堆中应换为“_a_max”（a是它们中的任意一个）


​       建堆时，先定义一个空列表，然后一个一个往里面压入元素。



itertools库：

​     整数集：itertools.count(x,y)（从x开始往大数的整数，间隔为y）

​     循环地复制一组变量：itertools.cycle(list)

​     所有排列：itertools.permutations(集合，选取个数)

​     所有组合：itertools.combinations

​     拼接列表的另一种方式：itertools.chain(list1,list2)

​     已排序列表去重：[i for i,_ in itertools.groupby(list)]（每种元素只能保留一个）

​            或者list(group)[:n]（group被定义为分组，保留每组的n个元素）

 collections库：

​       双端队列：创建：a=deque(list)

​         

​       有序字典：Ordereddict()

​       默认值字典：a=defaultdict(默认值)，如果键不在字典中，会自动添加值为默认值的键值对，而不报KeyError。

​       计数器：Counter(str)，返回以字符种类为键，出现个数为值的字典

sys库：sys.exit()用于及时退出程序

​    sys.setrecursionlimit()用于调整递归限制（尽量少用，递归层数过多会引起MLE）

statistics库：statistics 是 Python 标准库中用于统计学计算的模块，提供了各种用于处理统计数据的函数。以下是 statistics 模块中一些常用函数的简要介绍：

1.mean(data)：计算数据的平均值（均值）。

2.harmonic_mean(data)：计算数据的调和平均数。

3.median(data)：计算数据的中位数。

4.median_low(data)：计算数据的低中位数。

5.median_high(data)：计算数据的高中位数。

6.median_grouped(data, interval=1)：计算分组数据的估计中位数。

7.mode(data)：计算数据的众数。

8.pstdev(data)：计算数据的总体标准差。

9.pvariance(data)：计算数据的总体方差。

10.stdev(data)：计算数据的样本标准差。

11.variance(data)：计算数据的样本方差。

这些函数能够帮助你在 Python 中进行常见的统计计算。在使用这些函数时，你需要将数据作为参数传递给相应的函数，然后函数会返回计算结果。

数据处理：二进制：bin()，八进制：oct()，十六进制：hex()，整型：int()，浮点型：float(),

​     保留n位小数：round(原数字，保留位数)（如不写保留位数，则默认保留到整数）；’%.nf’%原数字；’{:.nf}’.format(原数字)；

​     n位有效数字：’%.ng’%原数字；’{:.ng}’.format(原数字)

​     最大值max(),最小值min()

​     ASCII转字符：chr();字符转ASCII：ord()

​     判断数据类型：isinstance(object,class)

其他：if，while循环；try，except 某error；



int(str,n)	将字符串`str`转换为`n`进制的整数。

for key,value in dict.items()	遍历字典的键值对。

for index,value in enumerate(list)	枚举列表，提供元素及其索引。

dict.get(key,default) 	从字典中获取键对应的值，如果键不存在，则返回默认值`default`。

list(zip(a,b))	将两个列表元素一一配对，生成元组的列表。

math.pow(m,n)	计算`m`的`n`次幂。

math.log(m,n)	计算以`n`为底的`m`的对数。

lrucache	

```py
from functools import lru_cache
@lru_cache(maxsize=None)
```

bisect

```python
import bisect
# 创建一个有序列表
sorted_list = [1, 3, 4, 4, 5, 7]
# 使用bisect_left查找插入点
position = bisect.bisect_left(sorted_list, 4)
print(position)  # 输出: 2
# 使用bisect_right查找插入点
position = bisect.bisect_right(sorted_list, 4)
print(position)  # 输出: 4
# 使用insort_left插入元素
bisect.insort_left(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 5, 7]
# 使用insort_right插入元素
bisect.insort_right(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 4, 5, 7]
```

字符串

1. `str.lstrip() / str.rstrip()`: 移除字符串左侧/右侧的空白字符。

2. `str.find(sub)`: 返回子字符串`sub`在字符串中首次出现的索引，如果未找到，则返回-1。

3. `str.replace(old, new)`: 将字符串中的`old`子字符串替换为`new`。

4. `str.startswith(prefix) / str.endswith(suffix)`: 检查字符串是否以`prefix`开头或以`suffix`结尾。

5. `str.isalpha() / str.isdigit() / str.isalnum()`: 检查字符串是否全部由字母/数字/字母和数字组成。

   6.`str.title()`：每个单词首字母大写。

enumerate:

```py
print(list(enumerate(['a','b','c']))) # 输出：[(1, 2), (1, 3), (2, 3)]
```

counter：计数

```python
from collections import Counter
# 创建一个Counter对象
count = Counter(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])
# 输出Counter对象
print(count)  # 输出: Counter({'apple': 3, 'banana': 2, 'orange': 1})
# 访问单个元素的计数
print(count['apple'])  # 输出: 3
# 访问不存在的元素返回0
print(count['grape'])  # 输出: 0
# 添加元素
count.update(['grape', 'apple'])
print(count)  # 输出: Counter({'apple': 4, 'banana': 2, 'orange': 1, 'grape': 1})
```

permutations：全排列

```python
from itertools import permutations
# 创建一个可迭代对象的排列
perm = permutations([1, 2, 3])
# 打印所有排列
for p in perm:
    print(p)
# 输出: (1, 2, 3)，(1, 3, 2)，(2, 1, 3)，(2, 3, 1)，(3, 1, 2)，(3, 2, 1)
```

combinations：组合

```python
from itertools import combinations
# 创建一个可迭代对象的组合
comb = combinations([1, 2, 3], 2)
# 打印所有组合
for c in comb:
    print(c)
# 输出: (1, 2)，(1, 3)，(2, 3)
```

reduce：累次运算

```python
from functools import reduce
# 使用reduce计算列表元素的乘积
product = reduce(lambda x, y: x * y, [1, 2, 3, 4])
print(product)  # 输出: 24
```

product：笛卡尔积

```python
from itertools import product
# 创建两个可迭代对象的笛卡尔积
prod = product([1, 2], ['a', 'b'])
# 打印所有笛卡尔积对
for p in prod:
    print(p)
# 输出: (1, 'a')，(1, 'b')，(2, 'a')，(2, 'b')
```


