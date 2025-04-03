# Kruskal 算法的正确性证明

Kruskal 算法是用于求解 **最小生成树（MST, Minimum Spanning Tree）** 的一种贪心算法。要证明其正确性，我们需要说明它能够正确构造最小生成树。

## Kruskal 算法的基本步骤

1. **排序**：按边的权重从小到大排序。
2. **构建树**：从权重最小的边开始，依次选择边，**只要不形成环**，就加入到生成树中。
3. **终止条件**：当选取的边数达到 $n-1$（其中 $n$ 是图中节点数）时，算法结束。

## 正确性证明

我们使用 **贪心策略的交换性证明法** 来证明 Kruskal 算法的正确性。

### 引理 1：MST 的性质

对于任意连通图 $G = (V, E)$，设 $T^*$ 是其最小生成树。对于 $G$ 中的任意一个割（cut），最小权重的割边（即横跨割的权重最小的边）必定属于某个最小生成树。这被称为 **最小割性质（Cut Property）**。

### 引理 2：贪心策略的正确性

Kruskal 算法始终选择当前图中权重最小的边，且不形成环。这一过程可以等价于不断在图中选择满足 **最小割性质** 的边，因此 Kruskal 算法选出的边一定属于最小生成树。

### 正式证明

#### 假设 Kruskal 选择的树 $T$ 不是最优的，存在一个更优的树 $T^*$

1. 设 Kruskal 选出的最小生成树为 $T$，最优的最小生成树为 $T^*$。
2. Kruskal 选出的边按权重递增顺序排列，且不形成环。
3. 假设 $T$ 与 $T^*$ 不是同一棵树，即存在一条 Kruskal 选择的边 $e = (u, v)$ 在 $T$ 中，但不在 $T^*$ 中。

#### 交换法则

1. 在 $T^*$ 中，加入 Kruskal 选中的边 $e$ 会形成一个环（因为 $T^*$ 是树）。
2. 在这个环中，必然存在一条边 $e'$ 也连接了两个连通分量（因为环内的边总数多于节点数减一）。
3. 若 $e'$ 的权重大于或等于 $e$，我们可以用 $e$ 替换 $e'$，得到新的树 $T'$。
4. $T'$ 仍然是一个生成树，且权重不大于 $T^*$。
5. 重复上述过程，最终可以得到一个与 Kruskal 生成的 $T$ 相同的最小生成树。

因为我们可以通过这样的交换策略，使得 Kruskal 选出的树 $T$ 变成一个最优的最小生成树 $T^*$，这说明 Kruskal 选出的树也是最优的。

## 结论

由于 Kruskal 算法总是按照权重最小的贪心策略选择边，并且这种策略符合最小割性质，最终构造出的生成树一定是最优的。因此，Kruskal 算法的正确性得证。

---
# Kruskal重建树

 **Kruskal 重建树**（Kruskal Reconstruction Tree，简称 KRT）。它并不是一种直接用于求解最小生成树（MST）的算法，而是基于 Kruskal 算法构建的一种数据结构，通常用于解决一些与图的连通性、路径权重或分层相关的复杂问题。它在算法竞赛（如 OI 或 ACM-ICPC）中尤其常见。

## 什么是 Kruskal 重建树？

Kruskal 重建树是通过对 Kruskal 算法的执行过程进行改造，生成的一棵新的树。这棵树并不是原始图的最小生成树，而是通过记录 Kruskal 算法中连通分量的合并过程，构建出的一种层次结构。==**它的核心思想是将每次合并的连通分量抽象为一个新节点**==，从而形成一棵树。

#### 构建过程

1. **初始状态**：
   - 原始图的每个节点作为一个单独的连通分量，对应重建树的叶子节点。
   - 使用并查集（Union-Find）维护连通分量。

2. **按 Kruskal 算法排序**：
   - 将图的所有边按权重从小到大排序（与 Kruskal 算法相同）。

3. **合并与建树**：
   - 对于每条边 $e = (u, v)$，如果 $u$ 和 $v$ 属于不同的连通分量：
     - 创建一个新的节点（称为“虚节点”），其权值通常设为这条边的权值 $w(e)$。
     - 将 $u$ 所在连通分量的根和 $v$ 所在连通分量的根作为这个新节点的子节点。
     - 在并查集中合并 $u$ 和 $v$ 的连通分量。
   - 重复此过程，直到所有节点合并为一个连通分量。

4. **结果**：
   - 最终得到一棵树，树的叶子节点是原始图的节点，非叶子节点是每次合并时创建的虚节点。
   - 这棵树的节点总数为 $2n-1$（$n$ 个叶子节点和 $n-1$ 个虚节点）。

---
## Kruskal 重建树的性质

1. **层次性**：
   - 从根到叶子的路径上，虚节点的权值单调递减（因为 Kruskal 算法按权重递增合并）。
   - 对于任意两个原始节点 $u$ 和 $v$，它们在重建树中的最近公共祖先（LCA）的权值等于 $u$ 和 $v$ 在原图 MST 中路径上的最大边权。

2. **连通性**：
   - 重建树反映了原图中连通分量合并的顺序，适用于分析“最大边权小于等于某个值时的连通性”。

3. **应用性**：
   - 它可以将与边权相关的动态查询问题转化为树上的查询问题（如 LCA 或树上路径问题）。

---
## 示例

假设有一个图，节点为 $\{1, 2, 3\}$，边为：
- $1-2: 1$
- $2-3: 2$
- $1-3: 4$

#### 构建 Kruskal 重建树：

1. 初始时，叶子节点为 $\{1, 2, 3\}$。
2. 按权重排序，处理边 $1-2 (1)$：
   - $1$ 和 $2$ 不在同一连通分量，创建虚节点 $A$（权值 $1$），$A$ 的子节点为 $1$ 和 $2$。
   - 当前树：$A(1) \rightarrow \{1, 2\}$，$3$ 单独。
3. 处理边 $2-3 (2)$：
   - $2$（在 $A$ 下）和 $3$ 不在同一连通分量，创建虚节点 $B$（权值 $2$），$B$ 的子节点为 $A$ 和 $3$。
   - 当前树：$B(2) \rightarrow \{A(1) \rightarrow \{1, 2\}, 3\}$。
4. 边 $1-3 (4)$ 已连接，无需处理。

#### 结果树：

```
    B(2)
   /   \
  A(1)  3
 /   \
1     2
```

- 叶子节点：$1, 2, 3$。
- 虚节点：$A(1), B(2)$。
- $1$ 和 $2$ 的 LCA 是 $A$（权值 $1$），对应 MST 中路径最大边权。
- $1$ 和 $3$ 的 LCA 是 $B$（权值 $2$），对应 MST 中路径最大边权。

---
### 应用场景

1. **最小瓶颈路径**：
   - 查询两点间路径上的最小瓶颈，可以通过在重建树上求 LCA 解决。
1. **动态连通性**：
   - ==判断在边权阈值变化时，哪些点是连通的==。
1. **图的分层问题**：
   - ==将图按边权分层，转化为树上的操作==。

### 时间复杂度

- 构建 Kruskal 重建树的时间复杂度为 $O(m \log m)$（排序边）加上 $O(m \alpha(n))$（并查集操作），其中 $m$ 是边数，$n$ 是节点数，$\alpha(n)$ 是反阿克曼函数。
- 查询（如 LCA）可用树上倍增法优化到 $O(\log n)$。

---
# 数据结构的理解

我也是第一次听说这种神奇的数据结构，***而这种数据结构的思想非常类似于一步一步构建起来的可持久化数据结构（可爱的可持久化线段树:-)~），将每一步的==更新==都用一种简洁的表达存下来，==同时这种”更新的时序“形成了一种重要的维度信息==***。在Kruskal重建树中，这种时序表示了连通块不断增长的最小瓶颈，反应为祖先到叶子节点的路径；而在可持久化线段树中，该维度称之为版本。于是乎，在重建树中就可以查询每个点对应的不同瓶颈下的连通块信息（一直往上查找祖先，直到遇到瓶颈，该祖先的子树就是在该瓶颈下的连通块），而在可持久化线段树中则可以查询任意版本的区间统计信息。

![[Kruskal重建树和主席树同框.png]]

---
# 重建树的应用

## 寻找最小瓶颈

### 题目描述

$\text{sideman}$ 做好了回到 $\text{Gliese}$ 星球的硬件准备，但是 $\text{sideman}$ 的导航系统还没有完全设计好。为了方便起见，我们可以认为宇宙是一张有 $N$ 个顶点和 $M$ 条边的带权无向图，顶点表示各个星系，两个星系之间有边就表示两个星系之间可以直航，而边权则是航行的危险程度。

$\text{sideman}$ 现在想把危险程度降到最小，具体地来说，就是对于若干个询问 $(A, B)$，$\text{sideman}$ 想知道从顶点 $A$ 航行到顶点 $B$ 所经过的最危险的边的危险程度值最小可能是多少。作为 $\text{sideman}$ 的同学，你们要帮助 $\text{sideman}$ 返回家园，兼享受安全美妙的宇宙航行。所以这个任务就交给你了。

### 输入格式

第一行包含两个正整数 $N$ 和 $M$，表示点数和边数。

之后 $M$ 行，每行三个整数 $A$，$B$ 和 $L$，表示顶点 $A$ 和 $B$ 之间有一条边长为 $L$ 的边。顶点从 $1$ 开始标号。

下面一行包含一个正整数 $Q$，表示询问的数目。

之后 $Q$ 行，每行两个整数 $A$ 和 $B$，表示询问 $A$ 和 $B$ 之间最危险的边危险程度的可能最小值。

### 输出格式

对于每个询问， 在单独的一行内输出结果。如果两个顶点之间不可达， 输出 $\text{impossible}$。

这基本上就是最典中典的Kruskal重建树的题目了！

```cpp
// https://www.luogu.com.cn/problem/P2245
// Kruskal重建树模板
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;
const int INF = 0x7f7f7f7f;
const int maxn = 100010;
const int maxm = 300010;
struct edge1 {int u, v, w;} e1[maxm];
struct edge2 {int next, to, val;} e2[maxn << 1];
int n, m, cnt, f[maxn], head[maxn], depth[maxn], fa[maxn][30], w[maxn][30];

bool vis[maxn];
bool cmp(edge1 A, edge1 B) {return A.w < B.w;}
void add_edge(int a, int b, int c) {
    e2[++cnt].to = b;
    e2[cnt].next = head[a];
    e2[cnt].val = c;
    head[a] = cnt;
}
int find(int x) {
    if (x == f[x]) return x;
    else return f[x] = find(f[x]);
}
void kruskal() {
    sort(e1 + 1, e1 + m + 1, cmp);
    for (int i = 1; i <= n; i++) f[i] = i;
    for (int i = 1; i <= m; i++) {
        int c1 = find(e1[i].u);
        int c2 = find(e1[i].v);
        if (c1 == c2) continue;
        f[c2] = c1;
        add_edge(c1, c2, e1[i].w);
        add_edge(c2, c1, e1[i].w);
    }
}
void dfs(int node) {
    vis[node] = true;
    for (int i = head[node]; i; i = e2[i].next) {
        int son = e2[i].to;
        if (vis[son]) continue;
        fa[son][0] = node;
        depth[son] = depth[node] + 1;
        w[son][0] = e2[i].val;
        dfs(son);
    }
    return;
}
int lca(int x, int y)
{
    if (find(x) != find(y))
        return -1; //不连通，输出-1
    int ans = 0;
    if (depth[x] > depth[y])
        swap(x, y); //保证y节点更深
    //将y节点上提到于x节点相同深度
    for (int i = 20; i >= 0; i--)
        if (depth[fa[y][i]] >= depth[x])
        {
            ans = max(ans, w[y][i]); //更新最大载重（最小边权）
            y = fa[y][i];            //修改y位置
        }
    if (x == y)
        return ans; //如果位置已经相等，直接返回答案
    //寻找公共祖先
    for (int i = 20; i >= 0; i--)
        if (fa[x][i] != fa[y][i])
        {
            ans = max(ans, max(w[x][i], w[y][i])); 
            x = fa[x][i];
            y = fa[y][i]; //修改x,y位置
        }
    ans = max(ans, max(w[x][0], w[y][0]));
    //更新此时x,y到公共祖先最大载重，fa[x][0], fa[y][0]即为公共祖先
    return ans;
}
int main() {
    int x, y, z, q;
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        scanf("%d%d%d", &x, &y, &z);
        e1[i].u = x;
        e1[i].v = y;
        e1[i].w = z;
    }
    kruskal();
    for (int i = 1; i <= n; i++) 
        if (!vis[i]) {
            depth[i] = 1;
            dfs(i);
            fa[i][0] = i;
            //w[i][0] = INF;
        }
    //LCA初始化
    for (int i = 1; i <= 20; i++)
        for (int j = 1; j <= n; j++)
        {
            fa[j][i] = fa[fa[j][i - 1]][i - 1];
            w[j][i] = max(w[j][i - 1], w[fa[j][i - 1]][i - 1]);
        }
    cin >> q;
    while (q--) {
        scanf("%d%d", &x, &y);
        int ans = lca(x, y);
        if(ans == -1) printf("impossible\n");
        else printf("%d\n", ans);
    }
    return 0;
}
```

---
## 按照最小瓶颈分层

图里有n个点，m条无向边，每条边给定不同的边权，图里可能有若干个连通的部分  
一共有q条操作，每条操作都是如下的三种类型中的一种  
操作 1 x : 限制变量limit，把limit的值改成x  
操作 2 x : 点x不能走过任何边权小于limit的边，打印此时x所在的连通区域大小  
操作 3 x y : 第x条边的边权修改为y，题目保证修改之后，第x条边的边权排名不变  
$1 \le n、m、q \le 4 * 10^5$
测试链接 : https://www.luogu.com.cn/problem/P9638

**本质上还是利用了Kruskal重建树的最小瓶颈信息**，将limit边权限制看成是瓶颈从而使用倍增的方法找到对应的连通块子树，然后维护查询子树信息即可。（可能需要使用到一些高级数据结构，但是这个思路是不会变化的）

```cpp
// https://www.luogu.com.cn/problem/P9638

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pii = pair<int, int>;
const int mod = 1e9 + 7;
const int inf = 0x3f3f3f3f;
const ll llinf = 0x3f3f3f3f3f3f3f3f;

const int maxn = 4e5 + 7;
const int N = maxn * 2;
int n, m, q, fa[N], dep[N], siz[N], up[N][20], cnt = 0;
int val[N], edge2node[N];
vector<int> g[N];
struct edge {
    int u, v, w, id;
} edges[maxn];

int find(int x) {
    return fa[x] == x ? x : fa[x] = find(fa[x]);
}
void dfs(int u, int father) {
    dep[u] = dep[father] + 1;
    if (u <= n) siz[u] = 1;
    up[u][0] = father;
    for (int i = 1; i < 20; ++i) 
        up[u][i] = up[up[u][i - 1]][i - 1];
    for (int v : g[u]) {
        dfs(v, u);
        siz[u] += siz[v];
    }
}
int solve(int x, int limit) {
    for (int i = 19; i >= 0; --i) {
        if (up[x][i] > 0 && val[up[x][i]] >= limit) {
            x = up[x][i];
        }
    }
    return x;
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cin >> n >> m >> q;
    cnt = n;
    for (int i = 1; i <= n; ++i) fa[i] = i;
    for (int i = 1; i <= m; ++i) {
        edges[i].id = i;
        cin >> edges[i].u >> edges[i].v >> edges[i].w;
    }
    sort(edges + 1, edges + 1 + m, [](edge a, edge b) {
        return a.w > b.w; // 降序
    });
    for (int i = 1; i <= m; ++i) {
        int fu = find(edges[i].u), fv = find(edges[i].v);
        if (fu != fv) {
            int tmp = ++cnt;
            fa[tmp] = tmp;
            val[tmp] = edges[i].w;
            edge2node[edges[i].id] = tmp;
            fa[fu] = fa[fv] = tmp;
            g[tmp].push_back(fu);
            g[tmp].push_back(fv);
        }
    }
    for (int i = 1; i <= cnt; ++i) {
        if (i == find(i)) dfs(i, 0);
    }
    for (int i = 1, op, limit = 0, x, y; i <= q; ++i) {
        cin >> op;
        if (op == 1) cin >> limit;
        else if (op == 2) {
            cin >> x;
            int res = solve(x, limit);
            cout << siz[res] << '\n';
        }
        else {
            cin >> x >> y;
            int node = edge2node[x];
            if (node) val[node] = y;
        }
    }
}
```








