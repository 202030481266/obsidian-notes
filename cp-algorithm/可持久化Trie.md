# 定义

可持久化Trie是一种常见的可持久化数据结构，其中的原理和可持久化线段树非常类似，还是针对于变动的地方做改变。其中最为常见的是可持久化01Trie，常常会使用这个数据结构解决一些数字异或的难题，特别是涉及到一些区间查询（包括链式和树上路径）的情况。
# 讲解

相对来说，我觉得可持久化Trie的建模思路比较简单，一般来说都是涉及到了区间查询，而且01Trie主要也是围绕高效解决一些最大异或的问题。下面就是一道非常经典的可持久化01Trie运用的题目。
## 题目描述

给定一个非负整数序列 $\{a\}$，初始长度为 $N$。  

有 $M$ 个操作，有以下两种操作类型：  

1. `A x`：添加操作，表示在序列末尾添加一个数 $x$，序列的长度 $N$ 加 $1$。  
2. `Q l r x`：询问操作，你需要找到一个位置 $p$，满足 $l \le p \le r$，使得：$a[p] \oplus a[p+1] \oplus ... \oplus a[N] \oplus x$ 最大，输出最大值。

## 输入格式

第一行包含两个整数 $N, M$，含义如问题描述所示。     
第二行包含 $N$ 个非负整数，表示初始的序列 $A$。   
接下来 $M$ 行，每行描述一个操作，格式如题面所述。

## 输出格式

假设询问操作有 $T$ 个，则输出应该有 $T$ 行，每行一个整数表示询问的答案。

## 样例

### 样例输入

```
5 5
2 6 4 3 6
A 1 
Q 3 5 4 
A 4
Q 5 7 0 
Q 3 6 6
```

### 样例输出

```
4
5
6
```

## 提示

- 对于所有测试点，$1\le N,M \le 3\times 10 ^ 5$，$0\leq a_i\leq 10 ^ 7$。

这道题目解决方法也是不难的，主要是使用前缀和的思路构建一棵可持久化01Trie，0版本是空的异或前缀和，也就是0，1版本对应的是$a[1]$，2版本对应的是$a[1]\oplus a[2]$，3版本对应的是$a[1] \oplus a[2] \oplus a[3]$，后面的以此类推。那么对于查询则可以转换成查询在版本$[l-1,r-1]$中，找到一个最大的前缀和，使其和$x\oplus xorsum$最大，其中$xorsum$表示目前所有的数字的异或和。

```cpp
// P4735 最大异或和
// https://www.luogu.com.cn/problem/P4735

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pii = pair<int, int>;
const int mod = 1e9 + 7;
const int inf = 0x3f3f3f3f;
const ll llinf = 0x3f3f3f3f3f3f3f3f;

const int maxn = 3e5 + 10;
const int N = maxn * 50;
const int BIT = 30;

int root[maxn * 2], tot, tree[N][2];
int n, m, cnt[N], xor_sum;

int insert(int x, int i) {
    int rt = ++tot;
    tree[rt][0] = tree[i][0];
    tree[rt][1] = tree[i][1];
    cnt[rt] = cnt[i] + 1;
    for (int j = BIT, b, pre = rt, cur; j >= 0; --j) {
        b = (x >> j) & 1;
        i = tree[i][b];
        cur = ++tot;
        tree[cur][0] = tree[i][0];
        tree[cur][1] = tree[i][1];
        cnt[cur] = cnt[i] + 1; 
        tree[pre][b] = cur;
        pre = cur;
    } 
    return rt;
}
int query(int x, int i, int j) {
    int res = 0;
    for (int k = BIT, b; k >= 0; --k) {
        b = (x >> k) & 1;
        if (cnt[tree[j][b ^ 1]] - cnt[tree[i][b ^ 1]] > 0) {
            res += 1<<k;
            j = tree[j][b ^ 1];
            i = tree[i][b ^ 1];
        }
        else {
            j = tree[j][b];
            i = tree[i][b];
        }
    }
    return res;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cin >> n >> m;
    root[0] = insert(0, 0); // 前缀异或和为 0
    for (int i = 1, tmp; i <= n; ++i) {
        cin >> tmp;
        xor_sum ^= tmp;
        root[i] = insert(xor_sum, root[i-1]);
    }
    // [l, r]
    // [l-1,r-1]
    char op;
    for (int i = 1, l, r, num; i <= m; ++i) {
        cin >> op;
        if (op == 'A') {
            cin >> num;
            ++n;
            xor_sum ^= num;
            root[n] = insert(xor_sum, root[n-1]);
        }
        else {
            cin >> l >> r >> num;
            if (l == 1) {
                cout << query(xor_sum ^ num, 0, root[r-1]) << '\n';
            }
            else {
                cout << query(xor_sum ^ num, root[l-2], root[r-1]) << '\n';
            }
        }
    }
}
```

下面是一道经典的树上路径+可持久化Trie的模板题目的代码（具体思路很简单，就是先找到LCA然后就是树上路径的差分，和可持久化线段树非常类似）。

```cpp
// P6088 [JSOI2015] 字符串树
// https://www.luogu.com.cn/problem/P6088

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pii = pair<int, int>;
const int mod = 1e9 + 7;
const int inf = 0x3f3f3f3f;
const ll llinf = 0x3f3f3f3f3f3f3f3f;

const int maxn = 1e5 + 7;
const int N = maxn * 20;
const int B = 26;
int root[maxn], tot, tree[N][B], cnt[N];
int n, m, dep[maxn], fa[maxn][20];
vector<pair<int,string>> g[maxn];

int insert(string& s, int x) {
    int rt = ++tot;
    for (int i = 0; i < B; ++i) {
        tree[rt][i] = tree[x][i];
    }
    cnt[rt] = cnt[x] + 1;
    for (int i = 0, pre = rt, cur, b; i < s.size(); ++i) {
        b = s[i] - 'a';
        x = tree[x][b];
        cur = ++tot;
        for (int j = 0; j < B; ++j) {
            tree[cur][j] = tree[x][j];
        }
        cnt[cur] = cnt[x] + 1;
        tree[pre][b] = cur;
        pre = cur;
    }
    return rt;
}

int query(string& s, int x) {
    for (int i = 0, b; i < s.size(); ++i) {
        b = s[i] - 'a';
        x = tree[x][b];
        if (!x) return 0;
    }
    return cnt[x];
}

void dfs1(int u, int father, string& w) {
    dep[u] = dep[father] + 1;
    fa[u][0] = father;
    root[u] = insert(w, root[father]);
    for (int i = 1; i < 20; ++i) fa[u][i] = fa[fa[u][i - 1]][i - 1];
    for (auto [v, s] : g[u]) {
        if (v != father) {
            dfs1(v, u, s);
        }
    }
}

int lca(int x, int y) {
    if (dep[x] < dep[y]) swap(x, y);
    for (int i = 19; i >= 0; --i) {
        if (dep[fa[x][i]] >= dep[y]) x = fa[x][i];
    }
    if (x == y) return x;
    for (int i = 19; i >= 0; --i) {
        if (fa[x][i] != fa[y][i]) x = fa[x][i], y = fa[y][i];
    }
    return fa[x][0];
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cin >> n;
    string s, prefix, empty;
    for (int i = 1, u, v; i < n; ++i) {
        cin >> u >> v >> s;
        g[u].push_back({v, s});
        g[v].push_back({u, s});
    }
    dfs1(1, 0, empty);
    cin >> m;
    for (int i = 1, u, v; i <= m; ++i) {
        cin >> u >> v >> prefix;
        int l = lca(u, v);
        cout << query(prefix, root[u]) + query(prefix, root[v]) - query(prefix, root[l]) * 2 << '\n';
    }
}
```

