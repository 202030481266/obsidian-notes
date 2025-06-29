# C++ PBDS和Rope数据结构详细使用指南

## 1. PBDS (Policy-Based Data Structures)

### 1.1 基本配置

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
```

### 1.2 Tree容器

#### 基本定义

```cpp
// 有序集合（类似set但支持统计操作）
typedef tree<int, null_type, less<int>, rb_tree_tag, 
             tree_order_statistics_node_update> ordered_set;

// 有序映射（类似map但支持统计操作）
typedef tree<int, int, less<int>, rb_tree_tag, 
             tree_order_statistics_node_update> ordered_map;

// 支持重复元素的有序集合
typedef tree<int, null_type, less_equal<int>, rb_tree_tag, 
             tree_order_statistics_node_update> ordered_multiset;
```

#### 核心操作

```cpp
ordered_set s;

// 插入元素
s.insert(10);
s.insert(20);
s.insert(5);
s.insert(15);

// 删除元素
s.erase(10);

// 查找第k小的元素（0-indexed）
cout << *s.find_by_order(0); // 最小元素
cout << *s.find_by_order(1); // 第二小元素

// 查找小于x的元素个数
cout << s.order_of_key(15); // 小于15的元素个数

// 查找小于等于x的元素个数（对于multiset）
cout << s.order_of_key(15 + 1) - 1;

// 普通操作
cout << s.size();
cout << (s.find(15) != s.end()); // 查找元素是否存在
```

#### 实际应用示例

```cpp
// 示例1：动态第k小问题
ordered_set s;
s.insert(3);
s.insert(1);
s.insert(4);
s.insert(2);
cout << *s.find_by_order(1); // 输出2（第2小的元素）

// 示例2：逆序对计算
vector<int> arr = {3, 1, 4, 1, 5, 9, 2, 6};
ordered_set s;
long long inversions = 0;
for (int i = 0; i < arr.size(); i++) {
    inversions += s.size() - s.order_of_key(arr[i]);
    s.insert(arr[i]);
}
cout << "逆序对数量: " << inversions << endl;

// 示例3：区间排名查询
ordered_set s;
// 查询区间[L, R]中有多少个数
// s.order_of_key(R + 1) - s.order_of_key(L)
```

### 1.3 Priority Queue

支持合并操作的优先队列（实际上左偏树更加好写而且速度比这个更加快，还支持可持久化操作，所以这个只能是在有限场景下使用）

#### 基本定义和操作

```cpp
#include <ext/pb_ds/priority_queue.hpp>

// 可合并的大根堆
typedef __gnu_pbds::priority_queue<int> pq_max;

// 可合并的小根堆
typedef __gnu_pbds::priority_queue<int, greater<int>> pq_min;

pq_max pq1, pq2;

// 插入元素，返回迭代器
auto it1 = pq1.push(10);
auto it2 = pq1.push(20);
auto it3 = pq1.push(5);

// 访问最大元素
cout << pq1.top(); // 20

// 删除最大元素
pq1.pop();

// 修改元素值
pq1.modify(it1, 25); // 将10修改为25

// 删除指定元素
pq1.erase(it3);

// 合并两个优先队列
pq2.push(30);
pq2.push(15);
pq1.join(pq2); // pq2会被清空，所有元素合并到pq1

// 分割操作
pq_max pq3;
pq1.split(15, pq3); // 将pq1中小于等于15的元素移动到pq3
```

#### 应用场景

```cpp
// 可撤销的优先队列操作
vector<pq_max::point_iterator> handles;
pq_max pq;

// 插入并保存句柄
handles.push_back(pq.push(10));
handles.push_back(pq.push(20));
handles.push_back(pq.push(5));

// 后续可以直接删除或修改
pq.erase(handles[1]); // 删除值为20的元素
pq.modify(handles[0], 30); // 将10修改为30
```

### 1.4 Hash Table

#### 基本使用

```cpp
#include <ext/pb_ds/assoc_container.hpp>

// 哈希表
typedef cc_hash_table<int, int> hash_map;
typedef gp_hash_table<int, int> gp_hash_map; // 通常更快

gp_hash_map mp;
mp[1] = 10;
mp[2] = 20;

cout << mp[1]; // 10
cout << (mp.find(3) != mp.end()); // false
```

### 1.5 Trie

```cpp
#include <ext/pb_ds/trie_policy.hpp>

typedef trie<string, null_type, trie_string_access_traits<>, 
             pat_trie_tag, trie_prefix_search_node_update> pref_trie;

pref_trie t;
t.insert("hello");
t.insert("world");
t.insert("help");

// 前缀查询
auto range = t.prefix_range("hel");
for (auto it = range.first; it != range.second; ++it) {
    cout << *it << " "; // 输出 "hello help"
}
```

## 2. Rope数据结构

Rope的本质上是将字符串拆解为一个一个的块的叶子节点，然后使用BST来管理，做到查询和修改都有良好的时间复杂度。链表结构添加和删除块但是查找很慢，数组结构查找很快但是添加和删除都很慢。

### 2.1 基本操作

```cpp
#include <ext/rope>
using namespace __gnu_cxx;

// 创建rope
rope<char> r1("Hello");
rope<char> r2(" World");
rope<char> r3 = r1 + r2; // "Hello World"

// 基本操作
cout << r3.size(); // 11
cout << r3[0]; // 'H'
cout << r3.substr(0, 5); // "Hello"

// 插入操作 O(log n)
r3.insert(5, " Beautiful"); // "Hello Beautiful World"

// 删除操作 O(log n)
r3.erase(5, 10); // 删除位置5开始的10个字符

// 替换操作
r3.replace(0, 5, "Hi"); // 将"Hello"替换为"Hi"
```

### 2.2 高级操作

```cpp
rope<char> text("This is a long text for demonstration");

// 分割rope
rope<char> left = text.substr(0, 10);
rope<char> right = text.substr(10);

// 查找操作
size_t pos = text.find("long"); // 查找子串位置
if (pos != rope<char>::npos) {
    cout << "Found at position: " << pos << endl;
}

// 迭代器操作
for (auto it = text.begin(); it != text.end(); ++it) {
    cout << *it;
}

// 反向迭代器
for (auto it = text.rbegin(); it != text.rend(); ++it) {
    cout << *it;
}
```

### 2.3 性能优势场景

```cpp
// 场景1：频繁的字符串插入删除
rope<char> document;
// 在文档中间频繁插入文本 - O(log n)而不是O(n)
document.insert(1000, "new paragraph");
document.erase(500, 100);

// 场景2：大文件的文本编辑器
class TextEditor {
private:
    rope<char> content;
    
public:
    void insert_text(size_t pos, const string& text) {
        content.insert(pos, text.c_str());
    }
    
    void delete_text(size_t pos, size_t len) {
        content.erase(pos, len);
    }
    
    string get_line(size_t start, size_t end) {
        return content.substr(start, end - start);
    }
};

// 场景3：字符串的高效合并
vector<rope<char>> parts;
rope<char> result;
for (const auto& part : parts) {
    result += part; // 高效合并
}
```

## 3. 实际应用案例

### 3.1 使用ordered_set解决经典问题

```cpp
// 问题：动态维护数列，支持插入、删除、查询第k小、查询排名
class DynamicRankingSet {
private:
    ordered_set s;
    
public:
    void insert(int x) { s.insert(x); }
    void erase(int x) { s.erase(x); }
    int kth(int k) { return *s.find_by_order(k - 1); }
    int rank(int x) { return s.order_of_key(x) + 1; }
    int size() { return s.size(); }
};
```

### 3.2 使用rope实现高效文本操作

```cpp
// 问题：实现一个支持大量插入删除的文本缓冲区
class TextBuffer {
private:
    rope<char> buffer;
    
public:
    void insert(size_t pos, const string& text) {
        buffer.insert(pos, text.c_str());
    }
    
    void erase(size_t pos, size_t len) {
        buffer.erase(pos, len);
    }
    
    string substr(size_t pos, size_t len) {
        return buffer.substr(pos, len);
    }
    
    char at(size_t pos) {
        return buffer[pos];
    }
    
    size_t size() {
        return buffer.size();
    }
};
```

---

# PBDS和Rope数据结构操作复杂度详解

## 1. PBDS Tree容器复杂度分析

### 1.1 ordered_set/ordered_map 基本操作

|操作|时间复杂度|空间复杂度|说明|
|---|---|---|---|
|`insert(x)`|O(log n)|O(1)|插入元素|
|`erase(x)`|O(log n)|O(1)|按值删除|
|`erase(iterator)`|O(log n)|O(1)|按迭代器删除|
|`find(x)`|O(log n)|O(1)|查找元素|
|`count(x)`|O(log n)|O(1)|统计元素个数|
|`size()`|O(1)|O(1)|获取大小|
|`empty()`|O(1)|O(1)|判断是否为空|

### 1.2 ordered_set/ordered_map 特有统计操作

|操作|时间复杂度|空间复杂度|说明|
|---|---|---|---|
|`find_by_order(k)`|O(log n)|O(1)|查找第k小元素(0-indexed)|
|`order_of_key(x)`|O(log n)|O(1)|查找小于x的元素个数|

```cpp
// 复杂度示例代码
ordered_set s;
// 插入n个元素：总复杂度 O(n log n)
for (int i = 0; i < n; i++) {
    s.insert(arr[i]); // 每次 O(log n)
}

// 查询操作都是 O(log n)
int kth = *s.find_by_order(k-1);     // O(log n)
int rank = s.order_of_key(x);        // O(log n)
bool exists = s.find(x) != s.end();  // O(log n)
```

### 1.3 不同底层实现的复杂度对比

|树类型|插入|删除|查找|统计操作|空间复杂度|
|---|---|---|---|---|---|
|`rb_tree_tag` (红黑树)|O(log n)|O(log n)|O(log n)|O(log n)|O(n)|
|`splay_tree_tag` (伸展树)|O(log n)*|O(log n)*|O(log n)*|O(log n)*|O(n)|
|`ov_tree_tag` (有序向量树)|O(n)|O(n)|O(log n)|O(1)|O(n)|

*注：伸展树是摊还复杂度

## 2. PBDS Priority Queue复杂度分析

### 2.1 基本操作复杂度

|操作|Pairing Heap|Binary Heap|Binomial Heap|RC Binomial Heap|Thin Heap|
|---|---|---|---|---|---|
|`push(x)`|O(1)|O(log n)|O(log n)|O(log n)|O(1)|
|`top()`|O(1)|O(1)|O(1)|O(1)|O(1)|
|`pop()`|O(log n)*|O(log n)|O(log n)|O(log n)|O(log n)|
|`modify(it, x)`|O(log n)*|O(log n)|O(log n)|O(log n)|O(log n)|
|`erase(it)`|O(log n)*|O(log n)|O(log n)|O(log n)|O(log n)|
|`size()`|O(1)|O(1)|O(1)|O(1)|O(1)|
|`empty()`|O(1)|O(1)|O(1)|O(1)|O(1)|

*摊还复杂度

### 2.2 特有操作复杂度

|操作|Pairing Heap|Binary Heap|Binomial Heap|RC Binomial Heap|Thin Heap|
|---|---|---|---|---|---|
|`join(other)`|O(1)|O(n)|O(log n)|O(log n)|O(log n)|
|`split(pred, other)`|O(n)|O(n)|O(n)|O(n)|O(n)|

```cpp
// 不同堆类型的定义
typedef __gnu_pbds::priority_queue<int, greater<int>, pairing_heap_tag> pairing_pq;
typedef __gnu_pbds::priority_queue<int, greater<int>, binary_heap_tag> binary_pq;
typedef __gnu_pbds::priority_queue<int, greater<int>, binomial_heap_tag> binomial_pq;
typedef __gnu_pbds::priority_queue<int, greater<int>, rc_binomial_heap_tag> rc_binomial_pq;
typedef __gnu_pbds::priority_queue<int, greater<int>, thin_heap_tag> thin_pq;

// 合并操作复杂度示例
pairing_pq pq1, pq2;
// ... 插入元素到pq1和pq2
pq1.join(pq2); // Pairing heap: O(1), Binomial heap: O(log n)
```

## 3. PBDS Hash Table复杂度分析

### 3.1 基本操作复杂度

|操作|cc_hash_table|gp_hash_table|说明|
|---|---|---|---|
|`insert(x)`|O(1)*|O(1)*|插入键值对|
|`erase(x)`|O(1)*|O(1)*|删除元素|
|`find(x)`|O(1)*|O(1)*|查找元素|
|`operator[]`|O(1)*|O(1)*|访问/插入元素|
|`size()`|O(1)|O(1)|获取大小|
|`empty()`|O(1)|O(1)|判断是否为空|

*期望复杂度，最坏情况可能退化到O(n)

### 3.2 内存和性能对比

|特性|cc_hash_table|gp_hash_table|std::unordered_map|
|---|---|---|---|
|冲突解决|链地址法|开放地址法|链地址法|
|内存使用|较高|较低|中等|
|缓存友好性|较差|较好|中等|
|插入性能|中等|较好|中等|
|查找性能|中等|较好|中等|

```cpp
// 性能测试示例
const int N = 1000000;

// gp_hash_table 通常在大多数情况下性能最好
gp_hash_table<int, int> gp_map;
auto start = chrono::high_resolution_clock::now();
for (int i = 0; i < N; i++) {
    gp_map[i] = i * 2;  // 期望 O(1) 每次操作
}
auto end = chrono::high_resolution_clock::now();
// 总时间复杂度：O(n)
```

## 4. PBDS Trie复杂度分析

### 4.1 基本操作复杂度

| 操作                     | 时间复杂度  | 空间复杂度  | 说明   |
| ---------------------- | ------ | ------ | ---- |
| `insert(str)`          | O(n)   | str    | 插入   |
| `erase(str)`           | O(n)   | str    | 删除   |
| `find(str)`            | O(n)   | str    | 查找   |
| `prefix_range(prefix)` | O(n+k) | prefix | 前缀查找 |

```cpp
trie<string, null_type, trie_string_access_traits<>, 
     pat_trie_tag, trie_prefix_search_node_update> t;

// 插入操作复杂度取决于字符串长度
t.insert("hello");     // O(5)
t.insert("world");     // O(5)
t.insert("help");      // O(4)

// 前缀查询复杂度
auto range = t.prefix_range("hel");  // O(3 + 匹配结果数量)
```

## 5. Rope数据结构复杂度分析

### 5.1 基本操作复杂度

| 操作                       | 时间复杂度          | 空间复杂度  | 说明   |
| ------------------------ | -------------- | ------ | ---- |
| `rope(str)`              | O(             | str    | )    |
| `size()`                 | O(1)           | O(1)   | 获取长度 |
| `operator[](i)`          | O(log n)       | O(1)   | 随机访问 |
| `substr(pos, len)`       | O(log n + len) | O(len) | 子串提取 |
| `insert(pos, str)`       | O(log n + len) | str    | 插入子串 |
| `erase(pos, len)`        | O(log n + len) | O(1)   | 删除子串 |
| `replace(pos, len, str)` | O(log n + len) | str    | 替换子串 |
| `operator+(other)`       | O(log(n + m))  | O(1)   | 连接操作 |
| `find(str)`              | O(n)           | str    | 查找子串 |

### 5.2 与string操作的复杂度对比

|操作|std::string|rope|优势|
|---|---|---|---|
|构造|O(n)|O(n)|相同|
|访问|O(1)|O(log n)|string更好|
|插入(中间)|O(n)|O(log n)|**rope更好**|
|删除(中间)|O(n)|O(log n)|**rope更好**|
|连接|O(n + m)|O(log(n + m))|**rope更好**|
|子串|O(len)|O(log n + len)|取决于n和len|

```cpp
// 复杂度对比示例
string s = "Hello World";
rope<char> r("Hello World");

// 中间插入操作
s.insert(5, " Beautiful");  // O(n) - 需要移动后续所有字符
r.insert(5, " Beautiful");  // O(log n) - 只需要调整树结构

// 频繁插入场景下的复杂度
// string: n次插入 = O(n²)
// rope: n次插入 = O(n log n)
```

### 5.3 Rope内部结构和平衡性

|特性|复杂度影响|说明|
|---|---|---|
|树高度|O(log n)|影响所有树操作的复杂度|
|重平衡|摊还O(1)|自动维护平衡|
|叶子节点大小|常数因子|影响实际性能|
|内部节点开销|空间常数|比string占用更多内存|

## 6. 实际性能考虑

### 6.1 常数因子影响

```cpp
// 虽然复杂度相同，但常数因子不同
ordered_set s1;           // 红黑树，常数因子较小
set<int> s2;             // 通常也是红黑树，STL实现

// PBDS的优势主要在特有功能，而不一定是更快的基本操作
for (int i = 0; i < 100000; i++) {
    s1.insert(i);  // 可能略慢于STL
    // 但是可以使用 s1.find_by_order(k) 和 s1.order_of_key(x)
}
```

### 6.2 内存使用分析

|数据结构|内存开销|说明|
|---|---|---|
|ordered_set|每个节点额外存储子树大小|比std::set多4-8字节每节点|
|PBDS priority_queue|取决于堆类型|Pairing heap开销较大|
|rope|内部节点存储长度信息|比string多约2倍内存开销|
|PBDS hash_table|取决于负载因子|gp_hash_table内存效率更高|

### 6.3 使用建议

1. **ordered_set**: 当需要统计操作时使用，基本操作可能略慢于std::set
2. **PBDS priority_queue**: 当需要合并操作时使用，否则std::priority_queue就足够
3. **rope**: 只在频繁中间插入删除的大字符串场景使用
4. **PBDS hash_table**: 在性能敏感且不需要标准兼容性时考虑使用
