# **使用Go标准库 `container/heap`**：

- 需要实现`heap.Interface`接口的方法：`Len()`、`Less()`、`Swap()`、`Push()`和`Pop()`
- 然后使用`heap.Init()`、`heap.Push()`和`heap.Pop()`等函数操作堆

# **手动实现堆结构**：

- 基于数组实现了完整的最小堆
- 包含了堆的核心操作：
    - 向上调整（siftUp）- 用于插入元素后维护堆性质
    - 向下调整（siftDown）- 用于删除最小元素后维护堆性质
    - 插入（Insert）
    - 获取最小值（Peek）
    - 删除并返回最小值（ExtractMin）

大体上用的最多就只有上面两种方法，下面是一个具体的例子：

```go
package main

import (
	"container/heap"
	"fmt"
)

// 一、使用标准库 container/heap 实现的最小堆
// IntHeap 是一个由整数组成的最小堆
type IntHeap []int

// 实现 heap.Interface 接口的必要方法
func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] } // 小于号表示最小堆，大于号表示最大堆
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

// Push 和 Pop 方法用于实现 heap.Interface
func (h *IntHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// 二、从头手动实现的最小堆
type MinHeap struct {
	array []int
}

// 初始化一个新的最小堆
func NewMinHeap() *MinHeap {
	return &MinHeap{
		array: []int{},
	}
}

// 获取父节点索引
func (h *MinHeap) parent(i int) int {
	return (i - 1) / 2
}

// 获取左子节点索引
func (h *MinHeap) leftChild(i int) int {
	return 2*i + 1
}

// 获取右子节点索引
func (h *MinHeap) rightChild(i int) int {
	return 2*i + 2
}

// 交换两个元素
func (h *MinHeap) swap(i, j int) {
	h.array[i], h.array[j] = h.array[j], h.array[i]
}

// 向上调整
func (h *MinHeap) siftUp(i int) {
	for i > 0 && h.array[h.parent(i)] > h.array[i] {
		h.swap(h.parent(i), i)
		i = h.parent(i)
	}
}

// 向下调整
func (h *MinHeap) siftDown(i int) {
	minIndex := i
	left := h.leftChild(i)
	if left < len(h.array) && h.array[left] < h.array[minIndex] {
		minIndex = left
	}
	
	right := h.rightChild(i)
	if right < len(h.array) && h.array[right] < h.array[minIndex] {
		minIndex = right
	}
	
	if i != minIndex {
		h.swap(i, minIndex)
		h.siftDown(minIndex)
	}
}

// 插入元素
func (h *MinHeap) Insert(val int) {
	h.array = append(h.array, val)
	h.siftUp(len(h.array) - 1)
}

// 获取最小元素（不删除）
func (h *MinHeap) Peek() (int, error) {
	if len(h.array) == 0 {
		return 0, fmt.Errorf("堆为空")
	}
	return h.array[0], nil
}

// 删除并返回最小元素
func (h *MinHeap) ExtractMin() (int, error) {
	if len(h.array) == 0 {
		return 0, fmt.Errorf("堆为空")
	}
	
	min := h.array[0]
	h.array[0] = h.array[len(h.array)-1]
	h.array = h.array[:len(h.array)-1]
	
	if len(h.array) > 0 {
		h.siftDown(0)
	}
	
	return min, nil
}

// 堆的大小
func (h *MinHeap) Size() int {
	return len(h.array)
}

// 使用示例
func main() {
	// 1. 使用标准库实现的堆
	fmt.Println("使用 container/heap 包实现的堆:")
	h := &IntHeap{3, 1, 4, 1, 5, 9}
	heap.Init(h) // 这个操作是必要的，哪怕初始化为空
	fmt.Printf("最小值: %d\n", (*h)[0])
	heap.Push(h, 2)
	fmt.Printf("插入2后，最小值: %d\n", (*h)[0])
	fmt.Printf("弹出: %d\n", heap.Pop(h))
	fmt.Printf("弹出后，最小值: %d\n", (*h)[0])
	
	// 2. 使用自定义实现的堆
	fmt.Println("\n使用自定义实现的堆:")
	minHeap := NewMinHeap()
	minHeap.Insert(3)
	minHeap.Insert(1)
	minHeap.Insert(4)
	minHeap.Insert(1)
	minHeap.Insert(5)
	minHeap.Insert(9)
	
	min, _ := minHeap.Peek()
	fmt.Printf("最小值: %d\n", min)
	
	minHeap.Insert(2)
	min, _ = minHeap.Peek()
	fmt.Printf("插入2后，最小值: %d\n", min)
	
	extracted, _ := minHeap.ExtractMin()
	fmt.Printf("弹出: %d\n", extracted)
	
	min, _ = minHeap.Peek()
	fmt.Printf("弹出后，最小值: %d\n", min)
}
```

---
# 练习

[1942. 最小未被占据椅子的编号](https://leetcode.cn/problems/the-number-of-the-smallest-unoccupied-chair/)

```go
import (
	"container/heap"
	"sort"
)

type IntHeap []int
type EventHeap [][2]int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] } // 最小堆
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *IntHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}
func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h EventHeap) Len() int           { return len(h) }
func (h EventHeap) Less(i, j int) bool { return h[i][0] < h[j][0] }
func (h EventHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *EventHeap) Push(x interface{}) {
	*h = append(*h, x.([2]int))
}
func (h *EventHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func smallestChair(times [][]int, targetFriend int) int {
	n := len(times)
	ids := make([]int, n)
	for i := 0; i < n; i++ {
		ids[i] = i
	}
	// 按到达时间排序
	sort.Slice(ids, func(i, j int) bool {
		return times[ids[i]][0] < times[ids[j]][0]
	})

	// 找到 targetFriend 的索引
	p := 0
	for i := 0; i < n; i++ {
		if ids[i] == targetFriend {
			p = i
			break
		}
	}

	// 初始化堆
	h := &IntHeap{}    // 可用椅子堆
	eh := &EventHeap{} // 事件堆（离开时间和椅子编号）
	heap.Init(h)
	heap.Init(eh)

	// 将椅子编号 0 到 p 添加到可用椅子堆中
	for i := 0; i <= p; i++ {
		heap.Push(h, i)
	}

	// 处理到 targetFriend 之前的每个人
	for i := 0; i < p; i++ {
		// 释放所有在当前到达时间之前离开的椅子
		for eh.Len() > 0 && (*eh)[0][0] <= times[ids[i]][0] {
			ev := heap.Pop(eh).([2]int) // 类型断言为 [2]int
			heap.Push(h, ev[1])         // 将椅子放回可用堆
		}
		// 分配一个椅子
		number := heap.Pop(h).(int)                     // 类型断言为 int
		heap.Push(eh, [2]int{times[ids[i]][1], number}) // 记录离开时间和椅子编号
	}

	// 处理 targetFriend 的情况
	for eh.Len() > 0 && (*eh)[0][0] <= times[ids[p]][0] {
		ev := heap.Pop(eh).([2]int)
		heap.Push(h, ev[1])
	}
	return heap.Pop(h).(int) // 返回 targetFriend 分配的椅子
}
```
