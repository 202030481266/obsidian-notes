# Go语言字符串处理

Go语言提供了丰富的字符串处理功能，下面介绍几种主要的字符串操作方法：

## 字符串基础

1. **字符串声明**：

   ```go
   var s1 string = "Hello, 世界"
   s2 := "Go语言"
   ```

2. **字符串长度**：

   - `len()` 返回字节数
   - `utf8.RuneCountInString()` 返回字符数
   
   ```go
   s := "Hello, 世界"
   fmt.Println(len(s))         // 输出：13 (字节数)
   fmt.Println(utf8.RuneCountInString(s)) // 输出：9 (字符数)
   ```

## 常用字符串操作

1. **拼接字符串**：

   ```go
   s1 := "Hello"
   s2 := "World"
   result := s1 + " " + s2  // 使用+运算符
   result2 := fmt.Sprintf("%s %s", s1, s2)  // 使用Sprintf
   result3 := strings.Join([]string{s1, s2}, " ")  // 使用Join
   ```

2. **分割字符串**：

   ```go
   s := "a,b,c"
   parts := strings.Split(s, ",")  // ["a", "b", "c"]
   ```

3. **子串查找**：

   ```go
   s := "hello world"
   index := strings.Index(s, "world")  // 返回6
   contains := strings.Contains(s, "world")  // 返回true
   ```

4. **替换**：

   ```go
   s := "hello hello world"
   newS := strings.Replace(s, "hello", "hi", -1)  // "hi hi world"
   ```

5. **大小写转换**：

   ```go
   s := "Hello World"
   lower := strings.ToLower(s)  // "hello world"
   upper := strings.ToUpper(s)  // "HELLO WORLD"
   ```

6. **修剪空白**：

   ```go
   s := "  hello  "
   trimmed := strings.TrimSpace(s)  // "hello"
   ```

## 字符串与字节切片转换

```go
s := "hello"
b := []byte(s)  // 字符串转字节切片
s2 := string(b) // 字节切片转字符串
```

## 字符串遍历

1. **按字节遍历**：

   ```go
   s := "世界"
   for i := 0; i < len(s); i++ {
       fmt.Printf("%x ", s[i])
   }
   ```

2. **按字符(rune)遍历**：

   ```go
   s := "世界"
   for _, r := range s {
       fmt.Printf("%c ", r)
   }
   ```

## 字符串格式化

```go
name := "Alice"
age := 25
s := fmt.Sprintf("%s is %d years old", name, age)
```

## 字符串比较

```go
s1 := "hello"
s2 := "world"
if s1 == s2 {
    fmt.Println("equal")
} else {
    fmt.Println("not equal")
}
```

Go语言的字符串是不可变的，任何修改操作都会创建新的字符串。对于大量字符串操作，可以使用`strings.Builder`来提高性能。

---
# 使用 strings.Builder 高效构建字符串

`strings.Builder` 是 Go 语言中用于高效构建字符串的类型，特别适合需要频繁拼接字符串的场景。相比直接使用 `+` 或 `fmt.Sprintf`，它能显著减少内存分配和拷贝次数。

## 基本用法

### 1. 创建 Builder

```go
var builder strings.Builder
// 或者
builder := &strings.Builder{}
```

### 2. 添加内容

主要方法：

- `WriteString(s string)` - 写入字符串
- `Write(b []byte)` - 写入字节切片
- `WriteByte(c byte)` - 写入单个字节
- `WriteRune(r rune)` - 写入 Unicode 字符

```go
builder.WriteString("Hello")
builder.WriteString(" ")
builder.WriteString("World!")
```

### 3. 获取最终字符串

```go
result := builder.String()
```

## 完整示例

```go
package main

import (
	"fmt"
	"strings"
)

func main() {
	var builder strings.Builder
	
	// 添加内容
	builder.WriteString("姓名: ")
	builder.WriteString("张三")
	builder.WriteString("\n年龄: ")
	builder.WriteString("30")
	builder.WriteString("\n职业: ")
	builder.WriteString("程序员")
	
	// 获取构建的字符串
	profile := builder.String()
	fmt.Println(profile)
	
	// 输出:
	// 姓名: 张三
	// 年龄: 30
	// 职业: 程序员
}
```

## 高级用法

### 1. 重置 Builder

可以重用 Builder 对象，减少内存分配：

```go
builder.Reset()  // 清空内容
builder.WriteString("新的内容")
```

### 2. 性能优化

可以预先分配缓冲区大小：

```go
builder := strings.Builder{}
builder.Grow(100)  // 预分配100字节的缓冲区
```

### 3. 与其他接口兼容

`strings.Builder` 实现了多个 io 接口：

```go
// 实现了 io.Writer
fmt.Fprintf(&builder, "格式化: %d", 123)

// 实现了 io.ByteWriter
builder.WriteByte('A')

// 实现了 io.StringWriter
builder.WriteString("直接写入")
```

## 性能对比

```go
package main

import (
	"fmt"
	"strings"
	"time"
)

func concatWithPlus(n int) string {
	s := ""
	for i := 0; i < n; i++ {
		s += "a"
	}
	return s
}

func concatWithBuilder(n int) string {
	var builder strings.Builder
	for i := 0; i < n; i++ {
		builder.WriteString("a")
	}
	return builder.String()
}

func main() {
	const count = 10000
	
	start := time.Now()
	concatWithPlus(count)
	fmt.Println("使用 + 拼接耗时:", time.Since(start))
	
	start = time.Now()
	concatWithBuilder(count)
	fmt.Println("使用 Builder 拼接耗时:", time.Since(start))
}
```

输出示例：
```
使用 + 拼接耗时: 6.762304ms 
使用 Builder 拼接耗时: 24.197µs
```

## 注意事项

1. `strings.Builder` 不是线程安全的，不能在多个 goroutine 中并发使用
2. 一旦调用 `String()` 方法后，后续可以继续添加内容
3. 不要复制 Builder 值（应使用指针传递）

`strings.Builder` 是 Go 1.10 引入的，对于需要高性能字符串拼接的场景，它是首选方案。