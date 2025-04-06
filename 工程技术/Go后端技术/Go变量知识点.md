GO的语法其实很简单而且相当简洁，这里仅仅记录一些学习中容易忽视的语法知识点（语法糖），和其他语言相比属于特性的内容。

---
# 导出名

在 Go 中，如果一个名字以大写字母开头，那么它就是已导出的。例如，`Pizza` 就是个已导出名，`Pi` 也同样，它导出自 `math` 包。

`pizza` 和 `pi` 并未以大写字母开头，所以它们是未导出的。

在导入一个包时，你只能引用其中已导出的名字。 任何「未导出」的名字在该包外均无法访问。

执行代码，观察错误信息。

要修复错误，请将 `math.pi` 改名为 `math.Pi`，然后再试着执行一次。

```go
package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Println(math.pi) // error
	fmt.Println(math.Pi) // right
}
```

---
# 带名字的返回值

Go 的返回值可被命名，它们会被视作定义在函数顶部的变量。

返回值的命名应当能反应其含义，它可以作为文档使用。

没有参数的 `return` 语句会直接返回已命名的返回值，也就是「裸」返回值。

裸返回语句应当仅用在下面这样的短函数中。在长的函数中它们会影响代码的可读性。

```go
package main

import "fmt"

func split(sum int) (x, y int) {
	x = sum * 4 / 9
	y = sum - x
	return
}

func main() {
	fmt.Println(split(17))
}
```

---
# 常量

常量的声明与变量类似，只不过使用 `const` 关键字。

常量可以是字符、字符串、布尔值或数值。

常量不能用 `:=` 语法声明。

```go
package main

import "fmt"

const Pi = 3.14

func main() {
	const World = "世界"
	fmt.Println("Hello", World)
	fmt.Println("Happy", Pi, "Day")

	const Truth = true
	fmt.Println("Go rules?", Truth)
}
```

---
# 基本类型

Go 的基本类型有

```
bool

string

int  int8  int16  int32  int64
uint uint8 uint16 uint32 uint64 uintptr

byte // uint8 的别名

rune // int32 的别名
     // 表示一个 Unicode 码位

float32 float64

complex64 complex128
```

`int`、`uint` 和 `uintptr` 类型在 32-位系统上通常为 32-位宽，在 64-位系统上则为 64-位宽。当你需要一个整数值时应使用 `int` 类型， 除非你有特殊的理由使用固定大小或无符号的整数类型。

```go
package main

import (
	"fmt"
	"math/cmplx"
)

var (
	ToBe   bool       = false
	MaxInt uint64     = 1<<64 - 1
	z      complex128 = cmplx.Sqrt(-5 + 12i)
)

func main() {
	fmt.Printf("类型：%T 值：%v\n", ToBe, ToBe)
	fmt.Printf("类型：%T 值：%v\n", MaxInt, MaxInt)
	fmt.Printf("类型：%T 值：%v\n", z, z)
}
```

---
# 零值

没有明确初始化的变量声明会被赋予对应类型的 **零值**。

零值是：

- 数值类型为 `0`，
- 布尔类型为 `false`，
- 字符串为 `""`（空字符串）。

但是要注意的是，在 Go 语言中，if 语句后面跟的条件表达式**必须**是一个布尔类型 (bool) 的值。Go 不会像 C/C++/Python 那样，将数值类型（或其他类型）隐式地转换为布尔值（即 0 视为 false，非 0 视为 true）。这种设计与 Java 类似，都强调类型安全和代码的明确性。

所以，对于零值：

1. **数值类型 (int, float等):**
    
    - `var i int // i 的零值为 0`
        
    - `if i { ... }` **编译错误！** 必须写成 `if i == 0 { ... } 或 if i != 0 { ... }`
        
2. **布尔类型 (bool):**
    
    - `var b bool // b 的零值为 false`
        
    - `if b { ... }` **这是合法的！** 因为 b 本身就是布尔类型。这里条件判断的就是 false。
        
    - 如果要判断它是否为 false，可以写 `if !b { ... }`
        
3. **字符串类型 (string):**
    
    - `var s string // s 的零值为 "" (空字符串)`
        
    - `if s { ... }` **编译错误！** 必须写成 `if s == "" { ... } 或 if s != "" { ... }`
        
4. **指针、接口、切片、映射、通道 (nil 值):**
    
    - `var p *int // p 的零值为 nil`
        
    - `if p { ... }` **编译错误！** 必须写成 `if p == nil { ... } 或 if p != nil { ... }`
        

**总结:**

Go 语言的设计哲学之一就是“显式优于隐式”。它强制要求 if 条件必须是明确的布尔表达式，这有助于避免 C/C++ 中因隐式转换可能导致的一些潜在错误和混淆（例如误将赋值 if (a = 0) 当作比较 if (a == 0)）。所以，当需要判断一个变量是否为其零值时，你需要使用相应的比较运算符（== 或 !=）来显式地创建一个布尔表达式。

---
# 类型转换

表达式 `T(v)` 将值 `v` 转换为类型 `T`。

一些数值类型的转换：

```go
var i int = 42
var f float64 = float64(i)
var u uint = uint(f)
```

或者，更加简短的形式：

```go
i := 42
f := float64(i)
u := uint(f)
```

与 C 不同的是，Go 在不同类型的项之间赋值时需要**显式转换**，其在类型转换方面非常严格，基本上不允许不同类型之间的隐式转换。这是 Go 语言的一个显著特点，与其他许多主流编程语言（尤其是 C/C++，甚至一定程度上的 Java、Python、JavaScript）相比，显得更为“固执”。

