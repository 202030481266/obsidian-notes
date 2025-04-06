# if 和简短语句

和 `for` 一样，`if` 语句可以在条件表达式前执行一个简短语句。该语句声明的变量作用域仅在 `if` 之内。（在最后的 `return` 语句处使用 `v` 看看。）

这是一个Go的语法糖！虽然对于经常使用C-like风格的语言来说很难适应。

```go
package main

import (
	"fmt"
	"math"
)

func pow(x, n, lim float64) float64 {
	if v := math.Pow(x, n); v < lim {
		return v
	}
	return lim
}

func main() {
	fmt.Println(
		pow(3, 2, 10),
		pow(3, 3, 20),
	)
}
```

---
# if 和 else

在 `if` 的简短语句中声明的变量同样可以在对应的任何 `else` 块中使用。（在 `main` 的 `fmt.Println` 调用开始前，两次对 `pow` 的调用均已执行并返回其各自的结果。）

```go
package main

import (
	"fmt"
	"math"
)

func pow(x, n, lim float64) float64 {
	if v := math.Pow(x, n); v < lim {
		return v
	} else {
		fmt.Printf("%g >= %g\n", v, lim)
	}
	// can't use v here, though
	return lim
}

func main() {
	fmt.Println(
		pow(3, 2, 10),
		pow(3, 3, 20),
	)
}
```

Go 语言对 `if/else` 语句的格式有严格的要求，`else` 或 `else if` 必须紧跟在 `if` 代码块的右花括号 `}` 之后，并且在同一行。这会让来自 C/C++/Java/Python 等语法更灵活的语言的开发者感到有些惊讶甚至“震惊”。

这种设计主要是基于以下几个原因：

1.  **强制统一的代码风格 (Code Formatting and Consistency)**：
    *   Go 语言的设计哲学之一就是强调代码的可读性和一致性。通过强制规定 `else` 的位置，可以消除关于代码格式的无谓争论（比如 `else` 应该放在哪里）。
    *   这与 Go 语言强大的自动化工具 `gofmt` 密切相关。`gofmt` 会自动将你的代码格式化为官方推荐的标准样式。这种严格的语法规则使得 `gofmt` 的实现更简单，并且确保了所有 Go 代码在格式上看起来都非常相似，降低了阅读和维护他人代码的认知负担。

2.  **避免“悬挂 else” (Dangling Else) 问题和简化解析 (Parsing Simplification)**：
    *   虽然 Go 语言强制使用花括号 `{}` 包裹代码块，已经从根本上解决了 C/C++ 等语言中经典的“悬挂 else”（即 `else` 到底与哪个 `if` 匹配）的歧义问题，但这种严格的 `else` 位置规则进一步简化了语法解析器的设计。
    *   更重要的是，这与 Go 的 **自动分号插入 (Automatic Semicolon Insertion)** 规则有关。

3.  **自动分号插入规则 (The Key Technical Reason)**：
    *   Go 语言的语法规范定义了在何处自动插入分号。简单来说，如果一个新行（换行符）前面的最后一个标记（token）是以下几种之一，编译器（的词法分析器）就会在该标记后自动插入一个分号：
        *   标识符（identifier）
        *   整数、浮点数、虚数、字符或字符串字面量（literal）
        *   关键字 `break`, `continue`, `fallthrough`, `return`
        *   运算符 `++`, `--`
        *   右括号 `)`, 右方括号 `]`, 或右花括号 `}`
    *   考虑如果允许这样写：
        ```go
        if condition {
            // ...
        } // <--- 如果这里换行
        else { // <--- 'else' 在新的一行
            // ..
        }
        ```
    *   根据自动分号插入规则，因为 `}` 是可以触发分号插入的标记，所以在 `}` 后面和换行符之前，编译器会**自动插入一个分号**。代码实际上会被解析成：
        ```go
        if condition {
            // ...
        }; // <--- 自动插入的分号，结束了 if 语句
        else { // <--- 单独的 else 会导致语法错误，因为它没有紧跟着一个 if
            // ...
        }
        ```
    *   这会导致一个语法错误，因为 `else` 不能独立存在，它必须是 `if` 语句的一部分。
    *   通过强制要求 `} else {` 在同一行，可以确保在 `}` 和 `else` 之间不会因为换行而插入分号，从而保持 `if-else` 结构的完整性。

**总结:**

Go 语言强制 `else` 紧跟在 `}` 之后且在同一行，主要是为了：

*   **技术上**：配合自动分号插入规则，避免语法解析错误。
*   **风格上**：强制统一代码格式，提高可读性和一致性，并简化 `gofmt` 等工具的实现。

虽然初看起来可能有点奇怪或不习惯，但这是 Go 语言为了实现其设计目标（简洁、清晰、一致、易于工具化）而做出的权衡。习惯之后，大多数 Go 开发者会发现这种风格有助于保持代码库的整洁。

---
# switch 分支

`switch` 语句是编写一连串 `if - else` 语句的简便方法。它运行第一个 `case` 值 值等于条件表达式的子句。

Go 的 `switch` 语句类似于 C、C++、Java、JavaScript 和 PHP 中的，不过 Go 只会运行选定的 `case`，而非之后所有的 `case`。 在效果上，Go 的做法相当于这些语言中为每个 `case` 后面自动添加了所需的 `break` 语句（C-like的做法才是违背人类直觉的好吧！）。在 Go 中，除非以 `fallthrough` 语句结束，否则分支会自动终止。 Go 的另一点重要的不同在于 `switch` 的 `case` 无需为常量，且取值不限于整数。

```go
package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Print("Go 运行的系统环境：")
	switch os := runtime.GOOS; os {
	case "darwin":
		fmt.Println("macOS.")
	case "linux":
		fmt.Println("Linux.")
	default:
		// freebsd, openbsd,
		// plan9, windows...
		fmt.Printf("%s.\n", os)
	}
}
```

这里有一个技巧叫做：**无条件switch**。无条件的 `switch` 同 `switch true` 一样。这种形式能将一长串 `if-then-else` 写得更加清晰。

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	t := time.Now()
	switch {
	case t.Hour() < 12:
		fmt.Println("早上好！")
	case t.Hour() < 17:
		fmt.Println("下午好！")
	default:
		fmt.Println("晚上好！")
	}
}
```

---
# defer 推迟

defer 语句会将函数推迟到外层函数返回之后执行。推迟调用的函数其参数会立即求值，但直到外层函数返回前该函数都不会被调用。

这是属于GO的语法糖！

```go
package main

import "fmt"

func main() {
	defer fmt.Println("world")

	fmt.Println("hello")
}
```

推迟调用的函数调用会被压入一个栈中。 当外层函数返回时，被推迟的调用会按照后进先出的顺序调用。

```go
package main

import "fmt"

func main() {
	fmt.Println("counting")

	for i := 0; i < 10; i++ {
		defer fmt.Println(i)
	}

	fmt.Println("done")
}
/*
counting
done
9
8
7
6
5
4
3
2
1
0
*/
```


