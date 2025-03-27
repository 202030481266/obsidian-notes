GO的语法其实很简单而且相当简洁，这里仅仅记录一些学习中容易忽视的语法知识点（语法糖），和其他语言相比属于特性的内容。
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

