## 概述

Python中的`else`子句不仅适用于`if`语句，还能与`for`、`while`和`try`语句搭配使用。这一特性虽然不是什么秘密，但往往被忽视。`for/else`、`while/else`和`try/else`的语义与`if/else`有显著不同，刚开始接触时，`else`这个词的含义可能会让人困惑。不过，一旦习惯了这种用法，就会发现它非常自然。使用`else`子句通常能让代码更易读，还能避免设置额外的控制标志或多余的`if`语句。

## for语句中的else

在`for`循环中，`else`块只有在循环完整运行完毕时才会执行，也就是说，如果循环被`break`语句中止，`else`块就不会运行。例如，下面的代码片段展示了一个典型用法：

```python
for item in my_list:
    if item.flavor == 'banana':
        break
else:
    raise ValueError('No banana flavor found!')
```

这段代码的意思是：如果在`my_list`中找到香蕉口味，循环会提前退出；如果循环跑完都没找到（没有触发`break`），就会抛出一个异常。

## while语句中的else

与`for`类似，`while`循环中的`else`块只有在循环因条件为假而自然退出时才会执行。如果循环被`break`中止，`else`块将被跳过。这种行为与`for/else`的逻辑一致，但在`while`的上下文中更依赖于条件的变化。

## try语句中的else

在`try`语句中，`else`块只有在`try`块没有抛出任何异常时才会运行。[官方文档](https://docs.python.org/3/reference/compound_stmts.html)还特别指出，`else`块中抛出的异常不会被前面的`except`子句处理。以下是一个常见的错误写法和改进版本的对比。

先看不推荐的写法：

```python
try:
    dangerous_call()
    after_call()
except OSError:
    log('OSError...')
```

这种写法的问题在于，`try`块包含了两个函数调用，如果`after_call()`抛出异常，逻辑会变得模糊。更清晰的写法是：

```python
try:
    dangerous_call()
except OSError:
    log('OSError...')
else:
    after_call()
```

改进后的代码明确了`try`块只负责处理`dangerous_call()`的潜在异常，而`after_call()`只在`try`块成功时执行。这样既提高了可读性，也更符合错误处理的精确性。

## else块的跳过规则

无论是`for`、`while`还是`try`，如果出现异常，或者执行了`return`、`break`、`continue`等语句，导致控制权跳出复合语句的主体，`else`块都会被跳过。这一规则统一了所有场景下的行为。

## 对else关键字的反思（吐槽）

选择`else`作为这些语句的关键字可能并非最理想的设计。`else`通常暗示“排他性”，比如“要么这个，要么那个”。但在循环中，它的实际语义更像是“然后”——“运行完循环，然后做某事”。在`try`中也可以理解为“尝试这个，然后做那个”。相比之下，`then`可能是个更直观的替代词。然而，添加新关键字是语言设计的重大变动，Guido（Python之父）对此持谨慎态度。尽管如此，习惯了`else`的用法后，它仍然是一个实用的工具。

## 使用else的优势

在这些语句中使用`else`子句，往往能让代码更简洁、更直观。**特别是在需要表达“某操作成功完成后的后续步骤”时，它能省去额外的标志变量或条件判断（一般来说都需要一个标志位如果不使用else）**。这种特性在实际编码中非常实用，尤其是在复杂逻辑中。

## Python编程风格的背景

Python中`try/except`不仅用于错误处理，还常用于控制流程。这与一种名为EAFP的编程风格有关，全称是“==**Easier to Ask for Forgiveness than Permission**==”**（取得原谅比获得许可容易）。EAFP的核心是先假定条件成立，如果失败则捕获异常。这种风格在Python中很常见，代码中会有很多`try`和`except`语句，简洁明了（参考[Python官方词汇表](https://docs.python.org/3/glossary.html#term-eafp)）。

与之相对的是LBYL风格，全称“==**Look Before You Leap**==”（三思而后行）。这种风格在调用函数或访问属性前会显式检查条件，常见于C语言等其他语言中。LBYL的缺点是在多线程环境中可能引入条件竞争。例如，`if key in mapping: return mapping[key]`这样的代码，如果在检查和访问之间键被删除，就会失败。解决办法要么使用锁，要么转向EAFP风格。

如果选择EAFP风格，深入理解`try/else`就显得尤为重要。合理使用`else`子句能让代码在处理异常和控制流程时更加优雅。

## 总结

`else`在`for`、`while`和`try`中的应用是一个容易被忽视但非常强大的特性。它不仅能简化代码，还体现了Python的哲学。结合EAFP风格，`try/else`在错误处理和流程控制中尤为出色。只要适应了它的语义，就能在编程中发挥更大作用。

下面是一个练手的例子，使用else语句在EAFP风格的编码中极其有用！

```python
from typing import List


def use_else_exmaple(input_list: List[int]):
    for i in input_list:
        if i != 0:
            print(i)
        else:
            break
    else:
        print("Successfully print all the numbers!")
        return
    print("Found zero element in input list!")


def dangerous_call(input_value: int):
    if input_value == 0:
        raise Exception("Dangerous Call!")

def after_call(x: int, y: int) -> float:
    if x < 0:
        raise Exception("Bad Call!")
    return x / y

def use_else_exmaple_with_exception(x: int, y: int):
    try:
        dangerous_call(y)
    except Exception as e:
        print(f"{str(e)}")
    else:
        try:
            div = after_call(x, y)
        except Exception as e:
            print(f"{str(e)}")
        else:
            print("x divide y is %.2f" % div)


if __name__ == "__main__":
    use_else_exmaple([1, 2, 3, 0, 4, 5])
    use_else_exmaple([1, 2, 3, 4, 5])

    use_else_exmaple_with_exception(1, 0)
    use_else_exmaple_with_exception(-1, 1)
    use_else_exmaple_with_exception(1, 2)
```
