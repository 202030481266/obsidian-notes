C++20 确实引入了许多强大且实用的语言特性和语法改进。我来为您介绍一些最显著的特性以及它们的最佳实践。

## 概念与模块 (Concepts and Modules)

### 模块系统 (Modules)

C++20 引入了现代模块系统，替代了传统的头文件包含机制。

**特点：**

- 减少编译时间
- 避免宏和包含顺序问题
- 更好的封装

**最佳实践：**

- 使用 `export module my_module;` 定义模块
- 使用 `import module_name;` 导入模块
- 通过 `export` 关键字显式导出 API
- 将实现细节保留在模块内部，不导出

### 概念 (Concepts)

概念允许在模板参数上进行约束，提高代码的可读性和错误信息的质量。

**最佳实践：**

- 定义有意义的概念来表达需求
- 使用标准库概念（如 `std::integral`、`std::equality_comparable`）
- 组合现有概念创建复杂约束
- 与 `requires` 子句配合使用

```cpp
template<typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

template<Hashable T>
void process(T value) {
    // 处理可哈希的类型...
}
```

## 协程 (Coroutines)

C++20 引入了语言级协程支持，使异步编程更加简洁。

**特点：**

- 非阻塞 I/O 操作
- 简化异步和并发代码
- 懒惰计算和无限序列

**最佳实践：**

- 使用 `co_await`、`co_yield` 和 `co_return` 关键字
- 为特定领域实现自定义协程
- 结合 `std::future` 和协程简化异步代码
- 协程适合 I/O 密集型任务，不适合 CPU 密集型任务

```cpp
std::generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}
```

## 范围库 (Ranges)

范围库提供了更声明式的容器操作方式。

**特点：**

- 链式操作
- 惰性求值
- 更少的临时容器

**最佳实践：**

- 使用管道语法 (`|`) 连接操作
- 利用标准视图（如 `std::views::filter`、`std::views::transform`）
- 用范围代替迭代器对，提高代码可读性
- 组合视图构建复杂数据转换

```cpp
std::vector<int> nums = {1, 2, 3, 4, 5, 6};
auto even_squared = nums 
    | std::views::filter([](int n) { return n % 2 == 0; })
    | std::views::transform([](int n) { return n * n; });
```

## 三向比较运算符 (Spaceship Operator)

新的 `<=>` 运算符简化了比较操作。

**最佳实践：**

- 使用 `auto operator<=>(const MyClass&) const = default;` 自动生成所有比较运算符
- 为复杂对象定义自定义三向比较
- 利用 `std::strong_ordering`、`std::weak_ordering` 或 `std::partial_ordering` 表达比较语义

```cpp
struct Point {
    int x, y;
    auto operator<=>(const Point&) const = default; // 自动生成所有比较运算符
};
```

## 格式库 (Format Library)

类似 Python 的格式化功能，更安全、更灵活。

**最佳实践：**

- 使用 `std::format` 替代 `printf` 和 `std::stringstream`
- 利用命名参数提高可读性
- 自定义类型的格式化通过特化 `std::formatter` 实现
- 使用 `std::format_to` 提高性能

```cpp
std::string message = std::format("Hello, {}! You are {} years old.", name, age);
```

## 其他实用特性

### 指定初始化 (Designated Initializers)

```cpp
struct Point { int x; int y; };
Point p = {.x = 1, .y = 2}; // 清晰明了
```

### constexpr 的扩展

- 更多标准库函数支持 constexpr
- constexpr 虚函数
- constexpr 动态内存分配

**最佳实践：** 尽可能使用 constexpr 实现编译期计算，提高运行时性能

### std::span

非拥有型连续内存视图，替代指针和长度参数。

**最佳实践：**

- 用于函数参数传递连续序列
- 避免拷贝，提高性能
- 替代不安全的原始指针接口

```cpp
void process(std::span<int> data) {
    for (auto& item : data) {
        // 处理数据...
    }
}
```

### 特性测试宏

通过 `__cpp_*` 宏检测特定功能是否可用。

**最佳实践：** 使用特性测试宏编写兼容多个编译器的代码，而不是依赖 `__cplusplus` 值。

## 综合建议

1. **渐进采用**：先在新代码中使用 C++20 特性，再逐步更新旧代码
2. **理解概念**：深入学习概念和范围，它们是 C++20 中最革命性的变化
3. **使用标准库**：尽可能使用标准库提供的新功能，而不是自行实现
4. **性能意识**：了解新特性的性能特征，特别是协程和范围
5. **跨平台考虑**：检查目标编译器对 C++20 特性的支持情况

C++20 是 C++ 标准的一次重大更新，通过引入这些特性，使 C++ 编程更加现代化、安全和高效。

---

不经意从水群中知道 `[[likely]]` 这种语法。这是 C++20 中引入的一个重要的属性标注(attribute)，用于提供分支预测的提示。

## [[likely]] 和 [[unlikely]] 属性

`[[likely]]` 和 `[[unlikely]]` 是 C++20 引入的标准属性，用于向编译器提供分支预测的提示，帮助编译器优化代码生成。

### 基本用法

```cpp
if ([[likely]] x > 0) {
    // 编译器会优化这个分支，假设它更可能被执行
    // ...
} else {
    // 编译器认为这个分支不太可能执行
    // ...
}

if ([[unlikely]] error_occurred) {
    // 编译器会优化假设这个分支不太可能执行
    // ...
}
```

### 工作原理

- 这些属性给编译器提供了关于条件分支可能性的提示
- 编译器可以据此优化指令排序、分支预测和代码布局
- 在现代处理器上，正确的分支预测可以显著提高性能，减少流水线停顿

### 最佳实践

1. **基于数据使用**：只有在有明确的数据表明某个分支明显更可能发生时才使用
    
2. **常见用例**：
    
    - 错误处理路径标记为 `[[unlikely]]`
    - 热点路径标记为 `[[likely]]`
    - 边界检查标记为 `[[unlikely]]`
3. **性能关键代码**：主要在性能关键的循环或频繁调用的函数中使用
    
4. **避免过度使用**：
    
    - 不应该在没有证据支持的情况下猜测
    - 错误的提示可能适得其反，降低性能
5. **结合 profile-guided optimization (PGO)**：最好与 PGO 结合使用，让实际运行数据指导优化
    

### 语法注意事项

这些属性可以应用于 if 语句的条件表达式，也可以应用于 switch 语句的 case 标签：

```cpp
switch (value) {
    [[likely]] case 0:
        // 最常见的情况
        break;
    [[unlikely]] case 1:
        // 罕见情况
        break;
    // ...
}
```

### 与旧版本编译器的兼容性

在不支持这些属性的编译器中，它们会被安全地忽略。如果需要跨平台兼容，可以使用条件宏：

```cpp
#if defined(__cpp_attributes) && __cpp_attributes >= 201803L
    #define LIKELY [[likely]]
    #define UNLIKELY [[unlikely]]
#else
    #define LIKELY
    #define UNLIKELY
#endif

if (LIKELY x > 0) {
    // ...
}
```

### 性能影响

在正确使用的情况下，这些属性可以显著提高分支密集型代码的性能，特别是在：

- 处理罕见错误情况
- 实现性能关键算法
- 优化热点路径

这些属性是 C++ 标准化的语法，替代了之前各编译器提供的非标准分支提示方式（如 GCC 的 `__builtin_expect`）。