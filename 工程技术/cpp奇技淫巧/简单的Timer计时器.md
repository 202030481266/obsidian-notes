使用`std`库中的`chrono`库可以获取高精度的时间戳。而且利用作用域的方法就可以很优雅地计时我们想要执行的函数。利用构造函数和析构函数的特性来自动计时。这种技术称为RAII（Resource Acquisition Is Initialization，资源获取即初始化），非常适合用来进行函数执行时间的测量。

```cpp
#include <iostream>
#include <chrono>
#include <string>

class Timer {
private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_start;

public:
    // 构造函数：开始计时
    Timer(const std::string& name = "Timer") : m_name(name) {
        m_start = std::chrono::high_resolution_clock::now();
        std::cout << m_name << " 开始计时..." << std::endl;
    }

    // 析构函数：结束计时并输出时间
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
        std::cout << m_name << " 耗时: " 
                  << duration << " 微秒 (" 
                  << static_cast<double>(duration) / 1000 << " 毫秒)" << std::endl;
    }
};

// 使用示例
void someFunction() {
    // 创建Timer对象，开始计时
    Timer t("someFunction");
    
    // 执行需要计时的代码
    // 模拟耗时操作
    for (volatile int i = 0; i < 1000000; ++i) {}
}

int main() {
    std::cout << "测试计时器" << std::endl;
    
    // 方法1：直接在需要计时的作用域创建Timer对象
    {
        Timer t("Block 1");
        // 模拟一些操作
        for (volatile int i = 0; i < 500000; ++i) {}
    } // 当离开作用域时，Timer析构函数会被调用，输出执行时间
    
    // 方法2：在函数中使用
    someFunction();
    
    // 方法3：内联使用
    {
        Timer t("复杂计算");
        // 更复杂的操作
        for (volatile int i = 0; i < 2000000; ++i) {}
    }
    
    return 0;
}

```

这个计时器的设计非常巧妙，主要优点包括：

1. **自动计时**：利用C++对象的生命周期，在构造时开始计时，在析构时结束计时并输出结果
2. **使用简便**：只需创建一个Timer对象，无需手动调用开始/结束函数
3. **作用域绑定**：当Timer对象离开作用域时自动计算并显示时间
4. **高精度**：使用std::chrono库提供的高精度时钟
5. **可命名**：可以给每个计时器指定名称，方便识别不同的计时结果

使用方法非常简单：只需在要计时的代码块开始处创建一个Timer对象，当代码执行完毕离开作用域时，Timer对象会被自动销毁，并在析构函数中计算并输出执行时间。
