让静态变量出现在某一些地方，比如某一个函数中，这样子既可以限制访问域，又可以在程序整个声明周期内保持这个变量，我们可以利用这个特性做一些非常有趣的事情。

## 统计某一个函数的访问次数

```cpp
void f() {
	static int access = 0;
	access++;
	std::cout << "Access f() " << access << " times!" << std::endl;
}

int main() {
	for (int i = 0; i < 5; ++i) {
		f();
	}
}
```

我们可以得到这样的输出：

```
Access f() 1 times!
Access f() 2 times!
Access f() 3 times!
Access f() 4 times!
Access f() 5 times!
```

## 单例模式

```cpp
class Singleton {
public:
	Singleton& getSingleton() {
		static Singleton singleton;
		return singleton;
	}

	void getName() { std::cout << "Singleton" << std::endl; }
	
};
```

通过local static，这样子就可以保证每一次从`getSingleton`方法中获取相同的实例。