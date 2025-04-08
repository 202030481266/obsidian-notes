## 什么是上下文管理器

上下文管理器(Context Manager)是Python中一种特殊的对象，用于管理资源的获取和释放。它可以让你在特定代码块执行前进行设置，执行后进行清理，确保资源被正确处理，即使发生异常也能执行清理代码。

上下文管理器主要通过`with`语句使用，核心思想是"***获取资源-使用资源-释放资源***"的模式自动化。（也可以理解为，是`try/finally`模式的简化）

## with语句基础

基本语法：

```python
with expression as variable:
    # 代码块
```

当Python解释器执行此代码时会：

1. 计算expression，获取上下文管理器
2. 调用上下文管理器的`__enter__()`方法
3. 将`__enter__()`返回值赋给variable(如果有as子句)
4. 执行with语句体中的代码块
5. 无论代码块是否抛出异常，都会调用上下文管理器的`__exit__()`方法清理资源

## 常见的内置上下文管理器

### 文件操作（此方法必须使用as子句）

```python
with open('filename.txt', 'r') as f:
    content = f.read()
# 文件会在with块结束后自动关闭
```

### 锁管理

```python
import threading

lock = threading.Lock()

with lock:
    # 临界区代码
    # 离开with块时锁会自动释放
```

---
## 自定义上下文管理器

### 方法1：定义类

```python
class MyContextManager:
    def __init__(self, resource):
        self.resource = resource
        
    def __enter__(self):
        print("获取资源")
        return self.resource  # 返回值会被赋给as变量
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")
        # 返回True会吞掉异常，返回False或None会让异常继续传播
        return False  # 默认让异常继续传播，除非返回True或者None
```

### 方法2：使用contextlib.contextmanager装饰器

```python
from contextlib import contextmanager

@contextmanager
def my_context_manager(resource):
    print("获取资源")
    try:
        yield resource  # yield的值会被赋给as变量
    finally:
        print("释放资源")  # 无论是否发生异常，都会执行
```

下面是一个更加详细的例子，实现逆序的字符串输出：

```python
class LookingClas:

    def __enter__(self):
        import sys
        self.original_write = sys.stdout.write
        sys.stdout.write = self.reverse_write
        return "Hello! I'm a context mannager variable!"

    def reverse_write(self, text):
        self.original_write(text[::-1])

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout.write = self.original_write
        if exc_type is ZeroDivisionError:
            print("Please don't divide by zero!")
            return True
```

---
## contextlib模块详解

`contextlib`是Python标准库提供的上下文管理工具模块。

### @contextmanager装饰器（最为重要！）

@contextmanager 装饰器能减少创建上下文管理器的样板代码量，因为不用编写一个完整的类，定义 __enter__ 和 __exit__ 方法，而只需实现有一个 yield 语句的生成器，生成想让__enter__ 方法返回的值。在使用 @contextmanager 装饰的生成器中， **yield 语句的作用是把函数的定义体分成两部分： yield 语句前面的所有代码在 with 块开始时（即解释器调用 enter 方法时）执行， yield 语句后面的代码在 with 块结束时（即调用 exit 方法时）执行**：

```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    try:
        f = open(filename, mode)
        yield f
    finally:
        f.close()

# 使用
with file_manager('test.txt', 'w') as f:
    f.write('hello world')
```

上面的那个例子也可以写成：

```python
@contextmanager
def looking_class():
    import sys
    original_write = sys.stdout.write

    def reverse_write(text):
        original_write(text[::-1])

    try:
        sys.stdout.write = reverse_write
        yield "Hello! I'm a context mannager variable!"
    except ZeroDivisionError:
        print(original_write("Please don't divide by zero!"))
        return True
    finally:
        sys.stdout.write = original_write
```

`contextlib.contextmanager`这个类的 __enter__ 方法有如下作用。
(1) 调用生成器函数，保存生成器对象（这里把它称为 gen）。
(2) 调用 next(gen)，执行到 yield 关键字所在的位置。
(3) 返回 next(gen) 产出的值，以便把产出的值绑定到 with/as 语句中的目标变量上。
with 块终止时， __exit__ 方法会做以下几件事。
(1) 检查有没有把异常传给 exc_type；如果有，调用 gen.throw(exception)，在生成器函数定义体中包含 yield 关键字的那一行抛出异常。
(2) 否则，调用 next(gen)，继续执行生成器函数定义体中 yield 语句之后的代码。

### closing函数

将一个只有close()方法的对象转换为上下文管理器:

```python
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen('http://www.python.org')) as page:
    data = page.read()
# page.close()会被自动调用
```

### suppress上下文管理器

忽略指定的异常:

```python
from contextlib import suppress
import os

# 忽略文件不存在的异常
with suppress(FileNotFoundError):
    os.remove('可能不存在的文件.txt')
```

### nullcontext上下文管理器

提供一个什么都不做的上下文管理器:

```python
from contextlib import nullcontext

# 根据条件决定是否需要锁
lock = threading.Lock() if needs_lock else nullcontext()

with lock:
    # 执行操作
```

### ExitStack类

动态管理多个上下文管理器（这里是**后进先出**的顺序调用exit退出）:

```python
from contextlib import ExitStack

def process_files(file_list):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname)) for fname in file_list]
        # 所有文件会在with块结束时自动关闭
        for file in files:
            print(file.read())
```

---
## 实际应用最佳实践

### 1. 数据库连接管理

```python
@contextmanager
def db_transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise
```

### 2. 临时改变工作目录

```python
@contextmanager
def change_dir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)
```

### 3. 计时上下文管理器

```python
@contextmanager
def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.2f} seconds")
```

### 4. 嵌套与组合上下文管理器

```python
# 嵌套方式
with open('input.txt') as in_file:
    with open('output.txt', 'w') as out_file:
        out_file.write(in_file.read())

# 使用逗号组合多个上下文
with open('input.txt') as in_file, open('output.txt', 'w') as out_file:
    out_file.write(in_file.read())
```

### 5. 资源池管理

```python
class ResourcePool:
    def __init__(self, resources):
        self.resources = resources
        self.in_use = set()
        self.lock = threading.Lock()
    
    @contextmanager
    def acquire(self):
        with self.lock:
            resource = next(r for r in self.resources if r not in self.in_use)
            self.in_use.add(resource)
        try:
            yield resource
        finally:
            with self.lock:
                self.in_use.remove(resource)
```

---
## 拓展阅读（Python Cookbook 3rd）

# [8.3 让对象支持上下文管理协议](https://python3-cookbook.readthedocs.io/zh-cn/latest/c08/p03_make_objects_support_context_management_protocol.html#id1 "永久链接至标题")

### **问题**

你想让你的对象支持上下文管理协议(with语句)。

### **解决方案**

为了让一个对象兼容 `with` 语句，你需要实现 `__enter__()` 和 `__exit__()` 方法。 例如，考虑如下的一个类，它能为我们创建一个网络连接：

```python
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.sock = None

    def __enter__(self):
        if self.sock is not None:
            raise RuntimeError('Already connected')
        self.sock = socket(self.family, self.type)
        self.sock.connect(self.address)
        return self.sock

    def __exit__(self, exc_ty, exc_val, tb):
        self.sock.close()
        self.sock = None
```

当然，更好的做法要考虑并发环境，下面是线程安全版本的实现，每个线程都维护一个自己的实例字典：

```python
from socket import socket, AF_INET, SOCK_STREAM
import threading

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = AF_INET
        self.type = SOCK_STREAM
        self.local = threading.local()

    def __enter__(self):
        if hasattr(self.local, 'sock'):
            raise RuntimeError('Already connected')
        self.local.sock = socket(self.family, self.type)
        self.local.sock.connect(self.address)
        return self.local.sock

    def __exit__(self, exc_ty, exc_val, tb):
        self.local.sock.close()
        del self.local.sock
```

这个类的关键特点在于它表示了一个网络连接，但是初始化的时候并不会做任何事情(比如它并没有建立一个连接)。 连接的建立和关闭是使用 `with` 语句自动完成的，例如：

```python
from functools import partial

conn = LazyConnection(('www.python.org', 80))
# Connection closed
with conn as s:
    # conn.__enter__() executes: connection open
    s.send(b'GET /index.html HTTP/1.0\r\n')
    s.send(b'Host: www.python.org\r\n')
    s.send(b'\r\n')
    resp = b''.join(iter(partial(s.recv, 8192), b''))
    # conn.__exit__() executes: connection closed
```

### **讨论**

编写上下文管理器的主要原理是你的代码会放到 `with` 语句块中执行。 当出现 `with` 语句的时候，对象的 `__enter__()` 方法被触发， 它返回的值(如果有的话)会被赋值给 `as` 声明的变量。然后，`with` 语句块里面的代码开始执行。 最后，`__exit__()` 方法被触发进行清理工作。

不管 `with` 代码块中发生什么，上面的控制流都会执行完，就算代码块中发生了异常也是一样的。 事实上，`__exit__()` 方法的三个参数包含了异常类型、异常值和追溯信息(如果有的话)。 `__exit__()` 方法能自己决定怎样利用这个异常信息，或者忽略它并返回一个None值。 如果 `__exit__()` 返回 `True` ，那么异常会被清空，就好像什么都没发生一样， `with` 语句后面的程序继续在正常执行。

还有一个细节问题就是 `LazyConnection` 类是否允许多个 `with` 语句来嵌套使用连接。 很显然，上面的定义中一次只能允许一个socket连接，如果正在使用一个socket的时候又重复使用 `with` 语句， 就会产生一个异常了。不过你可以像下面这样修改下上面的实现来解决这个问题（**==上下文管理器的思路天然和操作系统的调用相似，资源获取之后一定会释放，并且是符合栈的概念的，因此可以使用栈来模拟==**）：

```python
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.connections = []

    def __enter__(self):
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        self.connections.append(sock)
        return sock

    def __exit__(self, exc_ty, exc_val, tb):
        self.connections.pop().close()

# Example use
from functools import partial

conn = LazyConnection(('www.python.org', 80))
with conn as s1:
    pass
    with conn as s2:
        pass
        # s1 and s2 are independent sockets
```

在第二个版本中，`LazyConnection` 类可以被看做是某个连接工厂。在内部，一个列表被用来构造一个栈。 每次 `__enter__()` 方法执行的时候，它复制创建一个新的连接并将其加入到栈里面。 `__exit__()` 方法简单的从栈中弹出最后一个连接并关闭它。 这里稍微有点难理解，不过它能允许嵌套使用 `with` 语句创建多个连接，就如上面演示的那样。

在需要管理一些资源比如文件、网络连接和锁的编程环境中，使用上下文管理器是很普遍的。 这些资源的一个主要特征是它们必须被手动的关闭或释放来确保程序的正确运行。 例如，如果你请求了一个锁，那么你必须确保之后释放了它，否则就可能产生死锁。 通过实现 `__enter__()` 和 `__exit__()` 方法并使用 `with` 语句可以很容易的避免这些问题， 因为 `__exit__()` 方法可以让你无需担心这些了。

---
## 杂谈与上下文管理器的思想：取出面包

在 PyCon US 2013 的主题演讲“What Makes Python Awesome”中（http://pyvideo.org/ video/1669/keynote-3）， Raymond Hettinger 说他第一次看到 with 语句的提案时，觉得“有点晦涩难懂”。这和我一开始的反应类似。 PEP 通常难以阅读， PEP 343 尤其如此。

然后， Hettinger 告诉我们，他认识到在计算机语言的发展历程中，子程序是最重要的发明。如果有一系列操作，如 A-B-C 和 P-B-Q，那么可以把 B 拿出来，变成子程序。这就好比把三明治的馅儿取出来，这样我们就能使用金枪鱼搭配不同的面包。可是，如果我们想把面包取出来，使用小麦面包夹不同的馅儿呢？这就是 with 语句实现的功能。 with 语句是子程序的补充。 Hettinger 接着说道：

> with 语句是非常了不起的特性。我建议你在实践中深挖这个特性的用途。使用with 语句或许能做意义深远的事情。 with 语句最好的用法还未被发掘出来。我预料，如果有好的用法，其他语言以及未来的语言会借鉴这个特性。或许，你正在参与的事情几乎与子程序的发明一样意义深远。

---
## 总结

上下文管理器是Python中强大的资源管理工具，通过`with`语句提供了一种优雅的方式来确保资源的正确获取和释放。

优点:

- 代码更加简洁、可读
- 自动处理异常情况
- 确保资源正确释放
- 减少资源泄漏风险

最佳实践:

- 对需要清理的资源使用上下文管理器
- 使用`contextlib`简化上下文管理器的创建
- 利用`ExitStack`管理多个动态确定的上下文
- 在需要的地方创建自定义上下文管理器，提高代码质量

记住，**好的上下文管理器应该遵循RAII原则(资源获取即初始化)，确保资源在不再需要时被正确释放**。