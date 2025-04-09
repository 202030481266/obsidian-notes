## 目录

- 1. 异步编程与协程基础概念
- 2. asyncio库介绍]
- 3. asyncio核心组件
- 4. 协程与任务详解
- 5. asyncio实战示例
- 6. 异步编程最佳实践
- 7. 常见陷阱与解决方案
- 8. 高级应用场景
- 9. 总结与资源

## 1. 异步编程与协程基础概念

### 1.1 同步 vs 异步

在传统的**同步编程**模型中，执行过程是线性的，一个操作完成后才会开始下一个操作：

```python
def download_file(url):
    # 下载文件，程序会在此处阻塞直到下载完成
    pass

def process_data(data):
    # 处理数据
    pass

# 同步执行：
data = download_file("https://example.com/file")  # 阻塞等待下载完成
process_data(data)  # 只有下载完成才会执行此行
```

而在**异步编程**模型中，程序可以在等待某一操作完成的同时继续执行其他操作：

```python
async def main():
    # 开始下载文件，但不等待其完成
    download_task = asyncio.create_task(download_file("https://example.com/file"))
    
    # 同时可以做其他事情
    other_task = asyncio.create_task(do_something_else())
    
    # 当需要下载结果时再等待其完成
    data = await download_task
    result = await other_task
    
    # 处理数据
    process_data(data, result)
```

### 1.2 协程(Coroutine)概念

**协程**是Python中实现异步编程的基础。它们可以看作是可以在执行过程中被暂停和恢复的特殊函数：

- **定义**：协程是一种特殊的生成器函数，能够在执行过程中暂停并稍后恢复
- **特点**：协程可以在执行到某个点（通常是I/O操作）时主动交出控制权，让程序执行其他任务
- **语法**：在Python中使用`async def`定义协程函数，使用`await`暂停协程执行

```python
async def simple_coroutine():
    print("协程开始")
    # 使用await暂停协程执行，让出控制权
    await asyncio.sleep(1)  
    print("协程恢复")
    return "完成"
```

### 1.3 事件循环(Event Loop)

**事件循环**是asyncio的核心，负责调度和执行协程：

- **作用**：协调多个协程的执行，决定何时运行哪个协程
- **类比**：可以把事件循环想象成一个调度员，它持续监控所有协程的状态，当一个协程等待I/O操作时，事件循环会切换到另一个可以执行的协程
- **工作流程**：
    1. 事件循环运行协程A
    2. 协程A执行到await语句，暂停并告诉事件循环它正在等待某个操作
    3. 事件循环切换到协程B继续执行
    4. 当协程A等待的操作完成时，事件循环会在适当的时间点恢复协程A的执行

## 2. asyncio库介绍

### 2.1 简史与设计目标

Python的`asyncio`库于Python 3.4引入，在3.5、3.6和3.7版本中逐步完善：

- Python 3.4：引入初始版asyncio库
- Python 3.5：引入`async`/`await`语法
- Python 3.6：改进了协程的实现
- Python 3.7：将asyncio从临时状态转为稳定状态，并添加了`asyncio.run()`等高级API

**设计目标**：

- 提供一种处理并发的方式，避免回调地狱和线程复杂性
- 高效处理大量网络连接和I/O操作
- 简化异步代码编写，提高可读性和可维护性

### 2.2 asyncio与并发的关系

**asyncio不是并行处理，而是并发处理**：

- **并发(Concurrency)**：同时管理多个任务，但不一定同时执行
- **并行(Parallelism)**：同时执行多个任务（通常需要多核CPU）

asyncio在单线程上实现了并发，通过在等待I/O操作时切换任务来提高效率。这特别适合I/O密集型任务，如网络请求、文件操作等。

### 2.3 适用场景

asyncio特别适合以下场景：

- **高并发网络服务**：Web服务器、API服务、代理服务器
- **大量网络I/O操作**：爬虫、API客户端、网络监控
- **实时数据处理**：聊天应用、通知系统、日志处理
- **微服务通信**：服务间协议通信、消息队列处理

## 3. asyncio核心组件

### 3.1 事件循环(Event Loop)详解

事件循环是asyncio的心脏，负责：

- 注册、执行和取消延时调用（协程、回调等）
- 执行网络I/O操作
- 运行子进程
- 处理各种事件

**常用事件循环API**：

```python
# 获取当前事件循环
loop = asyncio.get_running_loop()  # Python 3.7+

# 手动创建事件循环(在Python 3.7+中，通常不需要直接使用这些API)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# 运行协程直到完成
result = loop.run_until_complete(some_coroutine())

# 永久运行事件循环
loop.run_forever()

# 停止事件循环
loop.stop()

# 关闭事件循环
loop.close()
```

在Python 3.7+中，大多数情况下使用`asyncio.run()`就足够了：

```python
async def main():
    # 异步操作
    result = await some_coroutine()
    return result

# asyncio.run()创建事件循环，运行协程，然后关闭事件循环
result = asyncio.run(main())
```

### 3.2 协程(Coroutine)与任务(Task)

**协程**是使用`async def`定义的函数：

```python
async def fetch_data():
    return await some_async_operation()
```

**任务**是协程的包装器，它添加了状态跟踪和结果存储功能：

```python
# 创建任务
task = asyncio.create_task(fetch_data())  # Python 3.7+

# 等待任务完成
result = await task
```

**Task vs Future**：

- `Task`是`Future`的子类
- `Future`表示一个尚未完成的结果
- `Task`是一个特殊的Future，它包装协程并跟踪执行状态

### 3.3 异步上下文管理器

异步上下文管理器允许在进入和退出上下文时执行异步操作：

```python
class AsyncResource:
    async def __aenter__(self):
        # 异步获取资源
        await self.acquire_resource()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 异步释放资源
        await self.release_resource()

# 使用异步上下文管理器
async def use_resource():
    async with AsyncResource() as resource:
        await resource.do_something()
```

### 3.4 异步迭代器

异步迭代器允许对可能产生异步结果的集合进行迭代：

```python
class AsyncIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self.index >= len(self.data):
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # 模拟异步操作
        value = self.data[self.index]
        self.index += 1
        return value

# 使用异步迭代器
async def iterate_async():
    async for item in AsyncIterator([1, 2, 3, 4, 5]):
        print(item)
```

## 4. 协程与任务详解

### 4.1 创建与执行协程

**定义协程**：

```python
async def my_coroutine():
    await asyncio.sleep(1)
    return "结果"
```

**执行协程的几种方式**：

1. 使用`asyncio.run()`（推荐，Python 3.7+）：

```python
result = asyncio.run(my_coroutine())
```

2. 在其他协程中使用`await`：

```python
async def main():
    result = await my_coroutine()
    print(result)

asyncio.run(main())
```

3. 创建任务：

```python
async def main():
    task = asyncio.create_task(my_coroutine())
    result = await task
    print(result)

asyncio.run(main())
```

### 4.2 任务管理

**创建任务**：

```python
async def main():
    # 创建任务（不等待其完成）
    task = asyncio.create_task(my_coroutine())
    
    # 做其他事情...
    
    # 稍后等待任务完成
    result = await task
```

**任务状态**：

- `pending`：任务尚未完成
- `done`：任务已完成
- `cancelled`：任务被取消

**任务集合管理**：

```python
async def main():
    # 创建多个任务
    task1 = asyncio.create_task(my_coroutine1())
    task2 = asyncio.create_task(my_coroutine2())
    task3 = asyncio.create_task(my_coroutine3())
    
    # 等待所有任务完成
    results = await asyncio.gather(task1, task2, task3)
    
    # 或者等待第一个完成的任务
    done, pending = await asyncio.wait(
        [task1, task2, task3], 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # 取消剩余任务
    for task in pending:
        task.cancel()
```

### 4.3 asyncio.gather vs asyncio.wait

**asyncio.gather**：

- 并发运行多个协程并等待它们全部完成
- 返回结果列表，与输入协程顺序相同
- 如果任一任务失败，默认会传播异常

```python
async def main():
    # 并发运行三个协程并收集结果
    results = await asyncio.gather(
        fetch_url("url1"),
        fetch_url("url2"),
        fetch_url("url3"),
        return_exceptions=False  # 设为True时会将异常作为结果返回，而不是传播
    )
    # results是一个包含三个结果的列表
```

**asyncio.wait**：

- 等待多个协程，提供更多控制选项
- 返回(done, pending)两个任务集合
- 可以设置等待策略：
    - `FIRST_COMPLETED`：当任何任务完成时返回
    - `FIRST_EXCEPTION`：当任何任务引发异常时返回
    - `ALL_COMPLETED`：当所有任务完成时返回(默认)

```python
async def main():
    tasks = [
        asyncio.create_task(fetch_url("url1")),
        asyncio.create_task(fetch_url("url2")),
        asyncio.create_task(fetch_url("url3"))
    ]
    
    # 等待第一个完成的任务
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # 处理完成的任务
    for task in done:
        try:
            result = task.result()
            print(f"任务完成：{result}")
        except Exception as e:
            print(f"任务失败：{e}")
```

### 4.4 超时处理

使用`asyncio.wait_for`设置超时：

```python
async def main():
    try:
        # 等待协程，超时时间为5秒
        result = await asyncio.wait_for(
            fetch_url("https://example.com"), 
            timeout=5.0
        )
        print(f"结果：{result}")
    except asyncio.TimeoutError:
        print("操作超时")
```

使用`asyncio.timeout`上下文管理器（Python 3.11+）：

```python
async def main():
    try:
        # 5秒超时上下文
        async with asyncio.timeout(5.0):
            result = await fetch_url("https://example.com")
            print(f"结果：{result}")
    except asyncio.TimeoutError:
        print("操作超时")
```

### 4.5 取消任务

任务可以被取消，这是协程的一个重要优势：

```python
async def main():
    # 创建任务
    task = asyncio.create_task(long_running_operation())
    
    # 启动另一个计时任务
    try:
        # 最多等待5秒
        await asyncio.sleep(5)
        # 如果到这里，说明long_running_operation还没完成，取消它
        task.cancel()
        print("任务已取消")
    except asyncio.CancelledError:
        pass
        
    try:
        # 等待任务完成或被取消
        await task
    except asyncio.CancelledError:
        print("任务被成功取消")
```

## 5. asyncio实战示例

### 5.1 异步网络请求

使用`aiohttp`库进行并发网络请求：

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    """异步获取URL内容"""
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    """并发获取多个URL的内容"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        # 并发执行所有请求
        return await asyncio.gather(*tasks)

async def main():
    urls = [
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://stackoverflow.com",
        "https://news.ycombinator.com"
    ]
    
    start_time = time.time()
    results = await fetch_all(urls)
    duration = time.time() - start_time
    
    print(f"获取了{len(results)}个网页，总耗时: {duration:.2f}秒")
    for i, result in enumerate(results):
        print(f"URL {urls[i]}: {len(result)} 字符")

# 运行主协程
if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 构建异步Web服务器

使用`aiohttp`构建简单的异步Web服务器：

```python
from aiohttp import web
import asyncio

# 异步处理函数
async def handle_request(request):
    # 模拟异步处理
    await asyncio.sleep(0.1)
    return web.Response(text="Hello, asyncio world!")

# 异步处理数据库查询
async def get_user(request):
    user_id = request.match_info.get('id')
    # 模拟数据库查询
    await asyncio.sleep(0.2)
    return web.json_response({
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    })

# 创建应用
app = web.Application()
app.router.add_get('/', handle_request)
app.router.add_get('/users/{id}', get_user)

if __name__ == '__main__':
    web.run_app(app, port=8080)
```

### 5.3 异步文件操作

使用`aiofiles`进行异步文件操作：

```python
import asyncio
import aiofiles

async def read_file(filename):
    async with aiofiles.open(filename, mode='r') as f:
        return await f.read()

async def write_file(filename, content):
    async with aiofiles.open(filename, mode='w') as f:
        await f.write(content)

async def process_file(input_file, output_file):
    # 读取文件
    content = await read_file(input_file)
    
    # 处理内容
    processed_content = content.upper()
    
    # 写入结果
    await write_file(output_file, processed_content)
    return len(processed_content)

async def main():
    files_to_process = [
        ('input1.txt', 'output1.txt'),
        ('input2.txt', 'output2.txt'),
        ('input3.txt', 'output3.txt')
    ]
    
    tasks = [process_file(input_file, output_file) 
             for input_file, output_file in files_to_process]
    
    results = await asyncio.gather(*tasks)
    print(f"处理了{len(results)}个文件，共{sum(results)}个字符")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.4 综合实例：异步爬虫

实现一个简单的异步爬虫，演示异步编程的完整流程：

```python
import asyncio
import aiohttp
import aiofiles
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class AsyncCrawler:
    def __init__(self, start_url, max_depth=2, max_pages=20):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.session = None
        self.count = 0
    
    async def fetch(self, url, depth):
        """获取并解析URL"""
        if url in self.visited_urls or depth > self.max_depth or self.count >= self.max_pages:
            return
        
        self.visited_urls.add(url)
        self.count += 1
        print(f"正在抓取 ({self.count}/{self.max_pages}): {url}")
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    # 保存网页
                    await self.save_page(url, html)
                    # 提取链接
                    if depth < self.max_depth:
                        links = self.extract_links(url, html)
                        tasks = [self.fetch(link, depth + 1) for link in links]
                        await asyncio.gather(*tasks)
        except Exception as e:
            print(f"抓取 {url} 时出错: {e}")
    
    def extract_links(self, base_url, html):
        """提取页面中的链接"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            absolute_link = urljoin(base_url, link)
            # 只保留同域名链接
            if absolute_link.startswith(self.start_url) and absolute_link not in self.visited_urls:
                links.append(absolute_link)
        return links[:5]  # 每个页面最多处理5个链接，避免过度抓取
    
    async def save_page(self, url, html):
        """保存网页内容"""
        # 创建一个有效的文件名
        filename = re.sub(r'[^a-zA-Z0-9]', '_', url)
        filename = f"pages/{filename[:100]}.html"  # 截断过长的文件名
        
        try:
            async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                await f.write(html)
        except Exception as e:
            print(f"保存 {url} 时出错: {e}")
    
    async def run(self):
        """启动爬虫"""
        self.session = aiohttp.ClientSession()
        try:
            await self.fetch(self.start_url, 0)
        finally:
            await self.session.close()
            print(f"爬虫完成，共抓取 {len(self.visited_urls)} 个页面")

async def main():
    # 创建爬虫实例
    crawler = AsyncCrawler("https://python.org", max_depth=2, max_pages=10)
    # 运行爬虫
    await crawler.run()

if __name__ == "__main__":
    # 确保存在pages目录
    import os
    os.makedirs("pages", exist_ok=True)
    # 启动爬虫
    asyncio.run(main())
```

## 6. 异步编程最佳实践

### 6.1 何时使用异步编程

**适合场景**：

- I/O密集型任务
- 网络请求处理
- 文件操作
- 数据库查询
- 大量并发连接

**不适合场景**：

- CPU密集型计算
- 需要精确实时响应的场景
- 简单线性程序（引入异步可能增加复杂性）

### 6.2 异步代码结构化

**保持协程简洁**：

- 每个协程应该具有单一职责
- 避免过长的协程函数
- 合理拆分复杂逻辑

**结构化异步应用**：

```
project/
├── main.py            # 入口点
├── services/          # 业务逻辑层
│   ├── __init__.py
│   ├── user_service.py
│   └── data_service.py
├── repositories/      # 数据访问层
│   ├── __init__.py
│   └── async_repo.py
└── utils/             # 工具函数
    ├── __init__.py
    └── async_helpers.py
```

**分层设计**：

```python
# 数据访问层
class UserRepository:
    async def get_user(self, user_id):
        # 异步数据库操作
        pass
        
# 业务逻辑层
class UserService:
    def __init__(self, repo):
        self.repo = repo
        
    async def get_user_data(self, user_id):
        user = await self.repo.get_user(user_id)
        # 业务逻辑处理
        return user
        
# API层
async def handle_get_user(request):
    user_id = request.match_info['id']
    service = UserService(UserRepository())
    user = await service.get_user_data(user_id)
    return web.json_response(user)
```

### 6.3 错误处理与资源管理

**异常处理**：

```python
async def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"尝试 {attempt+1}/{max_retries}: 状态码 {response.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                # 指数退避
                await asyncio.sleep(2 ** attempt)
    
    raise Exception(f"在 {max_retries} 次尝试后无法获取 {url}")
```

**资源管理**：

```python
async def process_data():
    # 使用异步上下文管理器管理资源
    async with aiohttp.ClientSession() as session, \
              aiofiles.open("output.txt", "w") as f:
        # 获取数据
        async with session.get("https://example.com/data") as response:
            data = await response.text()
        
        # 处理数据
        processed = data.upper()
        
        # 保存结果
        await f.write(processed)
```

### 6.4 性能考量与优化

**并发控制**：

使用信号量限制并发数：

```python
async def fetch_all_with_limit(urls, limit=10):
    # 创建信号量，限制并发数为10
    semaphore = asyncio.Semaphore(limit)
    
    async def fetch_with_semaphore(url):
        # 获取信号量
        async with semaphore:
            # 在信号量控制下执行请求
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
    
    # 创建所有任务
    tasks = [fetch_with_semaphore(url) for url in urls]
    # 并发执行
    return await asyncio.gather(*tasks)
```

**避免CPU密集型操作阻塞事件循环**：

```python
import concurrent.futures

async def compute_intensive_task(data):
    # 创建进程池
    with concurrent.futures.ProcessPoolExecutor() as pool:
        # 在单独的进程中执行CPU密集型操作
        return await asyncio.get_event_loop().run_in_executor(
            pool, cpu_intensive_function, data
        )

def cpu_intensive_function(data):
    # 复杂计算
    result = 0
    for i in range(10000000):
        result += i * len(data)
    return result
```

## 7. 常见陷阱与解决方案

### 7.1 阻塞事件循环

**问题**：在协程中执行阻塞操作会阻塞整个事件循环

**错误示例**：

```python
async def bad_coroutine():
    # 这会阻塞事件循环！
    time.sleep(5)  # 使用同步sleep而非异步sleep
    return "完成"
```

**解决方案**：

```python
async def good_coroutine():
    # 正确的异步等待
    await asyncio.sleep(5)
    return "完成"

# 对于CPU密集型操作，使用run_in_executor
async def cpu_bound_task():
    loop = asyncio.get_running_loop()
    # 在线程池中执行阻塞函数
    result = await loop.run_in_executor(None, time.sleep, 5)
    return "完成"
```

### 7.2 忘记await协程

**问题**：忘记await协程会导致协程永远不会执行

**错误示例**：

```python
async def main():
    # 以下协程被创建但从未执行
    fetch_data()  # 缺少await！相当于生成器被创建但是从来没有被next!
```

**解决方案**：

```python
async def main():
    # 正确的方式
    await fetch_data()
    
    # 或者创建任务（不立即等待）
    task = asyncio.create_task(fetch_data())
    # ... 做其他事情 ...
    await task  # 稍后等待任务完成
```

### 7.3 不正确的任务管理

**问题**：创建任务后忘记等待或跟踪，可能导致"孤儿任务"

**错误示例**：

```python
async def main():
    # 创建任务但从未等待其完成
    asyncio.create_task(fetch_data())
    print("完成")  # 这会在fetch_data完成前打印
```

**解决方案**：

```python
async def main():
    # 创建任务
    task = asyncio.create_task(fetch_data())
    # 做其他事情...
	# 确保等待任务完成
	await task
	print("任务完成")
	
	# 或者收集所有任务并一并等待
	tasks = [
	    asyncio.create_task(fetch_data("url1")),
	    asyncio.create_task(fetch_data("url2"))
	]
	results = await asyncio.gather(*tasks)
```

### 7.4 异常处理遗漏

**问题**：未捕获的异常可能导致程序崩溃或任务静默失败

**错误示例**：

```python
async def main():
    # 如果task出现异常，整个程序可能会崩溃
    task = asyncio.create_task(risky_operation())
    await task
```

**解决方案**：

```python
async def main():
    task = asyncio.create_task(risky_operation())
    
    try:
        await task
    except Exception as e:
        print(f"任务失败: {e}")
        # 处理错误或记录日志
        
    # 或者使用gather的return_exceptions参数
    results = await asyncio.gather(
        risky_operation1(),
        risky_operation2(),
        return_exceptions=True
    )
    
    # 检查结果中的异常
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i} 失败: {result}")
```

### 7.5 竞态条件

**问题**：多个协程同时访问共享资源可能导致竞态条件

**解决方案**：使用`asyncio.Lock`管理对共享资源的访问

```python
async def safe_counter():
    # 创建锁
    lock = asyncio.Lock()
    counter = 0
    
    async def increment():
        nonlocal counter
        # 获取锁
        async with lock:
            # 临界区 - 只有一个协程可以进入
            current = counter
            await asyncio.sleep(0.01)  # 模拟操作耗时
            counter = current + 1
    
    # 并发运行多个递增操作
    await asyncio.gather(*[increment() for _ in range(100)])
    print(f"最终计数: {counter}")  # 应该是100
```

## 8. 高级应用场景

### 8.1 异步队列与生产者-消费者模式

使用`asyncio.Queue`实现异步生产者-消费者模式：

```python
async def producer(queue, items):
    """生产者函数，向队列中添加项目"""
    for item in items:
        # 生产项目
        await queue.put(item)
        print(f"生产: {item}")
        await asyncio.sleep(0.5)  # 模拟生产时间
    
    # 添加终止标记
    await queue.put(None)

async def consumer(queue, name):
    """消费者函数，从队列中获取并处理项目"""
    while True:
        # 获取项目
        item = await queue.get()
        
        # 检查终止标记
        if item is None:
            print(f"消费者 {name} 完成")
            # 传递终止标记给下一个消费者
            await queue.put(None)
            break
            
        # 处理项目
        print(f"消费者 {name} 处理: {item}")
        await asyncio.sleep(1)  # 模拟处理时间
        
        # 标记任务完成
        queue.task_done()

async def main():
    # 创建队列
    queue = asyncio.Queue(maxsize=3)  # 限制队列大小
    
    # 待处理项目
    items = list(range(10))
    
    # 创建生产者和消费者任务
    producer_task = asyncio.create_task(producer(queue, items))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, f"消费者-{i}"))
        for i in range(3)
    ]
    
    # 等待生产者完成
    await producer_task
    
    # 等待所有消费者完成
    await asyncio.gather(*consumer_tasks)
    
    print("所有任务完成")

# 运行主协程
asyncio.run(main())
```

### 8.2 异步流处理

实现一个异步数据流处理管道：

```python
async def data_source(num_items):
    """数据源，产生异步数据流"""
    for i in range(num_items):
        await asyncio.sleep(0.1)  # 模拟数据生成延迟
        yield i

async def process_data(data_stream):
    """处理数据流中的每一项"""
    async for item in data_stream:
        # 处理项目
        result = item * 2
        yield result

async def filter_data(data_stream):
    """过滤数据流中的项目"""
    async for item in data_stream:
        # 只保留偶数
        if item % 2 == 0:
            yield item

async def data_sink(data_stream):
    """数据接收器，处理最终结果"""
    results = []
    async for item in data_stream:
        print(f"接收: {item}")
        results.append(item)
    return results

async def main():
    # 创建完整的数据处理管道
    source = data_source(10)
    processed = process_data(source)
    filtered = filter_data(processed)
    
    # 运行管道并获取结果
    results = await data_sink(filtered)
    
    print(f"最终结果: {results}")

# 运行主协程
asyncio.run(main())
```

### 8.3 异步上下文管理器实现连接池

实现一个简单的异步数据库连接池：

```python
import asyncio
from contextlib import asynccontextmanager

class AsyncConnectionPool:
    def __init__(self, max_connections=5, connection_timeout=60):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections = []  # 可用连接
        self.semaphore = asyncio.Semaphore(max_connections)  # 限制连接数
    
    async def _create_connection(self):
        """创建一个新的数据库连接"""
        # 模拟连接创建
        await asyncio.sleep(0.1)
        connection = {"id": id({}), "created_at": asyncio.get_event_loop().time()}
        print(f"创建新连接: {connection['id']}")
        return connection
    
    async def _close_connection(self, connection):
        """关闭数据库连接"""
        # 模拟连接关闭
        await asyncio.sleep(0.05)
        print(f"关闭连接: {connection['id']}")
    
    @asynccontextmanager
    async def acquire(self):
        """获取一个连接，使用异步上下文管理器"""
        # 获取信号量，限制最大连接数
        async with self.semaphore:
            # 尝试获取现有连接
            connection = None
            if self.connections:
                connection = self.connections.pop()
                print(f"复用连接: {connection['id']}")
            
            # 如果没有可用连接，创建新连接
            if connection is None:
                connection = await self._create_connection()
            
            try:
                # 提供连接给调用者
                yield connection
            finally:
                # 检查连接是否过期
                now = asyncio.get_event_loop().time()
                if now - connection["created_at"] > self.connection_timeout:
                    # 连接过期，关闭它
                    await self._close_connection(connection)
                else:
                    # 连接仍然有效，放回池中
                    print(f"归还连接: {connection['id']}")
                    self.connections.append(connection)
    
    async def close(self):
        """关闭所有连接并清理资源"""
        while self.connections:
            connection = self.connections.pop()
            await self._close_connection(connection)

async def worker(pool, worker_id):
    """模拟工作负载"""
    async with pool.acquire() as connection:
        print(f"工作者 {worker_id} 使用连接 {connection['id']}")
        # 模拟数据库操作
        await asyncio.sleep(0.5)
        print(f"工作者 {worker_id} 完成")

async def main():
    # 创建连接池
    pool = AsyncConnectionPool(max_connections=3)
    
    try:
        # 创建多个工作任务
        tasks = [worker(pool, i) for i in range(10)]
        await asyncio.gather(*tasks)
    finally:
        # 关闭连接池
        await pool.close()

# 运行主协程
asyncio.run(main())
```

### 8.4 实现自定义异步迭代器

创建一个支持异步迭代的分页数据获取器：

```python
class AsyncPaginator:
    """异步分页迭代器，用于获取分页数据"""
    
    def __init__(self, fetch_func, start_page=1, items_per_page=10):
        self.fetch_func = fetch_func  # 获取页面数据的异步函数
        self.current_page = start_page
        self.items_per_page = items_per_page
        self.has_more = True
    
    def __aiter__(self):
        """返回异步迭代器"""
        return self
    
    async def __anext__(self):
        """获取下一页数据"""
        if not self.has_more:
            # 没有更多数据，停止迭代
            raise StopAsyncIteration
        
        # 获取当前页数据
        data = await self.fetch_func(self.current_page, self.items_per_page)
        
        # 检查是否有更多数据
        if not data or len(data) < self.items_per_page:
            self.has_more = False
        
        # 更新页码
        self.current_page += 1
        
        return data

# 示例用法
async def fetch_page_data(page, limit):
    """模拟API调用获取分页数据"""
    await asyncio.sleep(0.2)  # 模拟网络延迟
    
    # 模拟有50条数据
    total_items = 50
    start_idx = (page - 1) * limit
    
    if start_idx >= total_items:
        return []
    
    end_idx = min(start_idx + limit, total_items)
    return [f"项目-{i}" for i in range(start_idx, end_idx)]

async def main():
    # 创建分页迭代器
    paginator = AsyncPaginator(fetch_page_data, start_page=1, items_per_page=10)
    
    # 使用异步迭代处理所有页面
    page_num = 1
    async for page_items in paginator:
        print(f"页面 {page_num}: 获取到 {len(page_items)} 个项目")
        for item in page_items:
            print(f"  - {item}")
        page_num += 1
    
    print("所有数据已处理完毕")

# 运行主协程
asyncio.run(main())
```

## 9. 总结与资源

### 9.1 核心概念总结

- **协程(Coroutine)**：使用`async def`定义的函数，可以在执行过程中暂停和恢复
- **事件循环(Event Loop)**：协调协程执行的调度器，决定何时运行哪个协程
- **任务(Task)**：协程的包装器，添加了状态跟踪和结果存储
- **Future**：表示异步操作的最终结果
- **异步上下文管理器**：使用`async with`进行异步资源管理
- **异步迭代器**：使用`async for`进行异步迭代

### 9.2 关键函数和方法

- **基本函数**：
    
    - `asyncio.run(coro)` - 运行协程并返回结果
    - `asyncio.create_task(coro)` - 将协程包装为任务
    - `asyncio.gather(*coros)` - 并发运行多个协程
    - `asyncio.wait(tasks)` - 等待多个任务完成，提供更多控制
    - `asyncio.wait_for(coro, timeout)` - 设置超时等待协程
    
- **同步与通信**：
    
    - `asyncio.Lock()` - 创建异步锁
    - `asyncio.Event()` - 创建异步事件
    - `asyncio.Condition()` - 创建异步条件变量
    - `asyncio.Semaphore()` - 创建异步信号量
    - `asyncio.Queue()` - 创建异步队列
    
- **运行阻塞代码**：
    
    - `loop.run_in_executor(executor, func, *args)` - 在执行器中运行阻塞函数

### 9.3 设计模式与最佳实践

- **使用`asyncio.run()`作为入口点**（Python 3.7+）
- **使用任务管理协程**，而不是直接调用协程
- **正确处理异常**，避免未捕获的异常导致程序崩溃
- **适当限制并发数量**，使用信号量控制资源使用
- **避免阻塞事件循环**，将CPU密集型任务放入执行器
- **使用异步上下文管理器管理资源**
- **分层设计异步应用**，保持代码清晰和可维护

### 9.4 进阶学习资源

- **官方文档**：
    
    - [asyncio — Python 官方文档](https://docs.python.org/3/library/asyncio.html)
    
- **相关库**：
    
    - [aiohttp](https://docs.aiohttp.org/) - 异步HTTP客户端/服务器
    - [aiofiles](https://github.com/Tinche/aiofiles) - 异步文件I/O
    - [aiomysql](https://aiomysql.readthedocs.io/) - MySQL的异步客户端
    - [asyncpg](https://magicstack.github.io/asyncpg/) - PostgreSQL的异步客户端
    - [motor](https://motor.readthedocs.io/) - MongoDB的异步客户端
    
- **设计模式**：
    
    - [异步设计模式](https://www.roguelynn.com/words/asyncio-we-did-it-wrong/)
    - [使用Python进行生产级异步编程](https://github.com/yeraydiazdiaz/asyncio-coroutine-patterns)

### 9.5 持续学习建议

- **从简单示例开始**，逐步构建复杂应用
- **研读优质库源码**，学习最佳实践和设计模式
- **定期跟进Python更新**，了解asyncio的新特性和改进
- **参与社区讨论**，分享经验和学习他人解决方案
- **在实际项目中应用**，通过实践巩固理论知识

记住，asyncio是一个强大但复杂的工具。掌握它需要时间和实践，但一旦熟练，将极大提高处理I/O密集型任务的效率和性能。
