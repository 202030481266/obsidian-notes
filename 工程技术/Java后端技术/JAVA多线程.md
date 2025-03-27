# 推荐视频

[【狂神说Java】多线程详解](https://www.bilibili.com/video/BV1V4411p7EF)

# 线程创建

## 继承Thread类

Thread是Java中的一个基本的类，继承了老祖宗`java.lang.Object`，想要创建一个线程，那么一个常用的方法就是继承这个类，定义自己的`run`方法。

```java
package example;

public class JavaThreadExample extends Thread {
    // 必须实现run方法
    public void run() {
        for (int i = 0; i < 10000; ++i) {
            if (i % 1000 == 0) {
                System.out.printf("JavaThreadExample output %d\n", i);
            }
        }
    }
}
```

可以在main函数中测试一下：

```java
import example.*;

public class Main {
    public static void main(String[] args) {
        JavaThreadExample thread = new JavaThreadExample();

        thread.start(); // 注意这里必须要调用start方法，run方法是同步执行

        try {
            // 睡眠2秒（2000毫秒）
            Thread.sleep(15);
        } catch (InterruptedException e) {
            // 处理中断异常
            e.printStackTrace();
        }

        System.out.println("Main thread is done!");
    }
}
```

可以观察到输出中输出了`JavaThreadExample`类的第一条数据，然后输出了`Main thread is done!`，然后继续输出`JavaThreadExample`的内容，交替输出实现了真正意义上的多线程同时执行。

# 实现Runable接口

