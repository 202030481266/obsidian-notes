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

实际上，线程的创建可以接受一个`Runable`对象并且调用对应的构造函数：

```java
public Thread(Runnable task) {
	this(null, null, 0, task, 0);
}
```

因此可以实现自己的`Runable`接口来构建一个线程：

```java
package example;

public class JavaRunableExample implements Runnable {
    public void run() {
        for (int i = 0; i < 10000; ++i) {
            if (i % 1000 == 0) {
                System.out.printf("JavaRunableExample output %d\n", i);
            }
        }
    }
}
```

# Thread vs Runable

在 Java 中，创建线程时推荐使用实现 `Runnable` 接口而不是继承 `Thread` 类，主要有以下几个原因：

1. **灵活性更高（避免单继承限制）**  
   Java 是单继承的语言，一个类只能继承一个父类。如果直接继承 `Thread` 类，这个类就无法再继承其他类。而实现 `Runnable` 接口只是实现一个接口，类仍然可以继承其他类或实现其他接口，保持了设计的灵活性。

2. **资源共享更方便**  
   使用 `Runnable` 接口可以将任务（`Runnable` 对象）与线程（`Thread` 对象）分离。一个 `Runnable` 对象可以被多个 `Thread` 对象共享和复用，这样更适合需要多个线程执行相同任务的场景。而继承 `Thread` 类时，每个线程实例都是独立的，任务逻辑和线程本身绑定在一起，不利于共享。

3. **面向接口编程的设计原则**  
   Java 提倡面向接口编程，使用 `Runnable` 接口更符合这种设计理念。`Runnable` 定义了一个清晰的任务接口（`run()` 方法），将任务逻辑与线程的运行机制解耦，代码结构更清晰，维护性更好。

4. **线程池的兼容性**  
   在现代 Java 开发中，线程池（如 `ExecutorService`）是管理线程的常用方式。线程池接受 `Runnable` 或 `Callable` 任务，而无法直接使用继承 `Thread` 类的对象。将任务定义为 `Runnable` 可以无缝适配线程池，提高性能和资源利用率。

## 示例对比

- **继承 Thread 类：**
- 
  ```java
  class MyThread extends Thread {
      public void run() {
          System.out.println("Thread running");
      }
  }

  public class Main {
      public static void main(String[] args) {
          MyThread thread = new MyThread();
          thread.start();
      }
  }
  ```
  
这里 `MyThread` 继承了 `Thread`，无法再继承其他类。

- **实现 Runnable 接口：**
- 
  ```java
  class MyRunnable implements Runnable {
      public void run() {
          System.out.println("Runnable running");
      }
  }

  public class Main {
      public static void main(String[] args) {
          Thread thread = new Thread(new MyRunnable());
          thread.start();
      }
  }
  ```
  
这里 `MyRunnable` 只是实现了一个接口，可以被多个线程复用，也可以继承其他类。

## 总结

实现 `Runnable` 接口相比继承 `Thread` 类更灵活、可复用性更强，并且符合现代 Java 编程的最佳实践。因此，除非有特殊需求（如需要重写 `Thread` 类的其他方法），通常推荐使用 `Runnable` 来创建线程任务。