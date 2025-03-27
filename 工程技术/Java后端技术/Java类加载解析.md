### Java 类加载过程总结

Java 类的加载过程由 JVM 的类加载器完成，分为三个主要阶段：加载 (Loading)、链接 (Linking) 和初始化 (Initialization)。每个阶段涉及 JVM 的不同内存区域，包括方法区 (Method Area)、堆 (Heap) 和栈 (Stack)。以下是详细总结：

#### 1. 加载 (Loading)

- **作用**: 将类的 `.class` 文件加载到内存，生成代表类的 Class 对象。
- **过程**:  
  - 类加载器根据全限定名（如 com.example.MyClass）找到 `.class` 文件。
  - 字节码加载到方法区，Class 对象生成在堆中。
- **内存区域**:  
  - 方法区: 存储类的元数据（字段、方法、字节码等）。
  - 堆: 存储 Class 对象实例。

#### 2. 链接 (Linking)

链接分为三个子阶段：
- **(1) 验证 (Verification)**  
  - **作用**: 检查字节码的合法性（如魔数 CAFEBABE）。
  - **内存区域**: 方法区（验证对象是方法区中的字节码），栈（执行验证逻辑）。
- **(2) 准备 (Preparation)**  
  - **作用**: 为静态变量分配内存，设置默认值（例如 static int a 初始化为 0）。
  - **内存区域**: 方法区（静态变量存储在此）。
- **(3) 解析 (Resolution)**  
  - **作用**: 将符号引用（如 MyClass.method()）转为直接引用（内存地址）。
  - **内存区域**: 方法区（解析后的引用指向方法区或堆）。

#### 3. 初始化 (Initialization)

- **作用**: 执行类的静态初始化代码（如 static int a = 5 和静态代码块）。
- **过程**:  
  - 执行由编译器生成的 `<clinit>` 方法。
  - 静态变量赋值和静态块按代码顺序执行。
- **内存区域**:  
  - 方法区: 存储静态变量的最终值。
  - 栈: 执行 `<clinit>` 方法的栈帧。
  - 堆: 若静态块创建对象（如 static Object obj = new Object()），对象分配在堆中。

#### 4. 对象实例化（加载完成后）

- **作用**: 使用 new 创建对象实例。
- **内存区域**:  
  - 堆: 存储对象实例。
  - 栈: 构造方法执行时的局部变量和栈帧。
  - 方法区: 提供类的元信息（如字段布局）。

#### 触发时机

类加载在以下情况触发：
- 使用 new 创建对象。
- 访问 static 变量或方法。
- 使用反射（如 Class.forName()）。
- 初始化子类时，先加载父类。

---

### 示例总结

以代码为例：

```java
class Parent {
    static int a = 5;
    static {
        System.out.println("Parent static block, a = " + a);
    }
}

class Child extends Parent {
    static int b = 10;
    static {
        System.out.println("Child static block, b = " + b);
    }

    public static void main(String[] args) {
        System.out.println("Main start");
        Child child = new Child();
    }
}
```

- **加载**: Parent 和 Child 的字节码加载到方法区，Class 对象在堆中。
- **链接**:  
  - 验证 Parent.class 和 Child.class。
  - 准备时，Parent.a 和 Child.b 在方法区设为 0。
  - 解析继承关系。
- **初始化**:  
  - Parent.a 赋值为 5，执行静态块（栈参与）。
  - Child.b 赋值为 10，执行静态块。
- **执行 main**: 输出 "Main start"，new Child() 在堆中创建实例。

输出：

```
Parent static block, a = 5
Child static block, b = 10
Main start
```

---

### 总结表格
| 阶段                     | 方法区 (Method Area)                       | 堆 (Heap)                        | 栈 (Stack)                |
|--------------------------|--------------------------------------------|----------------------------------|---------------------------|
| 加载 (Loading)           | 存储字节码和类元数据                       | 生成 Class 对象                  | 不参与                    |
| 验证 (Verification)      | 检查字节码                                 | 不参与                           | 执行验证逻辑              |
| 准备 (Preparation)       | 分配 static 变量内存，设默认值             | 不参与                           | 不参与                    |
| 解析 (Resolution)        | 符号引用转为直接引用                       | 不参与                           | 不参与                    |
| 初始化 (Initialization)  | static 变量赋值，存储静态数据              | 创建 static 对象（如有）         | 执行 `<clinit>` 栈帧      |
| 对象实例化               | 提供类元信息                               | 分配对象实例                     | 存储局部变量和栈帧        |
