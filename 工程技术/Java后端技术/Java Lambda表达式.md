在 Java 中，Lambda 表达式（Lambda Expression）是 Java 8 引入的一项重要特性，它极大地增强了语言的函数式编程能力。Lambda 表达式提供了一种简洁的方式来表示匿名函数（即没有名称的函数），通常用于实现函数式接口（Functional Interface）。下面我将详细讲解 Lambda 表达式的概念、语法、使用场景以及相关细节。

---

### 1. **什么是 Lambda 表达式？**

Lambda 表达式本质上是一个匿名方法，它允许你将行为（代码）作为数据传递。它通常用来替代匿名内部类，尤其是在只需要实现一个抽象方法的情况下。Lambda 表达式的引入与 Java 的函数式编程支持密切相关，配合 `java.util.function` 包中的函数式接口和 Stream API 使用效果尤为强大。

- **核心思想**：用简洁的语法表示“输入参数 -> 输出结果”的映射。
- **依赖条件**：Lambda 表达式必须与函数式接口绑定，函数式接口是指只包含一个抽象方法的接口（可以有默认方法或静态方法）。

---

### 2. **Lambda 表达式的语法**

Lambda 表达式的基本语法由三部分组成：

```
(参数列表) -> { 表达式或语句块 }
```

- **参数列表**：可以为空，也可以包含多个参数。如果只有一个参数且类型可以推断，括号可以省略。
- **箭头（->）**：分隔参数和主体。
- **主体**：可以是一个表达式（无需大括号和 `return`），也可以是一个语句块（需要大括号和显式的 `return`）。

#### 示例：
1. 无参数：
   ```java
   () -> System.out.println("Hello, Lambda!")
   ```
2. 单参数：
   ```java
   x -> x * 2
   ```
3. 多参数：
   ```java
   (x, y) -> x + y
   ```
4. 带语句块：
   ```java
   (x, y) -> {
       System.out.println("Calculating...");
       return x + y;
   }
   ```

---

### 3. **函数式接口**

Lambda 表达式必须绑定到一个函数式接口。常见的函数式接口包括：

- `Runnable`：无参数，无返回值，`void run()`
- `Comparator<T>`：两个参数，返回 int，`int compare(T o1, T o2)`
- `Function<T, R>`：一个参数，返回结果，`R apply(T t)`
- `Predicate<T>`：一个参数，返回 boolean，`boolean test(T t)`
- `Consumer<T>`：一个参数，无返回值，`void accept(T t)`
- `Supplier<T>`：无参数，返回结果，`T get()`

Java 8 在 `java.util.function` 包中提供了大量预定义的函数式接口，方便开发者使用。

#### 示例：使用 Lambda 替换匿名内部类
传统匿名内部类：
```java
Runnable r = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running...");
    }
};
```
使用 Lambda 表达式：
```java
Runnable r = () -> System.out.println("Running...");
```

---

### 4. **Lambda 表达式的特点**

1. **简洁性**：减少了冗长的匿名内部类代码。
2. **类型推断**：参数类型通常可以省略，由编译器根据上下文推断。
   - 例如：`(Integer x, Integer y) -> x + y` 可以简化为 `(x, y) -> x + y`。
3. **闭包特性**：Lambda 表达式可以捕获外部变量（称为“捕获变量”），但这些变量必须是**有效最终变量**（effectively final），即定义后不再被修改。
   - 示例：
     ```java
     int num = 10;
     Runnable r = () -> System.out.println(num); // 合法，因为 num 未被修改
     ```

---

### 5. **使用场景**
Lambda 表达式在以下场景中特别有用：
#### (1) **集合操作与 Stream API**
配合 Stream API，Lambda 表达式可以简化集合的处理。
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
numbers.forEach(n -> System.out.println(n)); // 打印每个元素
int sum = numbers.stream().map(x -> x * 2).reduce(0, Integer::sum); // 每个元素乘 2 后求和
```

#### (2) **事件处理**
在 GUI 编程中，Lambda 表达式可以简化事件监听器的定义。
```java
button.addActionListener(e -> System.out.println("Button clicked!"));
```

#### (3) **多线程**
简化线程的创建。
```java
new Thread(() -> System.out.println("Thread running")).start();
```

#### (4) **自定义函数式逻辑**
通过 Lambda 传递行为。
```java
public static void process(Consumer<String> consumer) {
    consumer.accept("Hello");
}
process(s -> System.out.println(s + " World")); // 输出 "Hello World"
```

---

### 6. **方法引用（Method Reference）**

方法引用是 Lambda 表达式的简化和增强形式，用 `::` 表示。它可以进一步简化代码，但本质上仍是 Lambda 表达式的一种语法糖。  
- **类型**：
  1. 静态方法引用：`ClassName::staticMethod`
     - 示例：`Integer::parseInt`
  2. 实例方法引用：`object::instanceMethod`
     - 示例：`System.out::println`
  3. 特定类的实例方法引用：`ClassName::instanceMethod`
     - 示例：`String::toUpperCase`
  4. 构造器引用：`ClassName::new`
     - 示例：`ArrayList::new`

#### 示例：
```java
List<String> list = Arrays.asList("a", "b", "c");
list.forEach(System.out::println) // 等价于 s -> System.out.println(s)
```

---

### 7. **注意事项**
1. **变量捕获限制**：
   - Lambda 表达式只能引用外部的“有效最终变量”。如果变量在 Lambda 定义后被修改，会导致编译错误。
   - 示例：
     ```java
     int num = 10;
     num = 20; // 修改 num
     Runnable r = () -> System.out.println(num); // 编译错误
     ```

2. **this 引用**：
   - 在 Lambda 表达式中，`this` 指向的是定义 Lambda 的外部类的实例，而不是 Lambda 本身。

3. **性能**：
   - **Lambda 表达式在底层会被编译为匿名类的实例**，但 JVM 会通过 `invokedynamic` 指令优化性能，避免每次都创建新类。

---

### 8. **总结**

Lambda 表达式是 Java 函数式编程的基石，它通过简洁的语法和强大的表达能力简化了代码。它的核心是与函数式接口结合使用，广泛应用于集合操作、多线程、事件处理等领域。配合方法引用和 Stream API，Lambda 表达式让 Java 代码更现代化、更优雅。
