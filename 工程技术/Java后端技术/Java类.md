在 Java 中，类的类型可以根据其定义方式、作用域和用途进行分类。除了你提到的静态内部类、普通类和匿名内部类之外，Java 还有其他类型的类。以下是 Java 中常见的类类型及其简要说明：

### 1. **普通类 (Regular Class)**  

这是最基本的类类型，使用 `class` 关键字定义，通常独立存在于一个 `.java` 文件中。 
- 示例：
  ```java
  public class MyClass {
      // 类的实现
  }
  ```

### 2. **静态内部类 (Static Nested Class)**  

静态内部类是定义在另一个类内部的类，使用 `static` 关键字修饰。它不需要外部类的实例即可访问或实例化。  
- 示例：
  ```java
  public class OuterClass {
      static class StaticNestedClass {
          // 静态内部类的实现
      }
  }
  ```
- 特点：与外部类的实例无关，类似于外部类的静态成员。

### 3. **匿名内部类 (Anonymous Inner Class)**  

匿名内部类是没有名字的类，通常用于一次性实现接口或继承抽象类。它在定义的同时被实例化。  
- 示例：
  ```java
  Runnable r = new Runnable() {
      public void run() {
          System.out.println("Running");
      }
  };
  ```
- 特点：常用于事件监听器或简短的实现逻辑。

### 4. **成员内部类 (Member Inner Class)**  

成员内部类是定义在另一个类内部的非静态类。它与外部类的实例绑定，需要外部类的实例来创建。  
- 示例：
  ```java
  public class OuterClass {
      class InnerClass {
          // 内部类的实现
      }
  }
  ```
- 使用方式：
  ```java
  OuterClass outer = new OuterClass();
  OuterClass.InnerClass inner = outer.new InnerClass();
  ```
- 特点：可以访问外部类的所有成员（包括私有成员）。

### 5. **局部内部类 (Local Inner Class)**  

局部内部类是定义在方法或代码块内部的类，其作用域仅限于定义它的块。  
- 示例：
  ```java
  public class OuterClass {
      public void method() {
          class LocalClass {
              // 局部类的实现
          }
          LocalClass local = new LocalClass();
      }
  }
  ```
- 特点：只能在定义它的方法或块中使用。

### 6. **抽象类 (Abstract Class)**  
抽象类使用 `abstract` 关键字定义，不能直接实例化，通常用于定义模板或基类。  
- 示例：
  ```java
  public abstract class AbstractClass {
      abstract void doSomething();
  }
  ```

### 7. **接口 (Interface)**  

严格来说，接口不是类，但它是 Java 类型系统的重要组成部分。从 Java 8 开始，接口可以包含默认方法和静态方法。  
- 示例：
  ```java
  public interface MyInterface {
      void myMethod();
      default void defaultMethod() {
          System.out.println("Default");
      }
  }
  ```

### 8. **枚举类 (Enum Class)**  

枚举类使用 `enum` 关键字定义，是一种特殊的类，用于表示一组固定的常量。  
- 示例：
  ```java
  public enum Day {
      MONDAY, TUESDAY, WEDNESDAY
  }
  ```
- 特点：隐式继承自 `java.lang.Enum`，可以包含方法和字段。

### 9. **记录类 (Record Class)**  

从 Java 14 引入（Java 16 正式版），记录类是一种特殊的类，用于简化不可变数据载体的定义。  
- 示例：
  ```java
  public record Person(String name, int age) {}
  ```
- 特点：自动生成构造器、getter、`equals()`、`hashCode()` 和 `toString()` 方法。

### 10. **嵌套类 (Nested Class)**  

嵌套类是一个广义术语，包括静态内部类和非静态内部类（成员内部类、局部内部类、匿名内部类）。在 Java 中，所有定义在另一个类内部的类都属于嵌套类。

### 总结  

Java 中的类类型可以分为以下几类：  
- **顶级类**：普通类、抽象类、枚举类、记录类  
- **嵌套类**：  
  - 静态内部类  
  - 非静态内部类（成员内部类、局部内部类、匿名内部类）  

此外，接口虽然不是传统意义上的类，但在类型系统中也扮演重要角色。根据具体需求，Java 提供了丰富的类定义方式来满足不同的编程场景。