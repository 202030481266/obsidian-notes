#typescript #Claude 

# `??`空值合并符号，keyof，...

1. 空值合并运算符 `??`

```typescript
// 场景1：设置默认值
function greet(name: string | null | undefined) {
    // 如果name是null或undefined，使用'Guest'作为默认值
    console.log(`Hello, ${name ?? 'Guest'}!`);
}

greet('Alice');   // 输出：Hello, Alice!
greet(null);      // 输出：Hello, Guest!
greet(undefined); // 输出：Hello, Guest!

// 场景2：配置参数的默认值
interface Config {
    timeout?: number;
    retries?: number;
}

function createRequest(config: Config) {
    // 如果没有设置timeout，默认为3000
    const timeout = config.timeout ?? 3000;
    // 如果没有设置retries，默认为3
    const retries = config.retries ?? 3;

    console.log(`Timeout: ${timeout}, Retries: ${retries}`);
}

createRequest({});  // 输出：Timeout: 3000, Retries: 3
createRequest({ timeout: 5000 });  // 输出：Timeout: 5000, Retries: 3
```

2. `keyof` 运算符

```typescript
// 场景1：创建安全的对象属性访问器
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

interface Person {
    name: string;
    age: number;
    email: string;
}

const person: Person = {
    name: 'John',
    age: 30,
    email: 'john@example.com'
};

// 只能传入Person的有效键
const name = getProperty(person, 'name');     // 正确
const age = getProperty(person, 'age');       // 正确
// const invalid = getProperty(person, 'phone');  // 类型错误

// 场景2：约束对象的键
function createMapper<T>() {
    return <K extends keyof T>(key: K) => {
        return (obj: T): T[K] => obj[key];
    };
}

interface User {
    id: number;
    name: string;
    email: string;
}

const getUserName = createMapper<User>()('name');
const user: User = { id: 1, name: 'Alice', email: 'alice@example.com' };
console.log(getUserName(user));  // 输出：Alice
```

3. 展开运算符 `...`

```typescript
// 场景1：合并对象
interface BaseConfig {
    timeout: number;
    logging: boolean;
}

interface ExtendedConfig extends BaseConfig {
    retries?: number;
}

const baseConfig: BaseConfig = {
    timeout: 3000,
    logging: true
};

// 合并并覆盖配置
const customConfig: ExtendedConfig = {
    ...baseConfig,
    retries: 3  // 添加新属性
};

// 场景2：函数参数收集
function logPersonDetails(name: string, ...skills: string[]) {
    console.log(`Name: ${name}`);
    console.log('Skills:', skills);
}

logPersonDetails('John', 'JavaScript', 'TypeScript', 'React');
// 输出：
// Name: John
// Skills: ['JavaScript', 'TypeScript', 'React']

// 场景3：数组解构和合并
const fruits = ['Apple', 'Banana'];
const moreFruits = ['Cherry', 'Date'];

// 合并数组
const allFruits = [...fruits, ...moreFruits];
console.log(allFruits);  // ['Apple', 'Banana', 'Cherry', 'Date']

// 解构数组
const [first, ...rest] = allFruits;
console.log(first);  // 'Apple'
console.log(rest);   // ['Banana', 'Cherry', 'Date']
```

这些示例展示了这三个运算符在实际开发中的常见用法：

- `??` 主要用于提供安全的默认值
- `keyof` 用于创建类型安全的属性访问和约束
- `...` 用于对象合并、数组解构和函数参数收集
