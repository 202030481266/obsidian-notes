#React #前端 #Gemini

# 概念

作为初学者，我们可以把 Hooks 理解为**让你的函数组件（Functional Component）拥有“超能力”的一系列特殊函数**。

在过去，如果你想让一个组件拥有自己的状态（state）或者使用生命周期方法（比如组件加载后执行某些操作），你必须使用类组件（Class Component）。函数组件则很简单，只能接收 props 并渲染界面，被称为“无状态组件”。

Hooks 的出现彻底改变了这一点。它允许我们**在不编写类的情况下，使用 state 和其他 React 特性**。这让函数组件变得和类组件一样强大，但代码却更简洁、更易于理解。

---

### **为什么要使用 Hooks？**

简单来说，Hooks 解决了 React 开发中的几个痛点：

1. **告别复杂的 `this`**：在类组件中，你需要频繁地和 `this` 打交道，比如在构造函数中绑定事件处理函数，初学者很容易在这里犯错。函数组件和 Hooks 则完全不需要关心 `this` 的指向问题。
    
2. **更好地重用状态逻辑**：在 Hooks 出现之前，要跨组件复用一些有状态的逻辑（比如订阅数据、跟踪窗口大小等）非常麻烦，通常需要用到高阶组件（HOC）或 Render Props 等复杂模式。Hooks 允许你将这些逻辑提取到可复用的函数中（我们称之为“自定义 Hook”），让代码更清晰、更易维护。
    
3. **让组件逻辑更清晰**：在类组件中，一个功能相关的逻辑常常被拆分到不同的生命周期方法中（比如，在 `componentDidMount` 中订阅，在 `componentWillUnmount` 中取消订阅），而一些不相关逻辑却可能被堆在同一个生命周期方法里（比如在 `componentDidMount` 中同时获取数据和添加事件监听）。Hooks（尤其是 `useEffect`）可以让你根据功能把相关的代码组织在一起，提升代码的可读性。
    

---

### **使用 Hooks 的两大黄金法则**

在使用 Hooks 之前，你必须了解并遵守两条规则，这能确保 Hooks 正常工作：

1. **只能在函数组件的顶层调用 Hooks**：
    
    - **不要**在循环、条件判断或嵌套函数中调用 Hook。
        
    - **要**确保 Hooks 在每次组件渲染时都以完全相同的顺序被调用。React 依赖这个调用顺序来正确地在多次调用之间保持 Hook 的状态。
        
2. **只能在 React 函数中调用 Hooks**：
    
    - 你可以在 React 的函数组件中调用 Hooks。
        
    - 你可以在自定义 Hook 中调用其他 Hooks（自定义 Hook 我们稍后会提到）。
        
    - **不要**在普通的 JavaScript 函数中调用 Hook。
        

别担心记不住，现在流行的代码检查工具（ESLint）通常会自动帮你检查并提示这些规则。

---

### **最常用的几个 Hooks 详解（附代码示例）**

接下来，我们来学习几个最核心、最常用的 Hooks，这是你日常开发中最常打交道的“朋友”。

#### 1. `useState`：让组件拥有自己的状态

这是最基础也是最重要的 Hook。它允许你的函数组件拥有一个可以随时间变化的值（即 state）。

- **作用**：在组件中添加一个状态变量。
    
- **如何使用**：
    
    - `useState` 接收一个参数作为状态的**初始值**。
        
    - 它返回一个包含两个元素的数组：{当前状态值, 更新该状态的函数}。
        

**示例：一个简单的计数器**

JavaScript

```
import React, { useState } from 'react';

function Counter() {
  // 声明一个新的 state 变量，我们把它叫做 "count"
  // 0 是 count 的初始值
  // setCount 是我们用来更新 count 值的函数
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>你点击了 {count} 次</p>
      {/* 当用户点击按钮时，我们调用 setCount 来增加 count 的值 */}
      <button onClick={() => setCount(count + 1)}>
        点我增加
      </button>
      <button onClick={() => setCount(count - 1)}>
        点我减少
      </button>
    </div>
  );
}

export default Counter;
```

在这个例子中，每当 `setCount` 被调用，React 就会重新渲染 `Counter` 组件，并且 `count` 会使用最新的值。

#### 2. `useEffect`：处理副作用

“副作用”（Side Effects）是指在组件渲染过程中，需要与外部系统交互的操作。常见的副作用包括：

- 数据获取 (Fetching data from an API)
    
- 设置订阅 (Setting up a subscription)
    
- 手动更改 DOM
    
- 设置定时器 (`setTimeout`, `setInterval`)
    

`useEffect` 就像是类组件中 `componentDidMount`, `componentDidUpdate`, 和 `componentWillUnmount` 这三个生命周期函数的组合。

- **作用**：在函数组件中执行副作用操作。
    
- **如何使用**：
    
    - 它接收两个参数：一个**回调函数**和一个可选的**依赖项数组**。
        
    - `useEffect(() => { /* ... */ }, [dependencies]);`
        

**依赖项数组 (`[dependencies]`) 的作用至关重要：**

- **不提供数组**：`useEffect` 在**每次**组件渲染后都会执行。
    
- **提供一个空数组 `[]`**：`useEffect` 只在组件**首次**渲染后执行一次（类似于 `componentDidMount`）。
    
- **提供包含变量的数组 `[prop, state]`**：`useEffect` 会在首次渲染后执行，并且只有当数组中的**任何一个值发生变化**时，它才会再次执行。
    

**示例1：组件加载后修改文档标题**

JavaScript

```
import React, { useState, useEffect } from 'react';

function DocumentTitleChanger() {
  const [count, setCount] = useState(0);

  // useEffect 会在组件首次渲染以及 count 状态变化后执行
  useEffect(() => {
    // 这个函数就是副作用
    document.title = `你点击了 ${count} 次`;
    console.log('Title updated!');
  }, [count]); // 依赖项是 count

  return (
    <div>
      <p>你点击了 {count} 次 (请看浏览器标签页的标题)</p>
      <button onClick={() => setCount(count + 1)}>
        点我
      </button>
    </div>
  );
}

export default DocumentTitleChanger;
```

**示例2：组件加载后获取数据**

JavaScript

```
import React, { useState, useEffect } from 'react';

function UserData() {
  const [user, setUser] = useState(null);

  // 使用一个空数组，确保这个 effect 只运行一次
  useEffect(() => {
    fetch('https://jsonplaceholder.typicode.com/users/1')
      .then(response => response.json())
      .then(data => setUser(data));
  }, []); // 空数组意味着“只在挂载时运行”

  if (!user) {
    return <div>正在加载数据...</div>;
  }

  return (
    <div>
      <h1>用户名: {user.name}</h1>
      <p>邮箱: {user.email}</p>
    </div>
  );
}

export default UserData;
```

#### 3. `useContext`：跨层级共享数据

在 React 应用中，有时你需要将数据（例如：UI 主题、用户登录信息）传递给很多层级的子组件。如果一层一层地通过 props 传递，会非常繁琐，这就是所谓的“属性钻探”（Prop Drilling）。`useContext` 就是为了解决这个问题而生的。

- **作用**：让你能够不通过 props，直接在组件树的深层级中订阅和读取 React 的 context。
    

**示例：全局主题切换**

**1. 创建一个 Context (e.g., `ThemeContext.js`)**

JavaScript

```
import React from 'react';

// 创建一个 Context 对象
export const ThemeContext = React.createContext('light'); // 'light' 是默认值
```

**2. 在顶层组件使用 Provider 来提供 value**

JavaScript

```
import React, { useState } from 'react';
import { ThemeContext } from './ThemeContext';
import Toolbar from './Toolbar';

function App() {
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // 使用 ThemeContext.Provider 将当前的 theme 值传递下去
  return (
    <ThemeContext.Provider value={theme}>
      <div>
        <button onClick={toggleTheme}>切换主题</button>
        <Toolbar />
      </div>
    </ThemeContext.Provider>
  );
}

export default App;
```

**3. 在任何子组件中使用 `useContext` 来消费 value**

JavaScript

```
import React, { useContext } from 'react';
import { ThemeContext } from './ThemeContext';

function Toolbar() {
  // useContext 接收 Context 对象，并返回当前的 context 值
  const theme = useContext(ThemeContext);

  const style = {
    background: theme === 'dark' ? '#333' : '#FFF',
    color: theme === 'dark' ? '#FFF' : '#333',
    padding: '20px',
    border: '1px solid #ccc'
  };

  return (
    <div style={style}>
      当前主题是: {theme}
    </div>
  );
}

export default Toolbar;
```

在这个例子中，`Toolbar` 组件不需要从 `App` 组件接收任何 props，就能直接获取到 `theme` 的值。

#### 4. `useRef`：访问 DOM 或存储可变值

`useRef` 有两个主要的用途：

1. **访问 DOM 元素**：这是最常见的用法。你可以用它来直接操作一个 DOM 节点，比如让输入框自动聚焦。
    
2. **存储一个不触发重新渲染的可变值**：类似于类组件中的实例属性。当你改变 `ref.current` 的值时，组件**不会**重新渲染。
    

**示例：页面加载后自动聚焦输入框**

JavaScript

```
import React, { useRef, useEffect } from 'react';

function AutoFocusInput() {
  // 创建一个 ref 对象
  const inputRef = useRef(null);

  useEffect(() => {
    // 组件挂载后，让 input 元素聚焦
    // inputRef.current 指向的是真实的 DOM 节点
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []); // 空数组确保只运行一次

  return (
    <div>
      <p>这个输入框会自动聚焦</p>
      {/* 将 ref 对象附加到 input 元素的 ref 属性上 */}
      <input ref={inputRef} type="text" placeholder="在这里输入..." />
    </div>
  );
}

export default AutoFocusInput;
```

---

### **自定义 Hooks**

当你发现自己在多个组件中编写了重复的 Hooks 逻辑时，就可以将这些逻辑提取出来，放到一个单独的函数中，这个函数就是一个**自定义 Hook**。自定义 Hook 必须以 `use` 开头（例如 `useUserData`, `useWindowWidth`），这是一个约定。

这极大地增强了逻辑复用的能力。

### **总结**

希望这个详细的介绍能帮助你理解 React Hooks 是什么以及如何使用它们。

- **Hooks 是什么？** 是一些特殊函数，让你的函数组件也能使用 state 和生命周期等特性。
    
- **为什么用？** 代码更简洁，告别 `this`，逻辑复用更方便。
    
- **怎么用？**
    
    - `useState` 管理组件状态。
        
    - `useEffect` 处理副作用（数据获取、DOM 操作等）。
        
    - `useContext` 解决跨层级通信。
        
    - `useRef` 访问 DOM 或存储不触发渲染的值。
        
