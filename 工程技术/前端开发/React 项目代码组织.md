#React #Claude 

---
## TypeScript项目结构

```
src/
├── api/                     # API相关代码
│   ├── client.ts           # API客户端配置
│   ├── endpoints.ts        # API端点常量
│   └── services/           # 按模块分组的API服务
│       ├── userService.ts
│       ├── adminService.ts
│       ├── emailService.ts
│       ├── paymentService.ts
│       └── translationService.ts
├── types/                  # TypeScript类型定义
│   ├── api.ts             # API相关类型
│   ├── user.ts            # 用户相关类型
│   ├── admin.ts           # 管理员相关类型
│   ├── payment.ts         # 支付相关类型
│   ├── email.ts           # 邮件相关类型
│   └── common.ts          # 通用类型
├── hooks/                 # 自定义钩子
│   ├── useApi.ts          # 通用API钩子
│   └── queries/           # React Query钩子
│       ├── useUsers.ts
│       ├── useAdmin.ts
│       └── ...
├── utils/
│   ├── request.ts         # 请求工具函数
│   └── errorHandler.ts    # 错误处理
└── components/            # 你的UI组件
```

## TypeScript版本的核心代码

```ts
// 1. 通用类型定义 (src/types/common.ts)
export interface ApiResponse<T = any> {
  data: T;
  message: string;
  success: boolean;
  code: number;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
  sort?: string;
  order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface ApiError {
  message: string;
  code: string;
  details?: Record<string, any>;
}

// 2. 用户相关类型 (src/types/user.ts)
export interface User {
  id: number;
  name: string;
  email: string;
  avatar?: string;
  phone?: string;
  role: 'user' | 'admin' | 'moderator';
  status: 'active' | 'inactive' | 'suspended';
  createdAt: string;
  updatedAt: string;
}

export interface CreateUserRequest {
  name: string;
  email: string;
  password: string;
  phone?: string;
  role?: User['role'];
}

export interface UpdateUserRequest {
  name?: string;
  email?: string;
  phone?: string;
  avatar?: string;
}

export interface UserProfile {
  id: number;
  name: string;
  email: string;
  avatar?: string;
  phone?: string;
  preferences: {
    language: string;
    theme: 'light' | 'dark';
    notifications: boolean;
  };
}

// 3. 支付相关类型 (src/types/payment.ts)
export interface Payment {
  id: string;
  amount: number;
  currency: 'USD' | 'EUR' | 'CNY';
  status: 'pending' | 'completed' | 'failed' | 'cancelled';
  method: 'credit_card' | 'paypal' | 'alipay' | 'wechat';
  description: string;
  userId: number;
  createdAt: string;
  updatedAt: string;
}

export interface CreatePaymentRequest {
  amount: number;
  currency: Payment['currency'];
  method: Payment['method'];
  description: string;
  returnUrl?: string;
}

export interface PaymentMethod {
  id: string;
  type: Payment['method'];
  name: string;
  isEnabled: boolean;
  config: Record<string, any>;
}

// 4. API端点类型定义 (src/types/api.ts)
export interface ApiEndpoints {
  USER: {
    LIST: string;
    DETAIL: (id: number) => string;
    CREATE: string;
    UPDATE: (id: number) => string;
    DELETE: (id: number) => string;
    PROFILE: string;
  };
  PAYMENT: {
    CREATE: string;
    HISTORY: string;
    METHODS: string;
  };
  // ... 其他端点
}

// Service方法的类型定义
export interface UserServiceType {
  getUsers: (params?: PaginationParams) => Promise<PaginatedResponse<User>>;
  getUserById: (id: number) => Promise<User>;
  createUser: (userData: CreateUserRequest) => Promise<User>;
  updateUser: (id: number, userData: UpdateUserRequest) => Promise<User>;
  deleteUser: (id: number) => Promise<void>;
  getUserProfile: () => Promise<UserProfile>;
}

// 5. API客户端配置 (src/api/client.ts)
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse, ApiError } from '../types/common';

class ApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = process.env.REACT_APP_API_BASE_URL || '') {
    this.client = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config: AxiosRequestConfig) => {
        const token = localStorage.getItem('authToken');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: any): ApiError {
    if (error.response) {
      return {
        message: error.response.data?.message || '请求失败',
        code: error.response.data?.code || 'UNKNOWN_ERROR',
        details: error.response.data?.details,
      };
    } else if (error.request) {
      return {
        message: '网络连接失败',
        code: 'NETWORK_ERROR',
      };
    } else {
      return {
        message: error.message || '未知错误',
        code: 'UNKNOWN_ERROR',
      };
    }
  }

  // 通用请求方法
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, config);
    return response.data.data;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config);
    return response.data.data;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config);
    return response.data.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url, config);
    return response.data.data;
  }
}

export const apiClient = new ApiClient();

// 6. API端点常量 (src/api/endpoints.ts)
export const API_ENDPOINTS = {
  USER: {
    LIST: '/users',
    DETAIL: (id: number): string => `/users/${id}`,
    CREATE: '/users',
    UPDATE: (id: number): string => `/users/${id}`,
    DELETE: (id: number): string => `/users/${id}`,
    PROFILE: '/users/profile',
  },
  ADMIN: {
    DASHBOARD: '/admin/dashboard',
    USERS: '/admin/users',
    STATISTICS: '/admin/statistics',
  },
  PAYMENT: {
    CREATE: '/payments',
    HISTORY: '/payments/history',
    METHODS: '/payments/methods',
  },
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
  },
} as const;

// 7. 用户服务 (src/api/services/userService.ts)
import { apiClient } from '../client';
import { API_ENDPOINTS } from '../endpoints';
import { 
  User, 
  CreateUserRequest, 
  UpdateUserRequest, 
  UserProfile,
  PaginationParams,
  PaginatedResponse 
} from '../../types';

export class UserService {
  async getUsers(params?: PaginationParams): Promise<PaginatedResponse<User>> {
    try {
      return await apiClient.get<PaginatedResponse<User>>(
        API_ENDPOINTS.USER.LIST, 
        { params }
      );
    } catch (error) {
      throw this.handleError(error, '获取用户列表失败');
    }
  }

  async getUserById(id: number): Promise<User> {
    try {
      return await apiClient.get<User>(API_ENDPOINTS.USER.DETAIL(id));
    } catch (error) {
      throw this.handleError(error, '获取用户详情失败');
    }
  }

  async createUser(userData: CreateUserRequest): Promise<User> {
    try {
      return await apiClient.post<User>(API_ENDPOINTS.USER.CREATE, userData);
    } catch (error) {
      throw this.handleError(error, '创建用户失败');
    }
  }

  async updateUser(id: number, userData: UpdateUserRequest): Promise<User> {
    try {
      return await apiClient.put<User>(API_ENDPOINTS.USER.UPDATE(id), userData);
    } catch (error) {
      throw this.handleError(error, '更新用户失败');
    }
  }

  async deleteUser(id: number): Promise<void> {
    try {
      await apiClient.delete<void>(API_ENDPOINTS.USER.DELETE(id));
    } catch (error) {
      throw this.handleError(error, '删除用户失败');
    }
  }

  async getUserProfile(): Promise<UserProfile> {
    try {
      return await apiClient.get<UserProfile>(API_ENDPOINTS.USER.PROFILE);
    } catch (error) {
      throw this.handleError(error, '获取用户资料失败');
    }
  }

  private handleError(error: any, defaultMessage: string): Error {
    const message = error.message || defaultMessage;
    console.error('User Service Error:', error);
    return new Error(message);
  }
}

export const userService = new UserService();

// 8. 支付服务 (src/api/services/paymentService.ts)
import { apiClient } from '../client';
import { API_ENDPOINTS } from '../endpoints';
import { 
  Payment, 
  CreatePaymentRequest, 
  PaymentMethod,
  PaginationParams,
  PaginatedResponse 
} from '../../types';

export class PaymentService {
  async createPayment(paymentData: CreatePaymentRequest): Promise<Payment> {
    try {
      return await apiClient.post<Payment>(API_ENDPOINTS.PAYMENT.CREATE, paymentData);
    } catch (error) {
      throw this.handleError(error, '创建支付失败');
    }
  }

  async getPaymentHistory(params?: PaginationParams): Promise<PaginatedResponse<Payment>> {
    try {
      return await apiClient.get<PaginatedResponse<Payment>>(
        API_ENDPOINTS.PAYMENT.HISTORY, 
        { params }
      );
    } catch (error) {
      throw this.handleError(error, '获取支付历史失败');
    }
  }

  async getPaymentMethods(): Promise<PaymentMethod[]> {
    try {
      return await apiClient.get<PaymentMethod[]>(API_ENDPOINTS.PAYMENT.METHODS);
    } catch (error) {
      throw this.handleError(error, '获取支付方式失败');
    }
  }

  private handleError(error: any, defaultMessage: string): Error {
    const message = error.message || defaultMessage;
    console.error('Payment Service Error:', error);
    return new Error(message);
  }
}

export const paymentService = new PaymentService();

// 9. 通用API Hook (src/hooks/useApi.ts)
import { useState, useEffect, useCallback } from 'react';

interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseApiReturn<T> extends ApiState<T> {
  execute: (...args: any[]) => Promise<T>;
  setData: (data: T | null) => void;
  setError: (error: string | null) => void;
}

export const useApi = <T = any>(
  apiFunction: (...args: any[]) => Promise<T>,
  dependencies: any[] = []
): UseApiReturn<T> => {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(async (...args: any[]): Promise<T> => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const result = await apiFunction(...args);
      setState(prev => ({ ...prev, data: result, loading: false }));
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '未知错误';
      setState(prev => ({ ...prev, error: errorMessage, loading: false }));
      throw err;
    }
  }, dependencies);

  const setData = useCallback((data: T | null) => {
    setState(prev => ({ ...prev, data }));
  }, []);

  const setError = useCallback((error: string | null) => {
    setState(prev => ({ ...prev, error }));
  }, []);

  return {
    ...state,
    execute,
    setData,
    setError,
  };
};

// 10. React Query Hook (src/hooks/queries/useUsers.ts)
import { useQuery, useMutation, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { userService } from '../../api/services/userService';
import { User, CreateUserRequest, UpdateUserRequest, PaginationParams, PaginatedResponse } from '../../types';

export const useUsers = (params?: PaginationParams): UseQueryResult<PaginatedResponse<User>> => {
  return useQuery({
    queryKey: ['users', params],
    queryFn: () => userService.getUsers(params),
    staleTime: 5 * 60 * 1000, // 5分钟
  });
};

export const useUser = (id: number): UseQueryResult<User> => {
  return useQuery({
    queryKey: ['user', id],
    queryFn: () => userService.getUserById(id),
    enabled: !!id,
  });
};

export const useCreateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (userData: CreateUserRequest) => userService.createUser(userData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
};

export const useUpdateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, userData }: { id: number; userData: UpdateUserRequest }) => 
      userService.updateUser(id, userData),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      queryClient.invalidateQueries({ queryKey: ['user', variables.id] });
    },
  });
};

// 11. 组件中使用示例 (src/components/UserList.tsx)
import React, { useEffect } from 'react';
import { useApi } from '../hooks/useApi';
import { userService } from '../api/services/userService';
import { User, PaginationParams, PaginatedResponse } from '../types';

interface UserListProps {
  filters?: PaginationParams;
}

const UserList: React.FC<UserListProps> = ({ filters }) => {
  const {
    data: users,
    loading,
    error,
    execute: fetchUsers
  } = useApi<PaginatedResponse<User>>(userService.getUsers);

  useEffect(() => {
    fetchUsers(filters);
  }, [fetchUsers, filters]);

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error}</div>;

  return (
    <div>
      <h2>用户列表</h2>
      {users?.data.map((user: User) => (
        <div key={user.id}>
          <h3>{user.name}</h3>
          <p>{user.email}</p>
          <span>状态: {user.status}</span>
        </div>
      ))}
      
      {users?.pagination && (
        <div>
          第 {users.pagination.page} 页，共 {users.pagination.totalPages} 页
        </div>
      )}
    </div>
  );
};

export default UserList;

// 12. 类型安全的错误处理 (src/utils/errorHandler.ts)
import { ApiError } from '../types/common';

export const handleApiError = (error: unknown): string => {
  if (error && typeof error === 'object' && 'message' in error) {
    const apiError = error as ApiError;
    
    switch (apiError.code) {
      case 'VALIDATION_ERROR':
        return '输入数据有误，请检查后重试';
      case 'UNAUTHORIZED':
        return '未授权，请重新登录';
      case 'FORBIDDEN':
        return '权限不足';
      case 'NOT_FOUND':
        return '请求的资源不存在';
      case 'NETWORK_ERROR':
        return '网络连接失败，请检查网络';
      default:
        return apiError.message || '未知错误';
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return '发生了未知错误';
};

// 类型守卫函数
export const isApiError = (error: unknown): error is ApiError => {
  return (
    error !== null &&
    typeof error === 'object' &&
    'message' in error &&
    'code' in error
  );
};
```

## TypeScript配置文件

还需要配置一些TypeScript相关的文件：

### 1. tsconfig.json 配置

```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@/*": ["*"],
      "@/types/*": ["types/*"],
      "@/api/*": ["api/*"],
      "@/hooks/*": ["hooks/*"],
      "@/utils/*": ["utils/*"]
    }
  },
  "include": [
    "src"
  ]
}
```

### 2. 环境变量类型定义

```typescript
// src/types/env.d.ts
declare namespace NodeJS {
  interface ProcessEnv {
    REACT_APP_API_BASE_URL: string;
    REACT_APP_API_TIMEOUT: string;
    NODE_ENV: 'development' | 'production' | 'test';
  }
}
```

## TypeScript的主要变化和优势

### 1. **类型安全**

- 编译时就能发现错误
- IDE提供更好的智能提示
- API调用的参数和返回值都有类型约束

### 2. **更好的开发体验**

- 自动补全更准确
- 重构更安全
- 文档即代码（类型定义就是最好的文档）

### 3. **接口约定**

- 前后端可以共享类型定义
- API变更时能及时发现问题

### 4. **维护性更强**

- 代码可读性更高
- 团队协作时减少理解成本

### 5. **实际使用示例**

```typescript
// 类型安全的API调用
const handleCreateUser = async (userData: CreateUserRequest) => {
  try {
    const newUser = await userService.createUser(userData);
    // newUser 的类型是 User，IDE 会提供准确的提示
    console.log(`创建了用户: ${newUser.name}`);
  } catch (error) {
    // 类型安全的错误处理
    const errorMessage = handleApiError(error);
    setError(errorMessage);
  }
};

// 组件 props 类型约束
interface UserCardProps {
  user: User;
  onEdit: (id: number) => void;
  onDelete: (id: number) => void;
}

const UserCard: React.FC<UserCardProps> = ({ user, onEdit, onDelete }) => {
  // user 参数有完整的类型提示
  return (
    <div>
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      <button onClick={() => onEdit(user.id)}>编辑</button>
      <button onClick={() => onDelete(user.id)}>删除</button>
    </div>
  );
};
```

### 6. **推荐的包安装**

```bash
# TypeScript 相关
npm install -D typescript @types/react @types/react-dom @types/node

# API 请求
npm install axios
npm install -D @types/axios

# 状态管理（如果使用 React Query）
npm install @tanstack/react-query
```

使用TypeScript最大的好处是在开发阶段就能发现很多潜在的错误，特别是在API交互这种容易出错的地方。虽然初期需要写更多的类型定义，但长期来看会大大提高代码质量和开发效率。