现代的GO已经不需要配置`GOROOT`和`GOPATH`这些环境变量了，详细的配置教程参考下面微软的教程：

https://learn.microsoft.com/zh-cn/azure/developer/go/configure-visual-studio-code

# 配置代理

不配置代理，下载包的时候超级难受（和pip一个鸟样）

```shell
go version
go env -w GOPROXY=https://goproxy.cn
go env GOPROXY
```

