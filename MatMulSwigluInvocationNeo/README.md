## 目录结构介绍
```
├── MatMulInvocationNeo
│   ├── cmake                       // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本文件
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   ├── main.cpp                    // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   ├── matmul_custom.cpp           // 算子kernel实现
│   ├── matmul_custom_tiling.cpp    // 算子tiling实现
│   └── run.sh                      // 编译运行算子的脚本
```
## 代码实现介绍
本调用样例中实现的是[m, n, k]固定为[512, 1024, 512]的Matmul算子。
- kernel实现  
  Matmul算子的数学表达式为：
  $$
  C = A * B + Bias
  $$
  其中A的形状为[512, 1024], B的形状为[1024, 512], C的形状为[512, 512], Bias的形状为[1, 1024]。具体请参考[matmul_custom.cpp](./matmul_custom.cpp)。

- 调用实现  
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录

    ```bash
    cd ${git_clone_path}/samples/operator/MatMulCustomSample/KernelLaunch/MatMulInvocationNeo
    ```
  - 配置环境变量

    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```

  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
      - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]，默认值为cpu。

    注：针对Atlas 训练系列产品使用NPU仿真调试，会存在精度问题，可选择其他芯片进行NPU仿真调试。

    示例如下。

    ```bash
    bash run.sh -r cpu -v Ascend310P1
    ```


### 注意  
  当使用非昇腾设备在sim和cpu模式下执行，会提示找不到对应设备的错误。这是因为tiling构造函数的入参ascendcPlatform默认从系统中获取昇腾AI处理器型号，当前不支持用户手动指定。若用户在非昇腾设备运行样例，需要修改代码，替换构造函数为不传入ascendcPlatform的默认构造函数，生成Tiling结构体。
  ```c++
  // 原代码
  MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
  // 修改后代码
  MultiCoreMatmulTiling tilingApi;
  ```
  关于Matmul Tiling构造函数的详细信息，可参考[Ascend C 高阶API](https://hiascend.com/document/redirect/CannCommunityAscendCHighLevelApi)>Matmul >Matmul Tiling 章节。


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2023/05/21 | 更新本readme |