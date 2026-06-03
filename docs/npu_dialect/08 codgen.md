MLIR编译器和ISA解耦
1、MLIR编译器codegen模块把图级IR翻译成算子库能理解的静态请求，再把结果组织成模型产物。
2、算子库负责生成汇编指令、指令优化，维护指令buffer，使用汇编器生成二进制指令

NOTE: 需要实现ada200的汇编器

CodeGen主要流程：
 - 遍历Op，进入各算子自己的函数，构造结构化参数，调用算子库接口，生成算子对应的指令（算子库维护各kernel指令buffer）
 - MLIR编译器调用算子库获取指令buffer， 生成rxmodel

具体实现：
假设
 - 已有IR，做完内存分配，该有的参数都有。

算子codegen：
	- 每个算子实现各自的codegen：根据算子的特点，构造对应的结构化参数，再调用算子库生成指令

codegen主流程：
	for op in all_ops
    if op is Group:
      codegen_for_group(..)
      store_info(ctx)
    else:
      codegen_for_op -> call op impl（考虑多线程编译？）
      store_info(ctx)

	buildRxModel(ctx)



任务分解：
一、ada200算子库
二、ada200汇编器
三、codegen主体框架  -done
四、codegen算子层
五、rxmodel设计及实现

优先级
1 任务一和任务三
2 任务四和任务二
3 任务五
