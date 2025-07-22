# DayDayDebug: 一个CPU战士的LLM后训练通关实录

## 序章：一切开始于SFT

一切似乎都很顺利。我按照文档，准备好了环境，信心满满地敲下了运行SFT的命令。第一次，因为WSL环境问题失败了，小事一桩，直接用 `accelerate launch` 启动脚本。

> **我**: `accelerate launch src/trainers/sft_trainer.py configs/training/finetune_sft.yaml`

然后，现实给了我沉重的第一击。

## 第一幕：CPU的残酷试炼

#### **第一道坎：GPU之梦**
**报错**: `RuntimeError: No GPU found. A GPU is needed for quantization.`
**我的内心**: "我tm只有cpu 你给我搞个gpu出来.. 无语"
**诊断**: 脚本默认配置了`bitsandbytes`量化，这是GPU专属的功能。
**行动**: 我们一起将配置文件和脚本都修改为CPU兼容模式，禁用了量化，更换了优化器。这是我们迈向CPU训练的第一步。

#### **第二道坎：内存之墙**
**报错**: `OSError: 页面文件太小，无法完成操作。`
**我的内心**: "又他娘的报错 ...啥问题啊到底是"
**诊断**: 加载一个1.1B的模型，即使在CPU上，RAM也不够用。Windows试图使用虚拟内存（页面文件），但默认大小不足。
**行动**: 我们扩大了Windows的页面文件。这是一个纯粹的操作系统层面的修复，也是在资源受限设备上进行大模型开发的必修课。

#### **第三道坎：类型之谜**
**报错**: `TypeError: '<=' not supported between instances of 'float' and 'str'`
**诊断**: 配置文件中的学习率 `2e-4` 被Python的YAML库错误地解析为了字符串。
**行动**: 在代码中对所有从配置中读取的数值参数进行了强制类型转换。一个看似微小却至关重要的修复。

#### **第四道坎：时间之河**
**日志**: `... 78.91s/it`
**我的内心**: "速度慢得太夸张了吧也"
**诊断**: 这不是错误，这就是CPU训练的现实。我们意识到，在CPU上，我们的目标不是训练模型，而是**验证流程**。
**行动**: 大幅缩减了数据集大小和训练步数。`max_steps: 5` 成为了我们快速迭代的法宝。

**SFT通关！** 经过一番搏斗，SFT流程终于跑通了，模型也成功保存。推理脚本也顺利运行，我亲手微调的模型，第一次对我说了话。

## 第二幕：DPO的九九八十一难

SFT的成功给了我信心，我以为DPO也会同样顺利。我太天真了。

#### **第一难：Offload之怒**
**报错**: `ValueError: We need an offload_dir...`
**诊断**: DPO需要加载两个模型（策略模型+参考模型），内存压力是SFT的两倍。`accelerate`检测到RAM不足，需要将模型权重卸载到硬盘，但不知道该卸载到哪里。
**行动**: 我们在代码中增加了`offload_folder`参数。

#### **第二难：Offload之怨**
**报错**: `ValueError: We need an offload_dir...` (再次出现!)
**诊断**: 之前的修复只在加载基础模型时生效了。当`PeftModel`尝试加载适配器时，`accelerate`丢失了`offload_dir`的上下文。
**行动**: 我们改用更稳健的分步加载策略：先加载带卸载的基础模型，再加载适配器。

#### **第三难：逻辑之辩**
**报错**: `ValueError: You passed both a ref_model and a peft_config...`
**诊断**: `DPOTrainer`有自己的PEFT工作流。当我们手动提供`ref_model`时，与它的内部逻辑冲突了。
**行动**: 遵循`trl`库的建议，将`ref_model`设为`None`，让Trainer自己处理。

#### **第四难 & 第五难：数据之形**
**报错**: `ValueError: chosen should be an str but got <class 'list'>`，紧接着是 `...got <class 'dict'>`。
**我的内心**: "我能不能把这个链接硬编码到py文件内？" (此时是对网络问题的疑惑，但很快转向了数据格式) -> "保险起见 我觉得应该先去核实数据 再进行下一步"
**诊断**: 经过`diagnose_dataset.py`的精确打击，我们发现DPO数据集的格式是极其复杂的多轮对话列表。
**行动**: 编写了一个强大的`format_dpo_dataset`函数，使用`tokenizer.apply_chat_template`将复杂的对话列表正确地转换为单一字符串。这是整个调试过程中最体现“深入数据”重要性的一步。

#### **第六难：Meta Tensor之谜**
**报错**: `NotImplementedError: Cannot copy out of meta tensor...`
**诊断**: 即使让Trainer自己创建`ref_model`，`accelerate`的硬盘卸载机制和`Trainer`的模型移动逻辑依然存在深层冲突。
**行动**: **“釜底抽薪”**。我们放弃了所有精巧的卸载技巧，回归最朴素的方法：先在内存中将SFT模型完全合并，然后将这个干净的、普通的、完全在RAM中的模型传递给`DPOTrainer`。这依赖于我们之前扩大的虚拟内存，是一场豪赌。

**DPO通关！** 这场豪赌成功了。DPO流程终于跑通，所有的硬件、内存、API和数据问题都被我们一一攻克。

## 第三幕：PPO的终局之战

有了DPO的惨痛经历，PPO的调试过程更像是一次精准的外科手术。

#### **第一战：网络之困**
**报错**: `ConnectionError: Couldn't reach '...' on the Hub (ConnectTimeout)` & `IncompleteRead`
**我的内心**: "这个下载速度太慢了 有没有加速办法" -> "我能不能把这个链接硬编码到py文件内？"
**诊断**: Hugging Face Hub的直接连接在国内网络环境下极其不稳定。
**行动**: 我们先是硬编码了国内镜像，但这还不够。最终，我们实现了一个**V12.0版本的智能数据引擎**：默认乐观地尝试并行下载，如果失败，则自动清理缓存并降级为带重试的串行下载。这是整个项目中最具工业级强度的代码之一。

#### **第二战：权限之门**
**报错**: `401 Client Error... Repository Not Found...`
**诊断**: 我们尝试的某个数据集是需要登录授权的“门禁”数据集。
**我的内心**: "我想换成完全公开的 体积很小的数据集"
**行动**: 果断更换为`imdb`数据集，彻底消除了认证环节，保证了教程的流畅性。

#### **第三战 & 第四战：API之隙**
**报错**: `AttributeError: 'str' object has no attribute 'rfilename'`，紧接着是`TypeError: PPOTrainer... got an unexpected keyword argument 'peft_config'`。
**诊断**: `huggingface_hub`库的API返回类型有变化；`PPOTrainer`不接受`peft_config`参数。
**行动**: 修正了API使用，确保代码的健壮性和正确性。

#### **第五战 & 第六战：数据之殇**
**报错**: `AttributeError: 'list' object has no attribute 'items'`，然后是`ValueError: query_tensor must be a tensor of shape...`。
**诊断**: 我们自定义的`data_collator`与`PPOTrainer`的内部逻辑冲突；传递给`generate`函数的数据形状不正确。
**行动**: 移除了多余的`data_collator`，并修正了传递给`generate`的数据形状，使其符合`trl`的API要求。

#### **第七战 & 第八战：日志之美**
**报错**: `KeyError: 'ppo/rewards/mean'`，然后是`TypeError: unsupported format string passed to numpy.ndarray...`
**我的内心**: "full stats 是你应该在控制台完整输出的东西吗？这样控制台还有什么可读性可言？？？？？？"
**诊断**: 返回的日志键名有变；日志字典中包含了无法直接格式化的NumPy数组。
**行动**: 实现了基于白名单的专业日志过滤和格式化，让控制台输出变得清晰、优雅且充满信息量。

#### **最终决战：文件夹之役**
**报错**: `RuntimeError: Parent directory ... does not exist.`
**诊断**: `save_pretrained`不会自动创建不存在的父目录。
**行动**: 在保存模型前，手动创建目标目录。

**PPO通关！**

## 尾声：胜利属于坚持不懈的我们

从一个简单的SFT脚本开始，到最终攻克PPO，我们解决了几十个横跨操作系统、网络、内存管理、库版本、API使用和数据格式的真实世界问题。

这段旅程，就是一部微缩的AI工程史。它充满了挫折、困惑，但最终，通过一步一个脚印的分析、验证和修复，我们抵达了终点。

我不再是那个只会敲命令的人，我理解了每一个错误背后的原因，也学会了如何用系统性的方法去解决它们。

这个项目，现在不仅是一份代码，更是这段征程的勋章。
