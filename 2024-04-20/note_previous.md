[SIGMA](#SIGMA: A Sparse and Irregular GEMMAccelerator with Flexible Interconnects for DNN Training(HPCA20))

[Sparseloop](#Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling(MICRO’22))

[GoSPA](#GoSPA: An Energy-efﬁcient High-performance Globally Optimized SParse Convolutional Neural Network Accelerator)

[Sparse Tensor Core](#Sparse Tensor Core: Algorithm and Hardware Co-Design for Vector-wise Sparse Neural Networks on Modern GPUs)

[ASADI](#ASADI: Accelerating Sparse Attention using Diagonal-based In-situ Computing)





### SIGMA: A Sparse and Irregular GEMMAccelerator with Flexible Interconnects for DNN Training(HPCA20)

* 研究问题

深度学习中的矩阵乘大多数是非规则、稀疏的，这使得脉动阵列上的计算效率降低(元素映射分块、零值对计算结果无影响但仍需进行计算)。文章给出了一种新的元素映射的方式，提出了两个network改变了每个结点之间的互联、降低了数据传输时间复杂度。

* 挑战
  
  * 脉动阵列仍然要将零值映射到PE上，SIGMA只需要将非零值映射到结点上。
  * 随着脉动阵列规模的增加，数据加载、部分和累加耗时也随之增加。SIGMA使用的网络能实现O(1)的数据加载、O($log_2N$)的部分和累加。

* 解决方案

这篇文章主要用了两个结构 ：**Distribution Network**(**Benes Dist Network)**和**reduction network**

Distribution Network也叫Benes Dist Network，是引用[7]中的Benes topology,实现了数据在阵列中传播的时间复杂度是O(1) 。

Benes Dist Network中没有乘运算：A,B矩阵都从阵列上方注入，B矩阵元素(ABDEFGHI)先注入，A矩阵元素后注入(元素匹配是通过figure5 (v) (vi) 中的ID实现的)，当两个匹配的元素到绿框Multipliers时，进行相乘操作。

Reduction Network也叫FAN Topology ,Forwarding Adder network , Figure 6a), 就是将”流下来的“乘积，选择性地相加(本来矩阵乘应该相加的项进行相加).





### Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling(MICRO’22)

**启示与作用：**

1.       将稀疏加速的方式归类成: representation format , skipping
 IneffOps, gating IneffOps 

2.       指出SOTA的SAF(sparse
 accumulator feature)，即指出了SOTA加速的方法 

3.       模拟器方向论文，前半部分可以当成稀疏加速器的综述 

4.       多精度方向进行了分类(L1,L2,L3)[1],该文章则对稀疏加速器按照加速方法进行了分类,有利于设计空间探索？(模拟器**开源**,提供docker)



**概述**

1.论文前半部分对加速器分成三类(SAF:sparse accumulator feature):representation format , gating , skipping;

2.按SAF对SOTA分类

3.four per-dimension compressed formats: B(bitmask),CP,RLE,UOP;更高层的representation format可以由B,CP,RLE,UOP组合表示

4.gating分为leader-follower
intersection , douled-side intersection

5.skipping:

6.后半部分是模拟器的内容



### GoSPA: An Energy-efﬁcient High-performance Globally Optimized SParse Convolutional Neural Network Accelerator

**概述**

1.       该文是对CNN的加速，加速对象是矩阵卷积，矩阵卷积可以转换为向量点积

2.       提出了on-the-fly intersection,用于省去不必要的计算，同时降低了检测的开销(属于门控gating)

3.       重排了计算顺序，减少了存储访问开销(例如fig4中5要与ABCD分别相乘一次，那么5就得访存4次，重排后只要访存一次5同时与ABCD计算)

4.       硬件设计也是“利用更多的编码信息、更多的选择(同时也带来了开销)，实现没有任何多余的计算”

补充一些点：

* Fig2中做的是矩阵卷积，不是矩阵乘。但是能将矩阵卷积转化为向量乘加fig 5

* fig8中不同PE间是会共享激活的，例如fig11中FIFO-B#1与#2有共同的(3,-2,4),PID均为3，那么PID为3的激活要同时广播到两个PE中进行相乘

* APU：1. fig11中将激活矩阵分成四份放入FIFO-A中，FIFO-A#1中放PID=0的数，FIFO-A#i中放PID=i-1的数；

* fig9中FIFO-A中的数如何通过Routing Module进入FIFO-B呢？回到fig11 , PE中的weight是固定的，例如PE#1中恒为[1 0 ; 3 0] , WSP_1为1010(weight的bitmap) ,PID=0123 , WSP_1[1],WSP_1[3]=0表示这两个位置的数为0，也就意味着PID=1,3的激活不用送进该PE(该FIFO-B)因此，我们看到FIFO-A#1(PID=0)进入了FIFO-B(其中权重有PID=0)，不进入FIFO-B#2(无权重PID=0). 而FIFO-A#3(PID=2)均进入FIFO-B#1,2 (两个FIFO-b均有PID=2)  ; 进入FIFO-B后，每个元素逐拍进入PE

* PE内部：fig13一个PE内权重矩阵是固定的,例如图中权重为[0  1 ; 0 2] , 只有PID=1，3 ， 因此送入该PE的激活只有[0 2 -2 0]  [-2 0 3 -3]  PE中按PID乘，按CID加



### Sparse Tensor Core: Algorithm and Hardware Co-Design for Vector-wise Sparse Neural Networks on Modern GPUs

1. Co-design 软件算法：VectorSparse ; 硬件:修改GPUSim
2. 分析了generic sparsitying,unified sparsitying的缺点，提出了VectorSparsity:将矩阵拆成长度L的一维向量，对每个向量进行压缩(剪枝)长度为K(K=max(number of NNZ of every vector)),并记录offset
3. 介绍了tensor core结构相关知识(可参考)  figure7表示了剪枝后的向量如何乘矩阵
4. 扩展了GPGPUSim指令集





### ASADI: Accelerating Sparse Attention using Diagonal-based In-situ Computing

**概述**

​	文章找到了一个稀疏矩阵的新应用场景：稀疏对角矩阵，因此其编码方式、数据流都是按对角线来的。文章工作主要有两点：1. 建立新的对角线编码格式与数据流，并介绍了这种新编码格式在SPMM和SDDMM上的dataflow,提高了对角线的局部性  2. 提出了一种面向transformer模型的存内计算ReRAM的结构 ， 后者与稀疏关系不大。

​	感觉和脉动的行压缩(也是不改变A矩阵元素的列号)差不多，A矩阵的列与B矩阵的行对齐往里面流，只是这篇文章的应用场景是对角矩阵(邻接矩阵)，所以编码(压缩)方式就要按对角线来。

没找到能打这篇文章的地方，但是不是以后可以从软件的角度反推硬件的设计？也就是说做硬件要适当关注软件现有的缺陷？(软件硬件化)(大佬组软硬件都有人，软件放出来的蛋糕会落你手里？)(还是要动手做，无论软件硬件，这样才能发现问题，只空想感觉什么都是完美的、什么都被做完了 但TM都不开源，复现起来时间成本太高)      想不清烦:(



**关键图表**

![image-20240417171153476](images/image-20240417171153476.png)

![image-20240417171300184](images/image-20240417171300184.png)





### HotTiles: Accelerating SpMM with Heterogeneous Accelerator Architectures

**概述**

1. 目前的加速器用同一套PEs对稀疏矩阵中所有元素做计算，本文提出了Intra-Matrix Heterogeneity (IMH)（矩阵内异构），把一个稀疏矩阵分成多个tiles,较稠密的tile被称为`hot tile`，用计算密集的计算单元(`Hot Worker`)处理，稀疏的`tile`用访存友好(memory-bound)的计算单元(Cold Worker)处理。![image-20240418200052039](images/image-20240418200052039.png)

2. 提出了评估`cold tile`和`hot tile`的模型框架：估计访问主存的总bytes；估计每一个tile的计算时间(Execution Time，实际上算的是FLOPs)。

   ![image-20240418194727080](images/image-20240418194727080.png)



3.设计启发式算法去对每一个tile进行分类(cold or hot)

![image-20240418195137903](images/image-20240418195137903.png)

![image-20240418194331772](images/image-20240418194331772.png)

