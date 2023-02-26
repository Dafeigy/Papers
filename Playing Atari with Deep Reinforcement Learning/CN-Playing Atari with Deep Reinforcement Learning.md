
<center><h1>摘要</center>

> 我们提出首个直接通过传感器高维输入并成功地学习到策略控制的深度学习模型。这个模型的是一个卷积神经网络的结构，它使用Q-learning的变种进行训练，使用远摄关的图片作为输入，输出是预测未来收益的价值。我们将我们的方法运用至Arcade Learning环境中的雅达利2600的七个游戏里，在没有调整算法的结构的条件下，我们发现它在其中六个游戏中取得比之前方法更好的表现，它在其中三个游戏中取得了超越人类专家的表现。

<!--more-->

##  引言

直接从像视觉和语音等高维传感输入信息来学习控制智体一直以来都是强化学习(RL)的一个挑战。大部分成功的RL在这些领域的操作应用都需要依赖人工干预的线性价的值函数或策略函数进行特征提取整合。显然，这些RL算法的表现严重依赖于所选取的特征的质量。

最近深度学习的优势是它能够从原生的传感器数据中提取深层次的特征，并在计算机视觉任务[11,22,16]和语言识别任务[6,7]中取得突破性进展。这些方法使用了一系列的神经网络结构，包括卷积神经网络、多层感知机，受限玻尔兹曼机器和循环神经网络，他们都在监督和无监督学习中得到了应用,这样一来似乎很自然地会思考相似的技术是否能对RL的传感器数据有所帮助。

然而强化学习在从深度学习的观点来看有几个挑战。首先大部分成功的深度学习案例都需要大量的人工标注的训练数据。从另一个方面来说，RL的算法必须要能从量化的奖罚信号中进行学习，而这些奖罚信号通常都是充满噪声、延迟和离散的。执行动作后的以及随之而来的奖罚可能需要等待非常久的时间，在这一点上和有监督学习中的输入与输出的直接关联相比就显得艰难非常。另一个问题是绝大多数的深度学习算法都是基于样本数据之间是独立的这一假设的，但是在强化学习中通常就是状态与状态之间紧密相连，关系复杂。并且，在RL中数据概率分布会随着算法学习到新的行为而随之改变，这对深度学习基于固定概率的条件来说无疑是个的问题。

本文提出了一种可以克服这些挑战以学习到成功的控制策略的卷积神经网络，而它仅需要使用复杂的RL环境中的原生视频数据。这个网络使用了一种Q-learning 的变种算法[26]进行训练，使用随机梯度下降法进行网络参数的更新。为了避免相关数据和概率不变的问题，我们使用了一种经验回放机制[13]，他能够随机地从经验池中抽取这些学习的经验，因此这能平缓训练时对先前学习到的动作分布概率。



<img src="https://s2.loli.net/2022/05/26/6w8c9R5T7DOHEBm.gif" alt="img" style="zoom:150%;" />

<small>图1  雅达利2600五种游戏的屏幕截图：（从左至右）Pong, Breakout, Space Invaders, Seaquest, Beam Rider</small>



我们将我们的方法应用于在Arcade Learning Environment(ALE)[3]中一系列的雅达利2600游戏中。雅达利2600是一个充满挑战性的RL测试平台，它为智体提供了高纬度的视觉输入信息(210*160大小的RGB彩色图像，屏幕刷新率为60Hz)以及为人类设计的苦难但多样有趣的挑战任务。我们的目标是去创建一个单独的神经网络作为智体使它能够成功的学习玩尽可能多的游戏。这个网路并不会被提供任何游戏相关的信息或者人为给定的视觉特征，并且也不能直接读取游戏环境的内存数据，它只能从视觉信息、奖罚西欧马屁、游戏结束信号以及一个可选则的动作空间进行学习，就和人类玩家一样。更进一步，网络结构和所有的超参数将在训练过程中保持不变。目前网络已经在7个中的6个游戏中超越之前的所有强化学习算法的性能，并且在三个游戏中甚至还取得了超越人类专家游玩的水平。图片[1]展示了五个用于训练的游戏截图。



## 背景

我们将问题转换为如下场景：智体在环境$\cal \varepsilon$即雅达利2600模拟器中，在每一个时间步中智体从一系列的合法动作集合$\cal A=\{1,...,K\}$中，选取一个动作$a_t$。这个选择的动作将会被传递至模拟器中并更改了游戏的底层交互数据和游戏得分，一般来说$\sf \varepsilon$是随机的。这些底层的交互数据对智体而言不可见，对它而言它只能从模拟器中观测到一张图片$x_t\in X^d$，这张图片其实一个原生的像素值的向量，描绘了当前的模拟器画面。除此之外，智体还会接收到一个奖罚信号$r_t$，代表了游戏的分的变化。注意一般来说游戏分数取决于之前的动作决策序列和观测；关于动作决策的评价反馈可能在决策做出之后的许多时间步之后才得到。

由于智体只能观测到当前屏幕的图片信息，我们的任务只能观测到片面的信息，比如说对于整个游戏当前的状态而言，仅用一张图片$x_t$是很难表述的。因此我们需要考虑动作序列和观测的序列$s_t=x_1,x_2,...,x_{t-1},x_t$，我们将基于这些序列学习游戏的策略。我们认为所有的在模拟器中迭代的序列将在有限的时间步中终止。这种规范化使得我么你的问题变成一个庞大但有限的马尔可夫决策过程（MDP），在这种情况中每一个序列是一个独立的状态。最终，我们可以在马尔科夫链中使用标准的强化学习算法，仅简单地使用完整的状态序列$s_t$作为每一个时间的表征。

​		智体学习的目标是通过选择一个能在未来带来最大化回报的动作，并将这个动作与模拟器交互。我们基于“未来的回报将会随着每一个时间步以一个折扣因子$\gamma$的倍率进行衰减”这一标准的设想，定义未来$t$时刻未来折扣汇报$R_t = \sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}$，其中$T$是终止时的时间步。我们定义一个最优动作-价值函数$Q^{\star}$，它表示为在观测一些序列$s$后根据任一策略采取一些动作$a$所期望得到的最大回报值，定义如下：$Q^{*}(s,a)=\max_\pi\Bbb E[R_t|s_t=s,a_t=a,\pi]$，其中$\pi$是一个策略，表述了一个状态到动作的映射的关系（或者状态到动作的概率分布映射）。

​		最优的动作价值函数遵循一个非常的性质，即贝尔曼方程。它基于如下的设想：如果某一状态的最优的动作价值$Q^{\star}$在下一个时间步中成为了所有可行动作的参考，那么最优的策略就是选择动作价值最高的动作$a'$以获取最大化的期望值$r+\gamma Q^{\star}(s',a;)$，
$$
Q^{*}(s,a)=\Bbb E_{s'\sim\varepsilon}[r+\gamma \max_{a'}Q^{\*}(s',a')|_{s,a}]
$$
​		许多强化学习算法的背后逻辑就是找到用于衡量动作价值函数的方法，通过使用贝尔曼方程作为迭代更新，更新的动作价值为$Q_{i+1}(s,a)=\Bbb E[r+\gamma \max_{a'}Q_i(s',a')|s,a]$。这种动作价值的迭代算法会依最优的动作价值函数收敛，即当$i\rightarrow\infty$时$Q_i\rightarrow Q^{\star}$[23]。在实践中，这种方法是非常不切实际的，因为动作价值函数是会依据每一个状态序列进行独立的估计且不具备泛化能力。因此，我们很自然地会想到用一个近似的函数取拟合动作价值函数，$Q(s,a;\theta)\approx Q^{\star}(s,a)$。在强化学习领域常会使用一个线性的函数进行拟合，但是有时一个非线性的函数也会被用于拟合，比如说神经网络。我们参考使用了一个权重为$\theta$的神经网络作为价值评估网络——Q网络。Q网络可以通过最小化变化的每一迭代$i$过程中损失函数序列值$L_i(\theta_i)$来训练：
$$
L_i(\theta_i)=\Bbb E_{s,a\sim \rho(·)}[(y_i-Q(s,a;\theta_i))^2]
$$
​		其中$y_i=\Bbb E_{s'\sim\varepsilon}[r+\gamma \max_{a'}Q(s',a;\theta_{i-1})-Q(s,a;\theta_i)|_{s,a}]$是迭代$i$过程中的目标值，$\rho(s,a)$是一个基于所有状态$s$和动作$a$的概率分布，我们称之为行为分布。在前一步的迭代过程中的网络参数$\theta_{i-1}$会在优化损失函数值$L_i(\theta_i)$时被冻结（即不是所有时候都是冻结的），注意到目标值是基于神经网络的参数权重的，这和我们在有监督学习中的目标值有所区别，在监督学习中这个值在学习时一般都是固定住的。我们使用如下的梯度进行损失函数的微分：
$$
\nabla_{\theta_i}L_i(\theta_i)= \Bbb E_{s,a\sim\rho(·);s'\sim\varepsilon}[(r+\gamma\max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i))\nabla_{\theta_i}Q(s,a;\theta_i)] 
$$
​		比起按上述公式的梯度计算完整的期望值，我们通常使用随机梯度下降方法进行计算以优化损失。如果权重在每一个时间步根据行为分布$\rho$、模拟器环境$\varepsilon$的单个样本替换期望继续宁更新，那么我们就得到了熟悉的Q-learning算法了[26]。

​		注意到这是个model-free的算法，即它通过直接使用模拟器的环境$\varepsilon$抽样解决强化学习的问题，但不需要针对这个模拟器的环境$\varepsilon$进行建模和估计。同时它也是off-policy的，也就是说它会学习一个贪心的策略 $a=max_aQ(s,a;\theta)$同时也会以一定恰当的概率分布对状态空间进行探索。在实践中，我们一般使用贪心$\epsilon$策略，即使用$1-\epsilon$的概率使用决策的动作输出，以概率$\epsilon$选择一个随机的动作。



## 相关工作

​		也许强化学习中的最广为人知的成功故事是IBM 的 Gerald Tesauro 开发的一个玩西洋双陆棋戏的程序——TD-gammom ，也即一个通过强化学习自学如何下棋的神经网络，并获得了超越人类水平的成绩[24]。TD-gammom使用了一个model-free的和Q-learning近似的强化学习算法，使用一个多个的隐藏层的网络用于感知特征以拟合价值函数。

​		然而早期在尝试复现TD-gammom的时候，包括尝试在棋类游戏中使用相同的算法时，围棋和跳棋的效果就不如人意了。这就引来了一阵关于TD-gammom的方法是一个只能在双陆棋得到应用的成功特例的说法，也许是因为投骰子的随机性对状态空间的探索提供了帮助并且也使得它的价值函数尤其平缓顺滑[19]。

更进一步说，model-free的强化学习算法比如Q-learning使用了非线性函数进行动作价值函数的拟合[25]，也许off-policy的学习方式确实会导致Q网络的发散的情况[1]。这些争议与讨论带来的就是强化学习的主要理论工作转移到了线性函数的拟合以获取更好的收敛性和收敛的保障[25]。

​		最近，深度学习和强化学习的结合的趋势再次复苏。深度神经网络已经被应用于评估环境$\varepsilon$，受限玻尔兹曼机也被用于拟合价值函数[21]或者行为决策[9]。除此之外，最近梯度时分差分的方法的提出为Q-learning的发散问题带来了部分条件下的解决的方案。这些方法被证明当使用一个固定的用非线性函数拟合的策略可以保证模型的收敛[14]，或者可以使用一个受限的Q-learning的变种方法[15]用线性函数学习拟合一个控制策略，然而这些方法都没扩展至非线性的控制问题上。

​		也许之前和我们方法最接近的方法就是使用神经拟合Q-learning（NFQ）[20]，它使用公式2的损失函数进行优化，并且使用RPROP算法进行Q网络的参数更新。然而，它在每一个迭代计算的基于使用依据数据集大小的代价进行批量更新，因此我们认为随机梯度更新在每一个迭代过程中的大规模数据集中计算的代价将会是一个非常小的常数。NFQ同样被成功应用于简单的真实世界控制任务，而且也是使用简单的树蕨输入，首先通过自动笔爱你吗其去歇息低维度的任务表征，然后将NFQ应用于该表征中进行控制[12]。与之对比，我们的方法是一个端到端的强化学习过程，直接以视觉作为输入，结果也显示这种方法一样能通过动作价值函数的评价直接学习到特征。Q-learning在之前也被应用于结合了经验回放机制和简单的神经网络[13]中，但其输入也是基于低维度的状态而不是原始的视觉信息输入。

​		使用雅达利2600模拟器作为强化学习平台是参考了Marc G Bellemare[3]等的论文，他们将线性函数拟合和普通视觉信息输入应用在强化学习算法中。然后，实验的结果在后续使用更多特征和tug-of-war散列法随机映射至低维空间中的方法的改进后得到提升[2]。HyperNEAT进化架构[8]也被应用于Atari平台，用来（分别针对每款不同的游戏）形成一个表示游戏策略的神经网络。 当用模拟器的重置机制来与确定性序列做反复对抗训练时，我们发现这些策略可以利用几款Atari游戏中的设计缺陷。



## 深度强化学习

​		计算机视觉和语音识别领域最近取得的一些突破，靠的就是可以在大型训练集上高效地训练深度神经网络。 其中最成功的方法是通过使用基于随机梯度下降的轻量级更新，直接用原始输入进行训练。向深度神经网络输入足够多的数据，这样常常可以学习到比人工生成的特征更好的表征[11]。 这些成功案例为我们的强化学习方法提供了启发。我们的目标是将强化学习算法与深度神经网络对接起来，这里所说的神经网络可以直接学习RGB图像，并通过使用随机梯度更新来有效地处理训练数据。

​		Tesauro的TD-Gammon架构为这种方法提供了一个起点。该架构利用算法与环境的直接交互（或通过自玩，西洋双陆棋）产生的策略性经验样本$(s_t,a_t,r_t,s_{t+1},a_{t+1})$，对价值函数估计网络的参数进行更新。由于该算法在20年前能超越了水平最高的人类西洋双陆棋玩家，所以我们想知道，二十年的硬件改进以及现代深度神经网络架构和可扩展RL算法是否能让强化学习实现重大进展。

​		不同于TD-Gammon和类似的在线方法，我们使用了一种叫做“经验回放”[13]的方法：将智体在每个时间步长$e_t=(s_t,a_t,r_t,s_{t+1})$上的经验储存在数据集$\cal D=e_1,...,e_N,$中，将许多episode的经验汇集至经验池中。在算法进行内部循环时，我们将Q-learning算法更新或小批量更新应用于经验样本$\cal e\sim\cal D$，这些样本是从经验池中随机抽取的。 执行经验回放后，智体根据$\epsilon$贪心策略选择动作决策。由于用任意长度的历史表征作为神经网络的输入较难实现，所以我们的Q函数使用的是函数$\phi$生成的固定长度的历史表征。 算法1给出了完整的算法流程，我们将之称为深度Q-learning算法。



![image-20220526094753105](https://s2.loli.net/2022/05/26/AoOKdq4CNsDGiPu.png)

​		

这种方法和标准的online Q-learning相比有几个优点[23]。受限，每一步的经验都可能会在许多网络权重更新中起作用，数据的有效性大幅增加。其次由于样本之间存在很强的相关性，直接学习连续样本效率很低；随机化样本会破坏这些相关性，减少更新的方差。第三，当学习策略时，当前的参数确定用于训练参数的下一数据样本。举个例子，如果最大化动作是向左移动，训练样本将由左侧的样本主导;如果最大化动作切换到右边，训练数据分布也会切换到右边。很容易看出，这样可能会产生不必要的反馈循环，并且参数也可能会被困在局部最小值，甚至发生严重的发散[25]。通过使用经验回放，在先前状态下的行为分布就会得到变得均匀，这样学习过程就会变得平缓化，并参数也不会出现振荡或发散。需要指出，当使用经验池抽取经验进行学习时off-policy的方法的学习非常有必要（因为我们当前的参数和产生经验使用的参数不同），这就启发了我们使用Q-learning。

​		在实践中，我们的算法仅在经验池中存储最后N个经验元组，并且在执行更新时均匀地从D中随机采样。 这种方法在某些方面有一定局限，因为存储缓冲器并不区分重要的经验；而且由于存储容量N有限，存储缓冲器总是用最新的转移重写记忆。同样，均匀采样使得回放记忆中的所有转移具有相等的重要性。 更复杂的抽样策略可能会强调可以提供最多信息的转换，类似于优先扫除[17] 。



### 预处理和模型结构

​	直接使用原始的Atari框架（128色的210×160像素图像）可能在计算上要求很高，所以我们应用了一个基本的预处理步骤来减少输入维数。进行预处理时，首先将原始帧的RGB图像表示转换成灰度图像，并将其下采样成110×84图像。通过从图像上裁剪一个可以大致捕获到游戏区域的的84×84画面，获得最终的输入表征。 然后进行最后的裁剪，因为我们使用的是Alex等提出的的2D卷积GPU实现[11]，需要方形的输入图像。在该论文的实验中，算法1的函数$\phi$将该预处理过程应用于历史记忆的最后4帧，并将它们叠加以生成Q函数的输入。

​	用神经网络参数化Q函数的方法有许多种。 由于Q函数可以将历史动作对映射到其Q值的标量估计上，所以先前的一些的方法可以将历史和动作作为神经网络的输入[20,12]。 这类架构的主要缺点是需要单独进行一次前向传递来计算每个动作的Q值，这样会导致计算成本与动作数呈正比。 在我们使用的架构中，其中每个可能的动作都对应一个单独的输出单元，神经网络的输入只有状态表征。 输出则对应于输入状态的单个动作的预测Q值。 这类架构的主要优点是只需进行一遍前进传递，就可以计算某一给定状态下所有可能动作的Q值。 

​	下面我们将描述七个Atari游戏所使用的架构。 神经网络的输入是由$\phi$产生的84×84×4图像。 第一个隐藏层用16个步长（stride）为4的8×8卷积核与输入图像进行卷积， 并使用非线性的激活函数ReLU函数[10,18]。 第二个隐层用32个步长为2的4×4卷积核进行卷积，应用同样的非线性激活函数ReLU函数。最后一个隐层为完全连接层，由256个激活函数单元组成。输出层是一个线性的完全连接层，每个有效的动作对应一个输出。在我们研究的游戏中，有效动作的数量在4到18之间变化。我们把用该方法训练的卷积网络称为深度Q网络（DQN）。



## 实验

​		截至目前，我们用7款流行的ATARI游戏进行了试验——Beam Rider、Breakout、Enduro、Pong、Q*bert、Seaquest和Space Invader。在这7款游戏中，我们使用相同的网络架构、学习算法和超参数设置，以证明我们的方法还是能够在不获取特定游戏信息的条件下成功应用于多种游戏中。当在真实且未改动的游戏中对智体进行评估时，我们在训练期间只对游戏的奖励机制作出了一个改变。由于各游戏的得分范围大不相同，我们将所有正奖励都设定为1，将所有负奖励设定为-1，无变化情况设为0奖励。这样的奖励设置可以限制误差范围，便于在多种游戏中使用同一学习率。同时，该奖励机制还会影响智体的表现，因为它无法区分不同大小的奖励。

​		在这些试验中，我们使用的是mini batch大小为32的RMSProp算法。训练中的行为策略为：ϵ-greedy的ϵ在前100万帧从1 线性下降到0.1，然后保持在0.1不变。我们共训练了100万帧，并使用了最近100万帧的回放记忆。

​		在按照前文中的方法玩Atari游戏时，我们还使用了一种简单的跳帧技巧[3]。更确切地说，智体在每隔k帧而不是在每一帧观察并选择动作，在跳过的帧中则重复它的最后一个动作。由于模拟器向前运行一步需要的计算量少于智体选择一个动作的计算量，这种方法可以使智体在不大幅增加运行时间的情况下将游戏次数增加约k倍。除了Space Invader这款游戏，我们在其他游戏中都将k设为4，如果在这款游戏中将k设为4，就会看不见激光，因为跳过的帧与激光闪烁的时长相重叠。将k设定为3就可以看到激光，k值的改变就是不同游戏间的唯一超参数差异。

### 训练和稳定性

​		在监督学习中，通过使用训练集和验证集评估模型，我们可以轻易地追踪模型在训练期间的性能。但是在强化学习中，在训练期间准确评估智体的性能可能会十分困难。如Marc G Bellemare所述[3]，我们的评估指标是在若干游戏中智体在某一episode或游戏中得到的总奖励的平均值。而且我们在训练中周期性地计算该指标。总奖励均值指标往往很嘈杂，因为权重的小小改变可能会导致策略访问的状态的分布发生很大的变化。图2中最左侧的两个线图显示了总奖励均值在游戏Seapuest和Breakout的训练期间是如何变化的。这两个总奖励均值线图确实很嘈杂，给人的印象是学习算法的运行不稳定。

![image-20220526095121441](https://s2.loli.net/2022/05/26/uqRvEcmjGKDf9h5.png)

<small>图2：  左边两图分别展示了训练时Breakout和Seaquest每回合的平均奖励值。数据是使用$\varepsilon=0.05$的$\varepsilon -greedy$策略在运行10000步计算得到的。右图的两图则分别展示了Breakout和Seaquest部分状态下的平均最大的动作价值。大约每三十分钟每一个回合约有50000条抽样数据用于更新</small>

![image-20220526100408551](https://s2.loli.net/2022/05/26/Qr4Mp86NOykVTxL.png)

<small>图3  最左端的图表展示了Seaquest的30帧数据预测的价值函数。三张屏幕截图分别对应图表中的A，B，C三个时刻。</small>

​		右侧的两个线图则较为稳定，指标是指策略的预估动作分值函数Q，该函数的作用是预测在任何给定状态下智体遵循其策略所能获得的惩罚后的奖励。我们在训练开始前运行某一随机策略，收集固定数量的状态，并追踪这些状态的最大预测Q值的均值。从图2最右侧的两个线图可以看出，预测平均Q值的增加趋势要比智体获得的总奖励的均值平缓得多，其余5个游戏的平均Q值的增长曲线也很平缓。除了预测Q值在训练期间有较为平缓的增长，我们在试验中未发现任何发散问题。这表明，除了缺乏理论上的收敛保证，我们的方法能够使用强化学习信合和随机梯度下滑以稳定的方式训练大型神经网络。

### 可视化和价值函数

​		图3给出了游戏Seaquest中学到的价值函数的可视化形式。从图中可以看出，当屏幕左侧出现敌人后预测值出现跳跃（点A）。然后代理想敌人发射鱼雷，当鱼雷快要集中敌人时预测值达到最高点（点B）。最后当敌人消失后预测值差不多恢复到原始值（点C）。图3表明我们的方法能够在较为复杂一系列的事件中学习价值函数的变化方式。

### 主要的评估

​		我们将我们的结果与Marc中提出的方法进行了比较。该方法被称为“Sarsa”，Sarsa算法借助为Atari任务人工设计的多个特征集来学习线性策略，我们在[3,,4]中给出了表现最佳的特征集的得分[3]。Contingency算法的基本思路和Sarsa法相同，但是该方法可以通过学习智体控制范围内的屏幕区域的表征，来增强特征集[4]。需要指出，这两种方法都通过背景差分法吸纳了大量关于计算机视觉问题的知识，并将128种颜色中的每种颜色都作为一个单独的通道。由于许多Atari游戏中每种类型的目标所用的颜色通常都各不相同，将每种颜色作为一个单独的通道，这种做法类似于生成一个单独的二元映射，对每种目标类型进行编码。相比之下，我们的代理只接收原始RGB屏幕截图输入，并且必须学习自行检测目标。

​		除了给出学习代理（learned agents）的得分，我们还给出了人类专业游戏玩家的得分，以及一种均匀地随机选择动作的策略。人类玩家的表现将表示为玩了两小时游戏后得到的奖励中值。需要指出，我们给出的人类玩家的得分要比Bellemare等人[3]论文中给出的得分高得多。至于学习方法，我们遵循的是Bellemare等人的论文==[3,5]==中使用的评估策略，并且我们还通过将$\epsilon$设定为0.05运行 ε 贪心策略来获得固定步数的平均得分。表1的前5行给出了所有游戏的各游戏平均得分。在这7款游戏中，我们的方法（标记为DQN）虽然没有吸纳任何关于输入的先验知识，结果还是大幅超越了其他学习方法.。

​		我们同样将文献[8]中的策略搜索方法囊括进表格的后三行进行比较，并做出了两组使用该方法的结果。HNeat Best 的得分反映了通过使用人工标注的目标检测法现实游戏物体位置和类型.HNeat Pixel的得分反映的是使用特定的8种颜色通道代表雅达利游戏模拟器的特定物体类型。这两种方法依赖于确定的状态序列且不存在随机的扰动，因此与其他方法对比时对比的是单个会和下的最佳表现。相反的，我们的算法是在贪心$\epsilon$策略下进行评估的，因此必须对所有可能的情况进行归一化。然而，我们发现在所有展示的游戏中，除了Space Invaders这款游戏不单是我们算法的最优表现还是我们的平均表现都能达到一个相对较好的效果。

![image-20220526100647570](https://s2.loli.net/2022/05/26/aZeD57p8KGwxCWk.png)

<small>表1  上端的表格对比了使用贪心值为0.0.5的$\varepsilon-greedy$的不同学习方式平均总奖励。下端的表格展示了HNeat和DQN在单个回合中最佳表现分数。HNeat做出确定性策略与DQN使用$\varepsilon-greedy$的效果相同。</small>

​		最终我们发现我们的方法在Breakout，Enduro和Pong这三款游戏中达到了超越人类专家水平的水准，并且在Beam Rider种取得了与人类水平相近的成绩。Q*bert，Seaquest，Space Invaders这三款游戏我们依然与人类专家水平相比有较大的差距，但它充满了挑战性，因为他们需要网络找到一个从更长远的时间层面上考察的策略。

## 结论

​		本文介绍了一种新的强化学习的深度学习模型，并通过在雅达利2600游戏中证明了其对困难游戏的控制能力，而它仅需要原始的像素信息作为输入。我们同时也展示了一种结合了随机批量更新和经验回放机制Q-learning的变种算法来降低训练强化学习中的深度神经网络难度。我们的方法在七个测试的游戏中的六个取得SOTA成果，并且没有调整过超参数和网络模型结构。

## 参考文献

[1] Leemon Baird. Residual algorithms: Reinforcement learning with function approximation. In *Proceedings of the 12th International Conference on Machine Learning (ICML 1995)*, pages 30–37. Morgan Kaufmann, 1995.

[2] Marc Bellemare, Joel Veness, and Michael Bowling. Sketch-based linear value function ap- proximation. In *Advances in Neural Information Processing Systems 25*, pages 2222–2230, 2012.

[3] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. *Journal of Artificial Intelligence* *Research*, 47:253–279, 2013.

[4] Marc G Bellemare, Joel Veness, and Michael Bowling. Investigating contingency awareness using atari 2600 games. In *AAAI*, 2012.

[5] Marc G. Bellemare, Joel Veness, and Michael Bowling. Bayesian learning of recursively fac- tored environments. In *Proceedings of the Thirtieth International Conference on Machine* *Learning* *(ICML* *2013)*, pages 1211–1219, 2013.



[6] George E. Dahl, Dong Yu, Li Deng, and Alex Acero. Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. *Audio, Speech, and Language Pro-* *cessing,* *IEEE* *Transactions* *on*, 20(1):30 –42, January 2012.

[7] Alex Graves, Abdel-rahman Mohamed, and Geoffrey E. Hinton. Speech recognition with deep recurrent neural networks. In *Proc.* *ICASSP*, 2013.

[8] Matthew Hausknecht, Risto Miikkulainen, and Peter Stone. A neuro-evolution approach to general atari game playing. 2013.

[9] Nicolas Heess, David Silver, and Yee Whye Teh. Actor-critic reinforcement learning with energy-based policies. In *European* *Workshop* *on* *Reinforcement* *Learning*, page 43, 2012.

[10] Kevin Jarrett, Koray Kavukcuoglu, MarcAurelio Ranzato, and Yann LeCun. What is the best multi-stage architecture for object recognition? In *Proc. International Conference on Com-* *puter* *Vision* *and* *Pattern* *Recognition* *(CVPR* *2009)*, pages 2146–2153. IEEE, 2009.

[11] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet classification with deep con- volutional neural networks. In *Advances in Neural Information Processing Systems 25*, pages 1106–1114, 2012.

[12] Sascha Lange and Martin Riedmiller. Deep auto-encoder neural networks in reinforcement learning. In *Neural Networks (IJCNN), The 2010 International Joint Conference on*, pages 1–8. IEEE, 2010.

[13] Long-Ji Lin. Reinforcement learning for robots using neural networks. Technical report, DTIC Document, 1993.

[14] Hamid Maei, Csaba Szepesvari, Shalabh Bhatnagar, Doina Precup, David Silver, and Rich Sutton. Convergent Temporal-Difference Learning with Arbitrary Smooth Function Approxi- mation. In *Advances* *in* *Neural* *Information* *Processing* *Systems* *22*, pages 1204–1212, 2009.

[15] Hamid Maei, Csaba Szepesva´ri, Shalabh Bhatnagar, and Richard S. Sutton. Toward off-policy learning control with function approximation. In *Proceedings of the 27th International Con-* *ference* *on* *Machine* *Learning* *(ICML* *2010)*, pages 719–726, 2010.

[16] Volodymyr Mnih. *Machine Learning for Aerial Image Labeling*. PhD thesis, University of Toronto, 2013.

[17] Andrew Moore and Chris Atkeson. Prioritized sweeping: Reinforcement learning with less data and less real time. *Machine* *Learning*, 13:103–130, 1993.

[18] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann ma- chines. In *Proceedings of the 27th International Conference on Machine Learning (ICML* *2010)*, pages 807–814, 2010.

[19] Jordan B. Pollack and Alan D. Blair. Why did td-gammon work. In *Advances in Neural* *Information* *Processing* *Systems* *9*, pages 10–16, 1996.

[20] Martin Riedmiller. Neural fitted q iteration–first experiences with a data efficient neural re- inforcement learning method. In *Machine Learning: ECML 2005*, pages 317–328. Springer, 2005.

[21] Brian Sallans and Geoffrey E. Hinton. Reinforcement learning with factored states and actions.

*Journal* *of* *Machine* *Learning* *Research*, 5:1063–1088, 2004.

[22] Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and Yann LeCun. Pedestrian de- tection with unsupervised multi-stage feature learning. In *Proc. International Conference on* *Computer* *Vision* *and* *Pattern* *Recognition* *(CVPR* *2013)*. IEEE, 2013.

[23] Richard Sutton and Andrew Barto. *Reinforcement Learning:* *An Introduction*. MIT Press, 1998.

[24] Gerald Tesauro. Temporal difference learning and td-gammon. *Communications of the ACM*, 38(3):58–68, 1995.

[25] John N Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. *Automatic* *Control,* *IEEE* *Transactions* *on*, 42(5):674–690, 1997.

[26] Christopher JCH Watkins and Peter Dayan. Q-learning. *Machine learning*, 8(3-4):279–292, 1992.
