# Novel-Poisoning-Attacks-on-Decentralized-Federated-Learning
Master Project (Master Projekt) for Yunlong Li and Runxi Cui


keyword:poisoning attack 
【1】https://ieeexplore.ieee.org/document/9767718
【2】https://ieeexplore.ieee.org/document/10105919
【3】https://www.sciencedirect.com/science/article/abs/pii/S0167404823002912?via%3Dihub
【4】https://ieeexplore.ieee.org/document/9760102
【5】https://ojs.aaai.org/index.php/AAAI/article/view/26083
【6】https://ieeexplore.ieee.org/document/10327979
main【7】https://dl.acm.org/doi/10.1145/3534678.3539119

Defending：
【1】https://ojs.aaai.org/index.php/AAAI/article/view/26271
【2】https://dl.acm.org/doi/10.1145/3576915.3623193

ideas:
1.在攻击的时候实时调整策略，实时调整数据效果以影响其loss和跳过其识别策略，强化学习加权方法，强化学习加权的时候来影响联邦学习的每轮参数
2.在联邦学习的过程中，污染数据使得其梯度下降找鞍点的过程中让我们提供一个假鞍点给他，因此我们在同时利用数据训练的过程中，需要比他快找到鞍点并进行调整
