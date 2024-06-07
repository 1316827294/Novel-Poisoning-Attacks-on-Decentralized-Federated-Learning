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
main【8】https://dl.acm.org/doi/abs/10.1609/aaai.v37i4.25611

Defending：
【1】https://ojs.aaai.org/index.php/AAAI/article/view/26271
【2】https://dl.acm.org/doi/10.1145/3576915.3623193

ideas:
1.在攻击的时候实时调整策略，实时调整数据效果以影响其loss和跳过其识别策略，强化学习加权方法，强化学习加权的时候来影响联邦学习的每轮参数
2.在联邦学习的过程中，污染数据使得其梯度下降找鞍点的过程中让我们提供一个假鞍点给他，因此我们在同时利用数据训练的过程中，需要比他快找到鞍点并进行调整



baseline:
【1】https://github.com/Jenson66/Poisoning-Attack-on-FL/tree/main
【2】https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/tree/main 这个点结果有点假
[3]https://github.com/lishenghui/blades/tree/master 这个也可以跑出结果来


最像我们的工作：https://github.com/harshkasyap/FL-Hyperdeimensional-PoisoningAttack/blob/main/libs/sim.py对应的论文https://ieeexplore.ieee.org/document/9751167/metrics#metrics

def our_attack_descent(all_updates, model_re, n_attackers, dev_type='unit_vec', learning_rate=0.01, threshold_diff=1e-5):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.tensor([10.0]).float().cuda()

    while True:
        mal_update = (model_re - lamda * deviation)

        # 使用 all_updates 的中位数来更新 lamda
        median_val = torch.median(all_updates)
        lamda = lamda*0.5 - learning_rate * (lamda - median_val)
        # lamda *= 0.5
        # 计算更新后的损失，这里仍然使用 MSE
        loss = F.mse_loss(all_updates, mal_update.unsqueeze(0).expand_as(all_updates))

        if loss.item() < threshold_diff:
            break
        

    mal_update = (model_re - lamda * deviation)
    return mal_update.detach()
