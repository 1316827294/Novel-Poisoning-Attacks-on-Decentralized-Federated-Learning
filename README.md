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
【2】https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/tree/main这个点结果有点假
[3]https://github.com/lishenghui/blades/tree/master这个也可以跑出结果来


def our_attack_mkrum(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    # all_updates是需要攻击的训练后的梯度
    # model_re是计算需要攻击的用户提交的梯度更新的平均值。
    # n_attackers是2
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)#所有用户更新的标准偏差。标准偏差可以帮助识别数据中的异常模式，从而在联邦学习中采取相应的防御措施。

    lamda = torch.Tensor([3.0]).cuda()

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    return mal_update
