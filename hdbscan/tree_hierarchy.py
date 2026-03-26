import numpy as np
from typing import List, Tuple, Dict, Set

class TreeHierarchy:
    """
    在 Driver 端处理全局 MST，构建层次树，凝缩树并提取稳定簇。
    """
    def __init__(self, min_cluster_size: int):
        self.min_cluster_size = min_cluster_size

    def build_and_extract(self, global_mst: List[Tuple[int, int, float]]) -> Dict[int, int]:
        """
        根据全局 MST 提取最终簇标签。
        
        :param global_mst: 全局 MST 边列表 [(u, v, weight), ...]
        :return: 簇标签字典 {point_id: cluster_id}，噪声点不在字典中或值为 -1
        """
        if not global_mst:
            return {}
            
        # 1. 按权重降序排列 (模拟不断降低 lambda 阈值，砍断长边)
        sorted_edges = sorted(global_mst, key=lambda x: x[2], reverse=True)
        
        # 获取所有节点
        nodes = set()
        for u, v, w in sorted_edges:
            nodes.add(u)
            nodes.add(v)
            
        # 初始状态：整个图是一个大簇
        # 为了简化实现，这里我们使用自底向上的 Single Linkage 聚类 (类似于 Scipy 的 linkage)
        # 然后再自顶向下计算 Stability。
        # 这是一个简化的 HDBSCAN 树凝缩算法，适合教学和理解。
        
        # 使用 scipy.cluster.hierarchy 提供的功能可以大大简化代码并保证鲁棒性
        # 但为了满足 "手动实现内部逻辑" 的要求，这里手写一个简化的并查集聚类过程
        
        # 初始化每个点为一个独立的簇
        parent = {n: n for n in nodes}
        size = {n: 1 for n in nodes}
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        # 记录合并过程 (类似 linkage matrix)
        # 格式: (cluster1, cluster2, distance, new_size)
        # 我们需要按距离从小到大合并
        merge_edges = sorted(global_mst, key=lambda x: x[2])
        
        next_cluster_id = max(nodes) + 1 if nodes else 0
        
        # 记录每个簇的诞生距离 (用于计算 stability)
        birth_distance = {n: 0.0 for n in nodes}
        
        # 记录树结构 {parent_id: (child1_id, child2_id)}
        tree = {}
        
        for u, v, w in merge_edges:
            root_u = find(u)
            root_v = find(v)
            
            if root_u != root_v:
                new_size = size[root_u] + size[root_v]
                new_id = next_cluster_id
                next_cluster_id += 1
                
                parent[root_u] = new_id
                parent[root_v] = new_id
                parent[new_id] = new_id
                
                size[new_id] = new_size
                birth_distance[new_id] = w
                
                tree[new_id] = (root_u, root_v)
                
        root_node = next_cluster_id - 1
        
        # 2. 树凝缩 (Tree Condensation)
        
        # 记录每个节点被视为"真正簇"时的代表节点（即忽略了噪声脱落的节点）
        # condensed_tree 记录: {node_id: {"parent": parent_id, "child_size": size, "lambda_val": 1/dist}}
        
        condensed_tree = []
        
        # 为了更准确地实现 HDBSCAN 树凝缩，我们需要自顶向下遍历
        # 此时的 tree 是一个二叉树 {node_id: (left_id, right_id)}
        
        def condense_tree():
            # 记录当前有效的簇ID，和它在 BFS/DFS 过程中的"存活节点"
            # 队列中保存: (current_node, current_cluster_id)
            queue = [(root_node, root_node)]
            
            while queue:
                curr_node, curr_cluster = queue.pop(0)
                
                if curr_node not in tree:
                    continue # 叶子节点
                    
                left, right = tree[curr_node]
                
                left_is_cluster = size[left] >= self.min_cluster_size
                right_is_cluster = size[right] >= self.min_cluster_size
                
                dist = birth_distance[curr_node]
                lambda_val = 1.0 / dist if dist > 1e-6 else 1e6
                
                if left_is_cluster and right_is_cluster:
                    # 真正的分裂：当前簇结束，产生两个新簇
                    # 记录分裂事件
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": left,
                        "child_size": size[left],
                        "lambda_val": lambda_val
                    })
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": right,
                        "child_size": size[right],
                        "lambda_val": lambda_val
                    })
                    
                    # 继续向下
                    queue.append((left, left))
                    queue.append((right, right))
                    
                elif left_is_cluster:
                    # 右边是噪声脱落
                    # 左边继承当前的 cluster_id
                    queue.append((left, curr_cluster))
                    # 记录脱落点
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": right,
                        "child_size": size[right],
                        "lambda_val": lambda_val
                    })
                elif right_is_cluster:
                    # 左边是噪声脱落
                    queue.append((right, curr_cluster))
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": left,
                        "child_size": size[left],
                        "lambda_val": lambda_val
                    })
                else:
                    # 两个都小于 min_cluster_size，全部作为噪声脱落
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": left,
                        "child_size": size[left],
                        "lambda_val": lambda_val
                    })
                    condensed_tree.append({
                        "parent": curr_cluster,
                        "child": right,
                        "child_size": size[right],
                        "lambda_val": lambda_val
                    })
                    
        condense_tree()
        
        # 计算 Stability
        # 1. 计算每个簇的 lambda_birth
        cluster_birth = {root_node: 0.0} # 根节点诞生于 lambda=0
        for edge in condensed_tree:
            # edge["child"] 是新的簇 ID
            if edge["child"] in tree and size[edge["child"]] >= self.min_cluster_size:
                # 只有真正分裂出来的簇才记录 lambda_birth
                if edge["child"] not in cluster_birth:
                    cluster_birth[edge["child"]] = edge["lambda_val"]
                    
        # 2. 计算每个簇的 stability
        # stability = sum(lambda_death - lambda_birth)
        stability = {}
        for edge in condensed_tree:
            parent_cluster = edge["parent"]
            lambda_death = edge["lambda_val"]
            l_birth = cluster_birth.get(parent_cluster, 0.0)
            
            # 每个脱落的子节点（或分裂出去的簇）在脱落前，都为父簇贡献了 (lambda_death - lambda_birth) * size
            contribution = (lambda_death - l_birth) * edge["child_size"]
            stability[parent_cluster] = stability.get(parent_cluster, 0.0) + contribution
            
        # 3. 提取最优簇 (Cluster Extraction)
        # 自底向上提取，判断是否保留子簇
        
        # 找出所有是簇的节点
        all_clusters = list(cluster_birth.keys())
        # 按节点大小升序（自底向上）
        all_clusters.sort(key=lambda x: size[x])
        
        selected_clusters = set()
        
        # 为了快速找到一个簇分裂出的子簇，建立映射
        children_clusters = {}
        for edge in condensed_tree:
            p = edge["parent"]
            c = edge["child"]
            if c in cluster_birth: # c 也是一个真正的簇
                if p not in children_clusters:
                    children_clusters[p] = []
                children_clusters[p].append(c)
                
        # 初始化 propagated_stability
        propagated_stability = {c: stability.get(c, 0.0) for c in all_clusters}
        
        for cluster in all_clusters:
            # 这是一个叶子簇（没有再分裂成更小的有效簇）
            if cluster not in children_clusters:
                pass # 保持自身的 stability
            else:
                # 它有子簇，比较自身的 stability 和子簇的 stability 之和
                child_stab_sum = sum([propagated_stability[c] for c in children_clusters[cluster]])
                
                if stability.get(cluster, 0.0) > child_stab_sum:
                    # 自身更稳定，丢弃子簇
                    propagated_stability[cluster] = stability.get(cluster, 0.0)
                else:
                    # 子簇更稳定，当前簇的 stability 等于子簇之和
                    propagated_stability[cluster] = child_stab_sum
                    
        # 自顶向下收集被选择的簇
        def collect_clusters(cluster):
            if cluster not in children_clusters:
                selected_clusters.add(cluster)
                return
                
            child_stab_sum = sum([propagated_stability[c] for c in children_clusters[cluster]])
            
            if stability.get(cluster, 0.0) > child_stab_sum:
                selected_clusters.add(cluster)
                # 不再向下
            else:
                for c in children_clusters[cluster]:
                    collect_clusters(c)
                    
        collect_clusters(root_node)
        
        # 4. 映射回原始数据点
        labels = {}
        
        # 建立快速查找表，从叶子节点往上找，看它属于哪个 selected_cluster
        # 如果找到 root_node 都还没找到，说明它是噪声 (-1)
        
        def get_cluster_label(leaf_id):
            curr = leaf_id
            while curr != root_node:
                # 寻找 curr 的父 cluster (在 condensed tree 中)
                # 这稍微有点复杂，我们可以用之前的 tree 来向上找
                if curr in selected_clusters:
                    return curr
                curr = parent.get(curr, curr)
                if curr == parent.get(curr, curr): # 避免死循环
                    break
            if root_node in selected_clusters:
                return root_node
            return -1

        # 利用最初始建立的树向上溯源
        # 实际上在 condensed_tree 阶段，每个点脱落时的 parent 就是它最后归属的簇
        leaf_to_cluster = {}
        for edge in condensed_tree:
            # 这是一个单独脱落的叶子/噪声集合
            if edge["child"] not in cluster_birth:
                # 需要递归地把这个 child 树里所有的叶子节点都标记为 edge["parent"]
                def mark_leaves(node):
                    if node not in tree:
                        leaf_to_cluster[node] = edge["parent"]
                    else:
                        mark_leaves(tree[node][0])
                        mark_leaves(tree[node][1])
                mark_leaves(edge["child"])
                
        for leaf, c_id in leaf_to_cluster.items():
            # 检查这个 c_id 的祖先链里，哪一个被 selected_clusters 选中了
            final_c = -1
            curr = c_id
            # 需要一个快速向上找父簇的字典
            parent_cluster_map = {}
            for e in condensed_tree:
                if e["child"] in cluster_birth:
                    parent_cluster_map[e["child"]] = e["parent"]
                    
            while curr in parent_cluster_map:
                if curr in selected_clusters:
                    final_c = curr
                    break
                curr = parent_cluster_map[curr]
                
            if final_c == -1 and root_node in selected_clusters:
                final_c = root_node
                
            labels[leaf] = final_c

        # 重新编号簇 ID (从 0 开始，忽略 -1)
        unique_clusters = [c for c in set(labels.values()) if c != -1]
        cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        cluster_map[-1] = -1
        
        final_labels = {pid: cluster_map[cid] for pid, cid in labels.items() if pid in nodes}
        return final_labels
