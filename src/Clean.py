import csv
import json
import os
import time

from AnalyticsCache.cleaner_associations_cycle import associations_classier, discover_cleaner_associations, \
    PreParamClassier
from AnalyticsCache.get_cleaner_excute_info import sort_opinfo_by_weights, generate_plantuml_corrected, \
    getSingle_opinfo, cleaner_grouping
from AnalyticsCache.get_planuml_graph import PlantUML
from AnalyticsCache.handle_rules import getOpWeights, transformRulesToSpark, transformKnowledgeToSpark
from CoreSetSample.mapping_samplify import Generate_Sample
from SampleScrubber.cleaner_model import NOOP
from SampleScrubber.param_selector import customclean


def _listify(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _cleaner_id(cleaner):
    msg = getattr(cleaner, "msg", None)
    if msg:
        return str(msg)
    domain = ",".join(_listify(getattr(cleaner, "domain", None)))
    source = ",".join(_listify(getattr(cleaner, "source", None)))
    target = ",".join(_listify(getattr(cleaner, "target", None)))
    return f"{cleaner.__class__.__name__}:{source}->{target}:{domain}"


def _operator_record(cleaner, phase, order, level=None, node=None, sample_rows=None, rule_count=None):
    return {
        "operator_id": _cleaner_id(cleaner),
        "operator_class": cleaner.__class__.__name__,
        "phase": phase,
        "order": order,
        "level": level if level is not None else "",
        "node": str(node) if node is not None else "",
        "source_attributes": "|".join(_listify(getattr(cleaner, "source", None))),
        "target_attributes": "|".join(_listify(getattr(cleaner, "target", None))),
        "domain": "|".join(_listify(getattr(cleaner, "domain", None))),
        "sample_rows": sample_rows if sample_rows is not None else "",
        "rule_count": rule_count if rule_count is not None else "",
    }


def _rule_record(rule, phase, operator_id, order, level=None, node=None):
    predicate = getattr(rule, "predicate", ([], [], []))
    if predicate is None:
        predicate = ([], [], [])
    predicate_attrs = predicate[0] if hasattr(predicate, "__len__") and len(predicate) > 0 else []
    predicate_values = predicate[1] if hasattr(predicate, "__len__") and len(predicate) > 1 else []
    predicate_rows = predicate[2] if hasattr(predicate, "__len__") and len(predicate) > 2 else []
    return {
        "rule_id": order,
        "phase": phase,
        "operator_id": str(operator_id),
        "level": level if level is not None else "",
        "node": str(node) if node is not None else "",
        "target_attribute": str(getattr(rule, "domain", "")),
        "repair_value": str(getattr(rule, "repairvalue", "")),
        "predicate_attributes": "|".join(_listify(predicate_attrs)),
        "predicate_value_count": len(predicate_values) if hasattr(predicate_values, "__len__") else "",
        "predicate_row_count": len(predicate_rows) if hasattr(predicate_rows, "__len__") else "",
        "opinformation": str(getattr(rule, "opinformation", "")),
    }


def _append_rule_records(trace, edit_rule_list, phase, operator_id, level=None, node=None):
    for rule in edit_rule_list or []:
        trace["rules"].append(
            _rule_record(rule, phase, operator_id, len(trace["rules"]), level=level, node=node)
        )


def _append_weight_records(trace, grouped_opinfo, group_weights, sorted_dict):
    sorted_lookup = {str(key): value for key, value in (sorted_dict or {}).items()}
    for level_index, level in (grouped_opinfo or {}).items():
        for target_node, cleaners in level.items():
            weights = (group_weights or {}).get(target_node, [])
            for order, cleaner in enumerate(cleaners):
                weight = weights[order] if order < len(weights) else ""
                trace["weights"].append({
                    "level": level_index,
                    "node": str(target_node),
                    "operator_id": _cleaner_id(cleaner),
                    "order": order,
                    "weight": weight,
                    "sorted_group": str(sorted_lookup.get(str(target_node), "")),
                })


def _write_trace_files(trace_path, trace):
    if not trace_path:
        return
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    base_dir = os.path.dirname(trace_path)
    _write_csv(os.path.join(base_dir, "operator_trace.csv"), trace.get("operators", []))
    _write_csv(os.path.join(base_dir, "operation_rule_trace.csv"), trace.get("rules", []))
    _write_csv(os.path.join(base_dir, "operator_weight_trace.csv"), trace.get("weights", []))


def _write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def CleanonLocal(spark, cleanners, data, table_path, batch_size=500, single_max=30000):
    current = 'Tmp1'
    print("初始化清洗器和分析依赖关系...")
    Edges, singles, multis = PreParamClassier(cleanners)

    print("处理单属性清洗算子：")
    for single in singles:
        print(f"处理单属性清洗算子：{single}")
        # 获取当前单属性的域 (例如: "domain" 属性表示当前算子的目标属性)
        target_column = single.domain

        # 从 Spark DataFrame 中选择目标列，并将其转换为 pandas DataFrame
        sample_df = data.select(target_column).distinct().toPandas()

        if sample_df.empty:
            print(f"属性 {target_column} 的数据为空，跳过清洗。")
            continue

        print(f"抽样属性 {target_column} 的数据块大小: {sample_df.shape[0]}")

        # 调用 customclean 进行清洗操作
        preCleaners = [single]
        _, output, editRuleList, _ = customclean(sample_df, precleaners=preCleaners)

        if editRuleList:
            # 如果有清洗操作规则，应用到 Spark DataFrame 上
            print(f"应用 {target_column} 的清洗规则，编辑规则数量: {len(editRuleList)}")
            data, _ = transformRulesToSpark(editRuleList, data, batch_size=batch_size)
        else:
            print(f"属性 {target_column} 没有可应用的清洗规则。")

    source_sets, target_sets, explain, processing_order, plt = discover_cleaner_associations(Edges)
    print("执行层级和目标模型分类...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("执行层级 (并行组):", levels)
    print("目标模型分类:", models)

    editRuleDict = {}  # 用于统计溯源权重
    editRules = NOOP()  # 初始化无操作规则
    nextClean = True

    level_clean_status = {}
    clean_attempts = {}  # 新增：记录每个节点的清洗次数

    while nextClean:
        nextClean = False

        for level_index, level in enumerate(nodes):
            # 如果该层级还未记录过清洗状态，则初始化
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in clean_attempts:
                clean_attempts[level_index] = {node: 0 for node in level}

            print(f"\n处理第 {level_index + 1} 层级, 包含节点: {level}")
            editRuleLists = []  # 获取当前层级的清洗操作列表

            for node in level:
                # 只处理该层级中挖掘不到操作的节点
                if level_clean_status[level_index][node]:
                    print(f"  跳过已清洗结束的节点: {node}")
                    continue

                clean_attempts[level_index][node] += 1  # 增加清洗次数

                if clean_attempts[level_index][node] > 3:  # 超过3次则标记为完成
                    print(f"  当前节点 {node} 清洗次数超过限制，标记为完成")
                    level_clean_status[level_index][node] = True
                    continue

                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  当前节点: {node} 存在可用清洗信号")
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)
                    tset = list(tset)
                    print(f"  抽样处理：源属性 {sset} -> 目标属性 {node}")
                    sample_Block_df, error_threshold = Generate_Sample(data, sset, tset, models=models[node])
                    sample_df = sample_Block_df.toPandas()
                    print(f"数据块大小: {sample_df.shape[0]}")
                    if sample_df.shape[0] < single_max:
                        preCleaners = [single for single in singles if single.domain in (sset + [node])]
                        _, output, _, _ = customclean(sample_df, precleaners=preCleaners)
                        # 对块内算子加权计算
                        blockModels = None
                        weights = [1] * len(models[node])
                        for model, weight in zip(models[node], weights):
                            if blockModels is None:
                                blockModels = model * weight
                            else:
                                blockModels += model * weight
                        partition = [list(model.source) for model in models[node]]
                        print('分块属性')
                        print(partition)
                        editRule1, output, editRuleList, _ = customclean(output, cleaners=[blockModels],
                                                                         partition=partition,
                                                                         error_threshold=error_threshold)
                        editRule *= editRule1
                    else:
                        print(f"数据块大小超过阈值，逐个执行模型")
                        for m in models[node]:
                            print(f"处理模型: {m}")
                            # 提取当前模型的源和目标属性
                            sset = list(m.source)
                            tset = list(m.target)
                            sample_Block_df, error_threshold = Generate_Sample(data, sset, tset, models=[m],
                                                                               single_max=single_max)
                            sample_df = sample_Block_df.toPandas()
                            if sample_df.shape[0] == 0:
                                print(f"模型 {m} 处理后无数据")
                                continue
                            preCleaners = [single for single in singles if single.domain in (sset + [node])]
                            _, output, _, _ = customclean(sample_df, precleaners=preCleaners)
                            blockModels = m  # 直接使用单个模型
                            partition = [list(m.source)]
                            editRule1, output, editRuleList1, _ = customclean(output, cleaners=[blockModels],
                                                                              partition=partition,
                                                                              error_threshold=error_threshold)
                            editRule *= editRule1
                            editRuleList.extend(editRuleList1)
                else:
                    print(f"  当前节点: {node} 无可用清洗信号")

                editRuleLists.extend(editRuleList)
                editRules *= editRule

                if len(editRuleList) == 0:
                    print(f"  当前节点: {node} 无清洗操作")
                    level_clean_status[level_index][node] = True
                else:
                    print(f"  当前节点: {node} 有{len(editRuleList)}个清洗操作")
                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # 假设之前不是列表，现在重置为新列表
                else:
                    editRuleDict[str(node)] = editRuleList  # 键不存在，创建新键并赋值为新列表

            print(f"{level}的清洗完成")
            print("\n本层累积的编辑规则数:", str(len(editRuleLists)))

            if editRuleLists:  # 有编辑规则
                nextClean = True
                # data, _ = transformRulesToSpark(editRuleLists, data, batch_size=batch_size)
                # print("正在写入数据")
                # path = table_path + 'Tmp'
                # data.coalesce(1).write.mode('overwrite').csv(path, header=True)
                # print("写入数据大小：" + str(data.count()))
                # os.system('sync')  # 刷新缓存
                # data.unpersist()
                # spark.catalog.clearCache()
                # print("重新读取最新清洗的数据", str(path))
                # data = spark.read.csv(path, header=True, inferSchema=True)
                # print("读入数据大小" + str(data.count()))
                # data.persist()
                # print(f"数据清洗后变动，重新进行节点:{level}的清洗")
                while True:
                    # 按批次处理规则
                    data, remaining_rules = transformRulesToSpark(editRuleLists, data, batch_size=batch_size, maxhandle=3000)
                    print("正在写入数据...")
                    # 定义临时路径
                    if current == 'Tmp1':
                        path = table_path + 'Tmp2'
                        current = 'Tmp2'
                    else:
                        path = table_path + 'Tmp1'
                        current = 'Tmp1'
                    # 写入数据到文件系统
                    data.coalesce(1).write.mode('overwrite').csv(path, header=True)
                    print("写入数据大小：" + str(data.count()))
                    os.system('sync')  # 刷新缓存
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("重新读取最新清洗的数据", str(path))
                    data = spark.read.csv(path, header=True, inferSchema=True)
                    print("读入数据大小" + str(data.count()))
                    data.persist()
                    # print(f"数据清洗后变动，重新进行节点:{level}的清洗")

                    # 如果没有剩余规则，退出循环
                    if not remaining_rules:
                        print("所有规则处理完成")
                        break

                    print(f"有剩余的清洗操作没有应用，继续处理 {len(remaining_rules)} 个清洗操作")
                    editRuleLists = remaining_rules  # 使用剩余规则继续处理

                print(f"数据清洗后变动，重新进行节点 {level} 的清洗")
                break
            else:
                print("本层没有可以进行的清洗操作，进行下一层")
    print("\n清洗规则溯源分析:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    single_opinfo = getSingle_opinfo(singles)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print("\n清洗流程展示:")
    sorted_dict = sort_opinfo_by_weights(grouped_opinfo, group_weights)

    OpPlantuml = generate_plantuml_corrected(single_opinfo, grouped_opinfo, sorted_dict)
    PlantUML().process_str(OpPlantuml, directory=table_path)
    plt.savefig(rf"{table_path}/acplt.png", format="png")
    print("关联图存储到:", rf"{table_path}/acplt.png")
    print("\n绘制清洗流程图:", "CleanerPlantuml.svg")

    return data


def CleanonCloud(spark, cleaners, data, table_name, database_name, batch_size=300, single_max=20000, maxhandle=1000):
    # 初始化和分析清洗器
    print("初始化清洗器和分析依赖关系...")
    Edges, singles, multis = PreParamClassier(cleaners)
    source_sets, target_sets, explain, processing_order = discover_cleaner_associations(Edges)
    print("执行层级和目标模型分类...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("执行层级 (并行组):", levels)
    print("目标模型分类:", models)

    editRuleDict = {}  # 用于统计溯源权重
    editRules = NOOP()  # 初始化无操作规则
    nextClean = True
    tmp_table1 = f"{database_name}.{table_name}_tmp1"
    tmp_table2 = f"{database_name}.{table_name}_tmp2"
    current_tmp_table = tmp_table1
    level_clean_status = {}  # 用于记录每个层级的节点清洗状态
    level_knowledge_status = {}  # 用于记录知识库是否已应用

    while nextClean:
        nextClean = False
        for level_index, level in enumerate(nodes):
            # 如果该层级还未记录过清洗状态，则初始化
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in level_knowledge_status:
                level_knowledge_status[level_index] = {node: False for node in level}

            print(f"\n处理第 {level_index + 1} 层级, 包含节点: {level}")
            editRuleLists = []  # 获取当前层级的清洗操作列表

            for node in level:
                # 只处理该层级中未完成清洗的节点
                if level_clean_status[level_index][node]:
                    print(f"  跳过已清洗结束的节点: {node}")
                    continue

                # 初始化当前节点的清洗操作
                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  当前节点: {node} 存在可用清洗信号")

                    # 检查知识库是否已应用
                    if not level_knowledge_status[level_index][node]:
                        # **在此处添加 transformKnowledgeToSpark 调用**
                        for cleaner in models[node]:
                            data = transformKnowledgeToSpark(spark, database_name, table_name, data, cleaner)

                        # **写入和读取数据**
                        print("正在写入数据...")
                        next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                        data.write.mode('overwrite').saveAsTable(next_tmp_table)
                        print(f"写入数据大小：{data.count()}")
                        os.system('sync')  # 刷新缓存
                        data.unpersist()
                        spark.catalog.clearCache()
                        print("重新读取最新清洗的数据...")
                        data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                        print(f"读入数据大小：{data.count()}")
                        data.persist()
                        current_tmp_table = next_tmp_table
                        level_knowledge_status[level_index][node] = True

                    # 定义一个空集合
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)  # 转换为列表
                    tset = list(tset)
                    print(f"  抽样处理：源属性 {sset} -> 目标属性 {node}")
                    sample_Block_df, error_threshold = Generate_Sample(
                        data, sset, tset, models=models[node], single_max=single_max
                    )
                    sample_df = sample_Block_df.toPandas()
                    print(f"数据块大小: {sample_df.shape[0]}")

                    preCleaners = [single for single in singles if
                                   any(attr_set in sset + [node] for attr_set in single.domain)]
                    _, output, _, _ = customclean(sample_df, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight
                    partition = [list(model.source) for model in models[node]]
                    editRule1, output, editRuleList, _ = customclean(
                        output, cleaners=[blockModels], partition=partition, error_threshold=error_threshold
                    )
                    editRule *= editRule1
                else:
                    print(f"  当前节点: {node} 无可用清洗信号")

                editRuleLists.extend(editRuleList)
                editRules *= editRule  # 累积应用的清洗规则

                # 如果当前节点没有可用的清洗操作，标记为不需要再处理
                if not editRuleList:
                    level_clean_status[level_index][node] = True  # 标记该节点为无需再处理

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # 假设之前不是列表，现在重置为新列表
                else:
                    editRuleDict[str(node)] = editRuleList  # 键不存在，创建新键并赋值为新列表

            print(f"{level} 的清洗完成")
            print("\n本层累积的编辑规则数:", str(len(editRuleLists)))

            if editRuleLists:  # 有编辑规则
                nextClean = True
                while True:
                    # 处理数据，返回处理后的数据和剩余规则
                    data, remaining_rules = transformRulesToSpark(
                        editRuleLists, data, batch_size=batch_size, maxhandle=maxhandle
                    )
                    print("正在写入数据...")
                    next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                    data.write.mode('overwrite').saveAsTable(next_tmp_table)
                    print(f"写入数据大小：{data.count()}")
                    os.system('sync')  # 刷新缓存
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("重新读取最新清洗的数据...")
                    data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                    print(f"读入数据大小：{data.count()}")
                    data.persist()
                    current_tmp_table = next_tmp_table

                    if not remaining_rules:
                        print("所有规则处理完成")
                        break  # 所有规则处理完成，退出循环
                    print(f"有剩余的清洗操作没有应用，继续处理 {len(remaining_rules)} 个清洗操作")
                    editRuleLists = remaining_rules  # 使用剩余规则继续处理

                print(f"数据清洗后变动，重新进行节点 {level} 的清洗")
                # 不进行下一个 level 的清洗，重新在本 level 进行清洗
                break
            else:
                print("本层没有可以进行的清洗操作，进行下一层")

    print("\n清洗规则溯源分析:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print(group_weights)
    return data


def CleanonCloudRandom(spark, cleaners, data, table_name, database_name, batch_size=300, single_max=20000,
                       maxhandle=1000):
    # 初始化和分析清洗器
    print("初始化清洗器和分析依赖关系...")
    Edges, singles, multis = PreParamClassier(cleaners)
    source_sets, target_sets, explain, processing_order = discover_cleaner_associations(Edges)
    print("执行层级和目标模型分类...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("原始执行层级 (并行组):", levels)
    print("目标模型分类:", models)

    # 将 levels 反序以进行反向清洗
    levels = levels[::-1]
    nodes = nodes[::-1]
    print("反序后的执行层级 (并行组):", levels)

    editRuleDict = {}  # 用于统计溯源权重
    editRules = NOOP()  # 初始化无操作规则
    nextClean = True
    tmp_table1 = f"{database_name}.{table_name}_tmp1_random"
    tmp_table2 = f"{database_name}.{table_name}_tmp2_random"
    current_tmp_table = tmp_table1
    level_clean_status = {}  # 用于记录每个层级的节点清洗状态
    level_knowledge_status = {}  # 用于记录知识库是否已应用

    while nextClean:
        nextClean = False
        for level_index, level in enumerate(nodes):
            # 如果该层级还未记录过清洗状态，则初始化
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in level_knowledge_status:
                level_knowledge_status[level_index] = {node: False for node in level}

            print(f"\n处理反序后的第 {level_index + 1} 层级, 包含节点: {level}")
            editRuleLists = []  # 获取当前层级的清洗操作列表

            for node in level:
                # 只处理该层级中未完成清洗的节点
                if level_clean_status[level_index][node]:
                    print(f"  跳过已清洗结束的节点: {node}")
                    continue

                # 初始化当前节点的清洗操作
                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  当前节点: {node} 存在可用清洗信号")

                    # 检查知识库是否已应用
                    if not level_knowledge_status[level_index][node]:
                        # **在此处添加 transformKnowledgeToSpark 调用**
                        for cleaner in models[node]:
                            data = transformKnowledgeToSpark(spark, database_name, table_name, data, cleaner)

                        # **写入和读取数据**
                        print("正在写入数据...")
                        next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                        data.write.mode('overwrite').saveAsTable(next_tmp_table)
                        print(f"写入数据大小：{data.count()}")
                        os.system('sync')  # 刷新缓存
                        data.unpersist()
                        spark.catalog.clearCache()
                        print("重新读取最新清洗的数据...")
                        data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                        print(f"读入数据大小：{data.count()}")
                        data.persist()
                        current_tmp_table = next_tmp_table
                        level_knowledge_status[level_index][node] = True

                    # 定义一个空集合
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)  # 转换为列表
                    tset = list(tset)
                    print(f"  抽样处理：源属性 {sset} -> 目标属性 {node}")
                    sample_Block_df, error_threshold = Generate_Sample(
                        data, sset, tset, models=models[node], single_max=single_max
                    )
                    sample_df = sample_Block_df.toPandas()
                    print(f"数据块大小: {sample_df.shape[0]}")

                    preCleaners = [single for single in singles if
                                   any(attr_set in sset + [node] for attr_set in single.domain)]
                    _, output, _, _ = customclean(sample_df, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight
                    partition = [list(model.source) for model in models[node]]
                    editRule1, output, editRuleList, _ = customclean(
                        output, cleaners=[blockModels], partition=partition, error_threshold=error_threshold
                    )
                    editRule *= editRule1
                else:
                    print(f"  当前节点: {node} 无可用清洗信号")

                editRuleLists.extend(editRuleList)
                editRules *= editRule  # 累积应用的清洗规则

                # 如果当前节点没有可用的清洗操作，标记为不需要再处理
                if not editRuleList:
                    level_clean_status[level_index][node] = True  # 标记该节点为无需再处理

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList  # 假设之前不是列表，现在重置为新列表
                else:
                    editRuleDict[str(node)] = editRuleList  # 键不存在，创建新键并赋值为新列表

            print(f"{level} 的清洗完成")
            print("\n本层累积的编辑规则数:", str(len(editRuleLists)))

            if editRuleLists:  # 有编辑规则
                nextClean = True
                while True:
                    # 处理数据，返回处理后的数据和剩余规则
                    data, remaining_rules = transformRulesToSpark(
                        editRuleLists, data, batch_size=batch_size, maxhandle=maxhandle
                    )
                    print("正在写入数据...")
                    next_tmp_table = tmp_table2 if current_tmp_table == tmp_table1 else tmp_table1
                    data.write.mode('overwrite').saveAsTable(next_tmp_table)
                    print(f"写入数据大小：{data.count()}")
                    os.system('sync')  # 刷新缓存
                    data.unpersist()
                    spark.catalog.clearCache()
                    print("重新读取最新清洗的数据...")
                    data = spark.sql(f"SELECT * FROM {next_tmp_table}")
                    print(f"读入数据大小：{data.count()}")
                    data.persist()
                    current_tmp_table = next_tmp_table

                    if not remaining_rules:
                        print("所有规则处理完成")
                        break  # 所有规则处理完成，退出循环
                    print(f"有剩余的清洗操作没有应用，继续处理 {len(remaining_rules)} 个清洗操作")
                    editRuleLists = remaining_rules  # 使用剩余规则继续处理

                print(f"数据清洗后变动，重新进行节点 {level} 的清洗")
                # 不进行下一个 level 的清洗，重新在本 level 进行清洗
                break
            else:
                print("本层没有可以进行的清洗操作，进行下一层")

    print("\n清洗规则溯源分析:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print(group_weights)
    return data

def CleanonLocalWithnoSmple(spark, cleanners, data, table_path, batch_size=500, single_max=30000,
                            trace_path=None):
    current = 'Tmp1'
    print("初始化清洗器和分析依赖关系...")
    Edges, singles, multis = PreParamClassier(cleanners)
    trace = {
        "operators": [],
        "rules": [],
        "weights": [],
        "levels": [],
        "trace_path": trace_path or "",
    }

    print("处理单属性清洗算子：")
    for single in singles:
        print(f"处理单属性清洗算子：{single}")
        # 获取当前单属性的域 (例如: "domain" 属性表示当前算子的目标属性)
        target_column = single.domain

        print(f"开始清洗属性 {target_column} 的数据")
        # 将 Spark DataFrame 转换为 Pandas DataFrame
        df_pandas = data.select(target_column).distinct().toPandas()

        if df_pandas.empty:
            print(f"属性 {target_column} 的数据为空，跳过清洗。")
            continue

        print(f"属性 {target_column} 的数据大小: {df_pandas.shape[0]}")

        # 调用 customclean 进行清洗操作
        preCleaners = [single]
        _, output, editRuleList, _ = customclean(df_pandas, precleaners=preCleaners)
        trace["operators"].append(
            _operator_record(
                single,
                phase="single_attribute",
                order=len(trace["operators"]),
                sample_rows=int(df_pandas.shape[0]),
                rule_count=len(editRuleList or []),
            )
        )
        _append_rule_records(
            trace,
            editRuleList,
            phase="single_attribute",
            operator_id=_cleaner_id(single),
        )

        if editRuleList:
            # 如果有清洗操作规则，应用到 Spark DataFrame 上
            print(f"应用 {target_column} 的清洗规则，编辑规则数量: {len(editRuleList)}")
            data, _ = transformRulesToSpark(editRuleList, data, batch_size=batch_size)
        else:
            print(f"属性 {target_column} 没有可应用的清洗规则。")

    source_sets, target_sets, explain, processing_order, plt = discover_cleaner_associations(Edges)
    print("执行层级和目标模型分类...")
    levels, models, nodes = associations_classier(multis, source_sets, target_sets)
    print("执行层级 (并行组):", levels)
    print("目标模型分类:", models)
    trace["levels"] = [{"level": i, "nodes": [str(node) for node in level]} for i, level in enumerate(nodes)]

    editRuleDict = {}  # 用于统计溯源权重
    editRules = NOOP()  # 初始化无操作规则
    nextClean = True

    level_clean_status = {}
    clean_attempts = {}  # 新增：记录每个节点的清洗次数

    while nextClean:
        nextClean = False

        for level_index, level in enumerate(nodes):
            # 如果该层级还未记录过清洗状态，则初始化
            if level_index not in level_clean_status:
                level_clean_status[level_index] = {node: False for node in level}
            if level_index not in clean_attempts:
                clean_attempts[level_index] = {node: 0 for node in level}

            print(f"\n处理第 {level_index + 1} 层级, 包含节点: {level}")
            editRuleLists = []  # 获取当前层级的清洗操作列表

            for node in level:
                # 只处理该层级中挖掘不到操作的节点
                if level_clean_status[level_index][node]:
                    print(f"  跳过已清洗结束的节点: {node}")
                    continue

                clean_attempts[level_index][node] += 1  # 增加清洗次数

                if clean_attempts[level_index][node] > 3:  # 超过3次则标记为完成
                    print(f"  当前节点 {node} 清洗次数超过限制，标记为完成")
                    level_clean_status[level_index][node] = True
                    continue

                editRule = NOOP()
                editRuleList = []

                if node in models and models[node]:
                    print(f"  当前节点: {node} 存在可用清洗信号")
                    sset = set()
                    tset = set()
                    for m in models[node]:
                        sset = sset.union(m.source)
                        tset = tset.union(m.target)
                    sset = list(sset)
                    tset = list(tset)

                    # 合并节点（来自环检测，形如 'HospitalName,ProviderNumber'）需 split 还原为真实列；
                    # 单点节点直接当列名。这样既支持双向 FD（环），又避免把合并字符串当 Spark 列名 select。
                    node_cols = node.split(',') if isinstance(node, str) and ',' in node else [node]
                    select_cols = list(dict.fromkeys(['index'] + sset + node_cols))  # 保持顺序去重

                    # 将 Spark DataFrame 转换为 Pandas DataFrame
                    df_pandas = data.select(select_cols).toPandas()
                    print(f"数据块大小: {df_pandas.shape[0]}")
                    for model in models[node]:
                        trace["operators"].append(
                            _operator_record(
                                model,
                                phase="block_model",
                                order=len(trace["operators"]),
                                level=level_index,
                                node=node,
                                sample_rows=int(df_pandas.shape[0]),
                            )
                        )

                    preCleaners = [single for single in singles if single.domain in (sset + node_cols)]
                    _, output, _, _ = customclean(df_pandas, precleaners=preCleaners)

                    blockModels = None
                    weights = [1] * len(models[node])
                    for model, weight in zip(models[node], weights):
                        if blockModels is None:
                            blockModels = model * weight
                        else:
                            blockModels += model * weight

                    partition = [list(model.source) for model in models[node]]
                    print("分块属性:")
                    print(partition)
                    editRule1, output, editRuleList, _ = customclean(output, cleaners=[blockModels],
                                                                     partition=partition)
                    _append_rule_records(
                        trace,
                        editRuleList,
                        phase="block_model",
                        operator_id=f"block:{node}",
                        level=level_index,
                        node=node,
                    )
                    editRule *= editRule1
                else:
                    print(f"  当前节点: {node} 无可用清洗信号")

                editRuleLists.extend(editRuleList)
                editRules *= editRule

                if len(editRuleList) == 0:
                    print(f"  当前节点: {node} 无清洗操作")
                    level_clean_status[level_index][node] = True
                else:
                    print(f"  当前节点: {node} 有 {len(editRuleList)} 个清洗操作")

                if str(node) in editRuleDict:
                    if isinstance(editRuleDict[str(node)], list):
                        editRuleDict[str(node)].extend(editRuleList)
                    else:
                        editRuleDict[str(node)] = editRuleList
                else:
                    editRuleDict[str(node)] = editRuleList

            print(f"{level}的清洗完成")
            print("\n本层累积的编辑规则数:", str(len(editRuleLists)))

            if editRuleLists:
                nextClean = True
                data, _ = transformRulesToSpark(editRuleLists, data, batch_size=batch_size)
                break
            else:
                print("本层没有可以进行的清洗操作，进行下一层")
    print("\n清洗规则溯源分析:")
    grouped_opinfo = cleaner_grouping(nodes, models)
    single_opinfo = getSingle_opinfo(singles)
    group_weights = getOpWeights(editRuleDict, grouped_opinfo)
    print("\n清洗流程展示:")
    sorted_dict = sort_opinfo_by_weights(grouped_opinfo, group_weights)
    _append_weight_records(trace, grouped_opinfo, group_weights, sorted_dict)

    OpPlantuml = generate_plantuml_corrected(single_opinfo, grouped_opinfo, sorted_dict)
    PlantUML().process_str(OpPlantuml, directory=table_path)
    plt.savefig(rf"{table_path}/acplt.png", format="png")
    print("关联图存储到:", rf"{table_path}/acplt.png")
    _write_trace_files(trace_path, trace)

    return data
