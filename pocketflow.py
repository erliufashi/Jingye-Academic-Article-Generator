
import asyncio, warnings, copy, time # 导入必要的库：异步IO、警告、对象复制、时间

class BaseNode:
    """
    所有节点类型的最基础类。
    定义了节点的基本属性（参数、后继节点）和核心执行流程（prep -> exec -> post）。
    """
    def __init__(self):
        """
        初始化节点。
        - self.params: 存储节点的特定配置参数。
        - self.successors: 存储节点的后继节点，以动作（action）名称为键。
                          'default' 是默认的动作名称。
        """
        self.params, self.successors = {}, {}

    def set_params(self, params):
        """
        设置或更新节点的参数。
        通常由 Flow 在执行节点前调用，传递 Flow 级别的参数或运行时参数。
        Args:
            params (dict): 要设置的参数字典。
        """
        self.params = params

    def add_successor(self, node, action="default"):
        """
        添加一个后继节点。
        Args:
            node (BaseNode): 要添加的后继节点实例。
            action (str): 触发转换到此后继节点的动作名称。默认为 "default"。
        Returns:
            BaseNode: 返回添加的后继节点，方便链式调用。
        """
        if action in self.successors:
            # 如果该动作已有关联的后继节点，发出警告，因为旧的会被覆盖。
            warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action] = node
        return node # 返回后继节点，允许 node1 >> node2 >> node3 这样的链式写法

    # --- 节点生命周期方法 (子类通常需要重写这些) ---
    def prep(self, shared):
        """
        准备阶段 (Preparation)。
        在执行主要逻辑（exec）之前调用。
        通常用于从 'shared' 数据中提取所需输入或进行预处理。
        Args:
            shared (dict): 流程共享数据字典。
        Returns:
            Any: 返回的数据将传递给 'exec' 方法。
        """
        pass # 默认无操作

    def exec(self, prep_res):
        """
        执行阶段 (Execution)。
        节点的核心逻辑所在。
        Args:
            prep_res: 'prep' 方法的返回值。
        Returns:
            Any: 返回的数据将传递给 'post' 方法。
                 对于 Flow 中的节点，此返回值也可能用作决定下一个节点的 'action'。
        """
        pass # 默认无操作

    def post(self, shared, prep_res, exec_res):
        """
        后处理阶段 (Post-processing)。
        在执行主要逻辑（exec）之后调用。
        通常用于处理 'exec' 的结果，更新 'shared' 数据，或决定下一步的动作。
        Args:
            shared (dict): 流程共享数据字典。
            prep_res: 'prep' 方法的返回值。
            exec_res: 'exec' 方法的返回值。
        Returns:
            str | None: 对于 Flow 中的节点，返回一个字符串 'action' 来决定下一个节点。
                        返回 None 或其他非字符串值通常表示流程结束或使用默认转换。
        """
        pass # 默认无操作

    # --- 内部执行逻辑 ---
    def _exec(self, prep_res):
        """
        内部执行方法，直接调用 exec。
        Node 类会重写这个方法以加入重试逻辑。
        Args:
            prep_res: 'prep' 方法的返回值。
        Returns:
            Any: 'exec' 方法的返回值。
        """
        return self.exec(prep_res)

    def _run(self, shared):
        """
        内部运行方法，封装了完整的 prep -> _exec -> post 流程。
        Flow 会调用这个方法来执行单个节点。
        Args:
            shared (dict): 流程共享数据字典。
        Returns:
            str | None: 'post' 方法的返回值，用于 Flow 决定下一步。
        """
        p = self.prep(shared)       # 执行准备阶段
        e = self._exec(p)           # 执行核心逻辑（可能包含重试）
        return self.post(shared, p, e) # 执行后处理阶段，并返回结果（通常是 action）

    def run(self, shared):
        """
        公开的运行方法，用于独立运行一个节点（不通过 Flow）。
        如果节点有后继节点，会发出警告，因为此方法不会执行它们。
        Args:
            shared (dict): 共享数据字典。
        Returns:
            str | None: 'post' 方法的返回值。
        """
        if self.successors:
            # 提示用户此方法不会自动执行后继节点，应使用 Flow 来管理流程。
            warnings.warn("Node won't run successors. Use Flow.")
        return self._run(shared) # 调用内部运行逻辑

    # --- 语法糖 (Syntactic Sugar) ---
    def __rshift__(self, other):
        """
        重载右移运算符 `>>`。
        使得 `node1 >> node2` 等同于 `node1.add_successor(node2, action="default")`。
        用于方便地添加默认后继节点。
        Args:
            other (BaseNode): 要添加为默认后继的节点。
        Returns:
            BaseNode: 返回添加的后继节点，支持链式调用。
        """
        return self.add_successor(other) # 添加默认后继

    def __sub__(self, action):
        """
        重载减法运算符 `-`。
        当与字符串类型的 action 一起使用时 (例如 `node - "action"`),
        返回一个 _ConditionalTransition 对象，用于下一步的 `>>` 操作。
        Args:
            action (str): 条件转换的动作名称。
        Returns:
            _ConditionalTransition: 用于条件转换的辅助对象。
        Raises:
            TypeError: 如果 action 不是字符串。
        """
        if isinstance(action, str):
            # 创建一个临时对象，存储源节点和动作名称
            return _ConditionalTransition(self, action)
        raise TypeError("Action must be a string") # 动作必须是字符串


class _ConditionalTransition:
    """
    辅助类，用于实现 `node - "action" >> next_node` 语法。
    它临时存储源节点和动作名称。
    """
    def __init__(self, src, action):
        """
        初始化条件转换对象。
        Args:
            src (BaseNode): 源节点。
            action (str): 触发转换的动作名称。
        """
        self.src, self.action = src, action

    def __rshift__(self, tgt):
        """
        重载右移运算符 `>>`。
        当对 _ConditionalTransition 对象使用 `>>` 时 (例如 `(node - "action") >> next_node`),
        将目标节点 `tgt` 添加为源节点 `src` 的、由 `action` 触发的后继节点。
        Args:
            tgt (BaseNode): 目标后继节点。
        Returns:
            BaseNode: 返回添加的目标节点 `tgt`，支持链式调用。
        """
        # 调用源节点的 add_successor 方法，传入目标节点和特定动作
        return self.src.add_successor(tgt, self.action)


class Node(BaseNode):
    """
    标准的同步节点，继承自 BaseNode。
    增加了执行失败时的自动重试和回退逻辑。
    """
    def __init__(self, max_retries=1, wait=0):
        """
        初始化 Node。
        Args:
            max_retries (int): 最大尝试次数（包括首次尝试）。默认为 1（不重试）。
            wait (int | float): 每次重试前等待的秒数。默认为 0（不等待）。
        """
        super().__init__() # 调用父类（BaseNode）的初始化方法
        self.max_retries, self.wait = max_retries, wait # 存储重试次数和等待时间
        self.cur_retry = 0 # 当前重试次数（内部状态）

    def exec_fallback(self, prep_res, exc):
        """
        执行回退逻辑 (Execution Fallback)。
        当 'exec' 方法在所有重试尝试后仍然失败时调用。
        默认行为是重新抛出最后的异常。子类可以重写此方法以实现自定义的回退处理。
        Args:
            prep_res: 'prep' 方法的返回值。
            exc (Exception): 最后一次尝试时捕获到的异常对象。
        Raises:
            Exception: 默认重新抛出传入的异常。
        """
        raise exc # 默认重新抛出异常

    def _exec(self, prep_res):
        """
        重写了 BaseNode 的内部执行方法，加入了重试循环。
        Args:
            prep_res: 'prep' 方法的返回值。
        Returns:
            Any: 如果 'exec' 成功，返回其结果；如果所有重试都失败，返回 'exec_fallback' 的结果。
        """
        for self.cur_retry in range(self.max_retries): # 循环尝试 max_retries 次
            try:
                # 尝试执行核心逻辑
                return self.exec(prep_res)
            except Exception as e:
                # 如果发生异常
                if self.cur_retry == self.max_retries - 1:
                    # 如果这是最后一次尝试，调用回退方法
                    return self.exec_fallback(prep_res, e)
                if self.wait > 0:
                    # 如果设置了等待时间，且不是最后一次尝试，则等待
                    time.sleep(self.wait)
        # 理论上不应该执行到这里，但为防万一返回 None
        return None # Added for robustness, though loop structure should prevent reaching here


class Flow(BaseNode):
    """
    流程编排器，继承自 BaseNode。
    管理一系列节点的执行顺序。
    它本身也可以像一个节点一样被嵌套在其他 Flow 中。
    """
    def __init__(self, start):
        """
        初始化 Flow。
        Args:
            start (BaseNode): 流程的起始节点。
        """
        super().__init__() # 调用父类（BaseNode）的初始化方法
        self.start = start # 设置起始节点

    def get_next_node(self, curr, action):
        """
        根据当前节点和返回的动作，获取下一个要执行的节点。
        Args:
            curr (BaseNode): 当前执行完成的节点。
            action (str | None): 当前节点 'post' 方法返回的动作名称。
        Returns:
            BaseNode | None: 下一个要执行的节点实例。如果没有找到对应的后继节点，返回 None。
        """
        # 优先使用返回的 action 查找后继，如果 action 为 None 或空，则使用 "default"
        nxt = curr.successors.get(action or "default")
        if not nxt and curr.successors and action != "default": # Check action explicitly
            # 如果根据 action 找不到后继节点，并且该节点确实定义了其他后继节点，
            # 发出警告，提示流程可能在此意外结束。
            # We check action != "default" to avoid warning when default is missing but action=None
            warnings.warn(f"Flow ends: Action '{action}' not found in successors {list(curr.successors.keys())} for node {type(curr).__name__}")
        return nxt # 返回找到的下一个节点，或 None

    def _orch(self, shared, params=None):
        """
        内部编排逻辑 (Orchestration)。
        驱动流程从起始节点开始，按顺序执行节点。
        Args:
            shared (dict): 流程共享数据字典。
            params (dict | None): 传递给每个节点的参数。如果为 None，使用 Flow 自身的参数。
        """
        # 复制起始节点，避免修改原始 Flow 定义中的节点状态
        curr = copy.copy(self.start)
        # 确定要传递给节点的参数，优先使用传入的 params，否则使用 Flow 自身的 params
        p = (params or {**self.params})

        while curr: # 当还有需要执行的节点时循环
            curr.set_params(p)         # 为当前节点设置参数
            c = curr._run(shared)      # 执行当前节点 (prep -> _exec -> post)
            # 获取下一个节点，注意再次 copy 以免影响 Flow 定义
            next_node_instance = self.get_next_node(curr, c)
            curr = copy.copy(next_node_instance) if next_node_instance else None # 更新当前节点为下一个节点

    def _run(self, shared):
        """
        内部运行方法，执行 Flow 的 prep -> _orch -> post 流程。
        Args:
            shared (dict): 流程共享数据字典。
        Returns:
            str | None: Flow 自身的 'post' 方法的返回值。
        """
        pr = self.prep(shared)       # 执行 Flow 自身的准备阶段 (可选)
        self._orch(shared)         # 执行流程编排，运行内部节点
        # 执行 Flow 自身的后处理阶段 (可选)，注意 exec_res 传 None，因为 Flow 不执行 exec
        return self.post(shared, pr, None)

    def exec(self, prep_res):
        """
        Flow 类不应该直接执行 'exec' 逻辑。它的核心是编排其他节点。
        调用此方法会引发错误。
        """
        raise RuntimeError("Flow can't exec. Its role is orchestration.")


class BatchNode(Node):
    """
    批量处理节点，继承自 Node。
    它的 'exec' 方法会对输入列表中的每个项目调用父类（Node）的 '_exec' 方法。
    """
    def _exec(self, items):
        """
        重写内部执行方法，以迭代处理输入列表。
        Args:
            items (list | None): 'prep' 方法返回的、需要批量处理的项目列表。
        Returns:
            list: 一个包含每个项目处理结果的列表。
        """
        # 对列表中的每个 item 调用父类（即 Node 或其子类）的 _exec 方法
        # 这个 super(BatchNode, self) 确保调用的是继承链中 BatchNode 上一层的 _exec
        # （通常是 Node 的 _exec，包含了重试逻辑）
        return [super(BatchNode, self)._exec(i) for i in (items or [])]


class BatchFlow(Flow):
    """
    批量处理流程，继承自 Flow。
    它的 'prep' 方法应该返回一个参数字典的列表。
    流程会对每个参数字典执行一次内部的编排逻辑 ('_orch')。
    """
    def _run(self, shared):
        """
        重写内部运行方法，以支持批量执行流程编排。
        """
        # 执行 Flow 自身的 prep，期望返回一个列表，其中每个元素是用于一次流程执行的参数字典
        pr = self.prep(shared) or []
        for bp in pr: # 遍历 prep 返回的参数字典列表
            # 对每个参数字典 bp，执行一次完整的流程编排 (_orch)
            # 传递给节点的参数是 Flow 的基础参数和当前批次的特定参数 bp 的合并
            self._orch(shared, {**self.params, **bp})
        # 执行 Flow 自身的 post，exec_res 仍然是 None
        return self.post(shared, pr, None)


class AsyncNode(Node):
    """
    异步节点基类，继承自 Node。
    将节点的生命周期方法 (prep, exec, post, exec_fallback) 转换为异步版本。
    """
    # --- 阻止调用同步方法 ---
    def prep(self, shared): raise RuntimeError("Use prep_async.")
    def exec(self, prep_res): raise RuntimeError("Use exec_async.")
    def post(self, shared, prep_res, exec_res): raise RuntimeError("Use post_async.")
    def exec_fallback(self, prep_res, exc): raise RuntimeError("Use exec_fallback_async.")
    def _run(self, shared): raise RuntimeError("Use run_async.") # 明确需要使用异步运行

    # --- 异步生命周期方法 (子类需要重写这些) ---
    async def prep_async(self, shared):
        """异步准备阶段。"""
        pass
    async def exec_async(self, prep_res):
        """异步执行阶段。"""
        pass
    async def exec_fallback_async(self, prep_res, exc):
        """异步执行回退逻辑。"""
        raise exc # 默认重新抛出异常
    async def post_async(self, shared, prep_res, exec_res):
        """异步后处理阶段。"""
        pass

    # --- 内部异步执行逻辑 ---
    async def _exec(self, prep_res):
        """
        内部异步执行方法，包含异步重试逻辑。
        """
        for i in range(self.max_retries): # 重试循环
            try:
                # 尝试异步执行核心逻辑
                return await self.exec_async(prep_res)
            except Exception as e:
                # 如果发生异常
                if i == self.max_retries - 1:
                    # 最后一次尝试失败，调用异步回退方法
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    # 如果需要等待，使用 asyncio.sleep 进行异步等待
                    await asyncio.sleep(self.wait)
        return None # Should not be reached

    async def run_async(self, shared):
        """
        公开的异步运行方法，用于独立运行异步节点。
        """
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared) # 调用内部异步运行逻辑

    async def _run_async(self, shared):
        """
        内部异步运行方法，封装了完整的异步 prep -> _exec -> post 流程。
        """
        p = await self.prep_async(shared)      # 异步准备
        e = await self._exec(p)                # 异步执行（含重试）
        return await self.post_async(shared, p, e) # 异步后处理

class AsyncBatchNode(AsyncNode, BatchNode):
    """
    异步批量处理节点。
    按顺序异步处理列表中的每一项。
    """
    async def _exec(self, items):
        """
        异步地、顺序地处理列表中的每一项。
        """
        # 使用列表推导式，但注意这里的 super 调用会进入 AsyncNode 的 _exec
        # 这里的 await 会让每个项的处理按顺序完成
        return [await super(AsyncBatchNode, self)._exec(i) for i in (items or [])]

class AsyncParallelBatchNode(AsyncNode, BatchNode):
    """
    异步并行批量处理节点。
    使用 asyncio.gather 同时处理列表中的所有项。
    """
    async def _exec(self, items):
        """
        异步地、并行地处理列表中的所有项。
        """
        # 使用 asyncio.gather 来并发执行所有项的 super()._exec 调用
        # super(AsyncParallelBatchNode, self) 会调用 AsyncNode 的 _exec
        tasks = [super(AsyncParallelBatchNode, self)._exec(i) for i in (items or [])]
        return await asyncio.gather(*tasks) # 等待所有并发任务完成

class AsyncFlow(Flow, AsyncNode):
    """
    异步流程编排器。
    可以管理同步和异步节点的混合执行。
    """
    async def _orch_async(self, shared, params=None):
        """
        内部异步编排逻辑。
        """
        curr, p = copy.copy(self.start), (params or {**self.params})
        while curr:
            curr.set_params(p)
            # 判断当前节点是否为异步节点
            if isinstance(curr, AsyncNode):
                # 如果是异步节点，调用其异步运行方法
                c = await curr._run_async(shared)
            else:
                # 如果是同步节点，调用其同步运行方法
                c = curr._run(shared)
            # 获取下一个节点，同样需要 copy
            next_node_instance = self.get_next_node(curr, c)
            curr = copy.copy(next_node_instance) if next_node_instance else None

    async def _run_async(self, shared):
        """
        内部异步运行方法，执行 AsyncFlow 的 prep -> _orch -> post 流程。
        """
        p = await self.prep_async(shared)          # 异步准备
        await self._orch_async(shared)            # 异步编排
        return await self.post_async(shared, p, None) # 异步后处理

class AsyncBatchFlow(AsyncFlow, BatchFlow):
    """
    异步批量处理流程。
    按顺序对 'prep_async' 返回的每个参数字典执行一次异步流程编排。
    """
    async def _run_async(self, shared):
        """
        顺序批量异步执行流程。
        """
        pr = await self.prep_async(shared) or [] # 异步获取批量参数列表
        for bp in pr:
            # 对每个参数字典，顺序地执行一次完整的异步流程编排
            await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None) # 异步后处理

class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    """
    异步并行批量处理流程。
    同时对 'prep_async' 返回的所有参数字典执行异步流程编排。
    """
    async def _run_async(self, shared):
        """
        并行批量异步执行流程。
        """
        pr = await self.prep_async(shared) or [] # 异步获取批量参数列表
        # 使用 asyncio.gather 并发执行所有批次的异步流程编排
        tasks = [self._orch_async(shared, {**self.params, **bp}) for bp in pr]
        await asyncio.gather(*tasks)
        return await self.post_async(shared, pr, None) # 异步后处理
