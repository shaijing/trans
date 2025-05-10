from torchdata.nodes import IterableWrapper,Mapper, ParallelMapper, Loader

dp = IterableWrapper(range(10))
map_dp_1 = Mapper(dp, lambda x: x + 1)

node = IterableWrapper(range(10))
node = ParallelMapper(node, map_fn=lambda x: x**2, num_workers=3, method="thread")
loader = Loader(node)
result = list(loader)
print(result)