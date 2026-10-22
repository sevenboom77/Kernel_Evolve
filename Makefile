plan:
	python benchmarks/run.py --operator small-m --out outputs/small-m-plan.json
	python benchmarks/run.py --operator graph --out outputs/graph-plan.json

gpu:
	python benchmarks/run.py --operator small-m --execute --gpu 0 --out outputs/small-m-gpu.json
	python benchmarks/run.py --operator graph --execute --gpu 0 --out outputs/graph-gpu.json
